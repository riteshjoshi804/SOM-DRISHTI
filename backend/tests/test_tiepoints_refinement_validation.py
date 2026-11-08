"""Tie-point selection, sub-pixel refinement and the three kinds of validation evidence."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from conftest import lunar_like

from app.config import GeometryConfig, GuidedSearchConfig, HoldoutConfig, RefinementConfig, TiePointConfig, ValidationConfig
from app.evaluation.validation import (
    ControlPoint,
    evaluate_control_points,
    holdout_validation,
    local_refinement_check,
    paired_refinement_holdout,
    parse_control_points,
    stratified_split,
)
from app.geometry.estimation import apply_transform, fit_least_squares
from app.geometry.tiepoints import anms_radii, score_points, select_tie_points, suppress_duplicates
from app.geometry.verification import verify_candidates
from app.refinement.subpixel import (
    AMBIGUOUS,
    FLAT,
    INVALID,
    OUTSIDE,
    REFINED,
    _coherent_shift,
    refine_correspondences,
)

SIZE = (600, 1200)
TRUE = np.array([[0.98, -0.17, 300.0], [0.17, 0.98, 120.0], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------------- tie points
def test_duplicates_keep_the_best_point() -> None:
    points = np.array([[10.0, 10.0], [12.0, 10.0], [50.0, 50.0]])
    kept = suppress_duplicates(points, np.array([0.2, 0.9, 0.5]), 8.0)
    assert kept.tolist() == [1, 2]


def test_anms_favours_spread_over_a_strong_cluster() -> None:
    rng = np.random.default_rng(0)
    cluster = rng.normal([100, 100], 8, (200, 2))
    lone = np.array([[500.0, 900.0], [80.0, 1000.0]])
    points = np.vstack([cluster, lone])
    scores = np.r_[rng.uniform(0.8, 1.0, 200), [0.3, 0.3]]
    radii = anms_radii(points, scores)
    top = np.argsort(-radii)[:4]
    assert 200 in top and 201 in top


def test_score_discounts_residual_but_keeps_confidence_order() -> None:
    assert score_points(np.array([0.9]), np.array([0.0]), 3.0)[0] == pytest.approx(0.9)
    assert score_points(np.array([0.9]), np.array([3.0]), 3.0)[0] == pytest.approx(0.45)


def test_selection_thins_clusters_and_never_invents_points() -> None:
    rng = np.random.default_rng(1)
    valid = np.ones((1200, 600), bool)
    dense = rng.uniform([0, 0], [150, 150], (600, 2))  # all in one corner
    sparse = np.array([[500.0, 1100.0], [450.0, 700.0], [100.0, 900.0]])
    points = np.vstack([dense, sparse])
    selected, summary = select_tie_points(points, rng.uniform(0.2, 1.0, len(points)), rng.uniform(0, 3, len(points)), 3.0, valid, TiePointConfig(), None, 5.0)
    assert set(selected.tolist()) <= set(range(len(points)))
    for lone in (600, 601, 602):
        assert lone in selected
    # The dense corner covers two cells of the grid along x and one along y, capped per cell.
    assert summary.selected < 60 and summary.selected >= 6
    assert summary.occupied_cells <= 7
    assert summary.source_coverage == pytest.approx(summary.occupied_cells / summary.valid_cells)
    assert summary.tie_points_per_km2 is not None and summary.distribution_score is not None


def test_selection_flags_too_few_points_instead_of_filling_cells() -> None:
    valid = np.ones((1200, 600), bool)
    points = np.array([[10.0, 10.0], [500.0, 1000.0]])
    selected, summary = select_tie_points(points, np.array([0.9, 0.9]), np.zeros(2), 3.0, valid, TiePointConfig(min_tie_points=12), None)
    assert len(selected) == 2 and summary.below_minimum


def test_overlap_coverage_ignores_cells_that_cannot_match() -> None:
    valid = np.ones((800, 800), bool)
    overlap = np.zeros((800, 800), bool)
    overlap[:, :400] = True
    rng = np.random.default_rng(2)
    points = rng.uniform([0, 0], [400, 800], (400, 2))
    _, summary = select_tie_points(points, np.full(400, 0.9), np.zeros(400), 3.0, valid, TiePointConfig(), overlap)
    assert summary.overlap_cells == 32 and summary.overlap_coverage == pytest.approx(1.0)
    assert summary.source_coverage == pytest.approx(0.5)


def test_distribution_score_is_higher_when_points_are_even() -> None:
    valid = np.ones((800, 800), bool)
    xx, yy = np.meshgrid(np.arange(50, 800, 100), np.arange(50, 800, 100))
    even = np.c_[xx.ravel(), yy.ravel()].astype(float)
    uneven = np.vstack([even[:4], np.random.default_rng(3).uniform(0, 100, (60, 2))])
    _, a = select_tie_points(even, np.full(len(even), 0.9), np.zeros(len(even)), 3.0, valid, TiePointConfig(min_separation_px=0))
    _, b = select_tie_points(uneven, np.full(len(uneven), 0.9), np.zeros(len(uneven)), 3.0, valid, TiePointConfig(min_separation_px=0))
    assert a.distribution_score == pytest.approx(1.0)
    assert b.distribution_score < a.distribution_score


def test_selection_respects_maximum() -> None:
    rng = np.random.default_rng(4)
    points = rng.uniform([0, 0], [600, 1200], (800, 2))
    selected, _ = select_tie_points(points, rng.uniform(0.2, 1, 800), np.zeros(800), 3.0, np.ones((1200, 600), bool), TiePointConfig(max_tie_points=50))
    assert len(selected) == 50


# --------------------------------------------------------------------------------- refinement
@pytest.fixture(scope="module")
def textured():
    return (lunar_like((500, 500), 11) * 250 + 2).astype(np.float32)


def _shifted_pair(textured: np.ndarray, dx: float, dy: float):
    warp = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    moved = cv2.warpAffine(textured, warp, (500, 500), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return np.clip(np.rint(textured), 0, 255).astype(np.uint8), np.clip(np.rint(moved), 0, 255).astype(np.uint8)


def _identity(points: np.ndarray) -> np.ndarray:
    return np.repeat(np.eye(2)[None], len(points), axis=0)


def _points() -> np.ndarray:
    xx, yy = np.meshgrid(np.arange(60, 460, 60), np.arange(60, 460, 60))
    return np.c_[xx.ravel(), yy.ravel()].astype(np.float64)


# Measured median errors differ by estimator: phase correlation is clearly the least accurate here
# (about 0.3 px on 31 px patches), so it has its own limit rather than a shared, looser one.
@pytest.mark.parametrize("strategy,limit", [("ncc", 0.25), ("lucas_kanade", 0.25), ("ecc", 0.25), ("phase", 0.5)])
def test_every_strategy_recovers_a_known_subpixel_shift(textured, strategy: str, limit: float) -> None:
    source, reference = _shifted_pair(textured, 1.37, -0.64)
    src = _points()
    start = src.copy()  # initial reference points deliberately sit 1.5 px from the truth
    cfg = RefinementConfig(strategy=strategy, search_radius_px=4)
    mask = np.ones(source.shape, np.uint8)
    out = refine_correspondences(source, mask, reference, mask, src, start, _identity, cfg)
    assert out.summary.refined >= 0.7 * len(src), out.summary.rejected_by_reason
    truth = src + np.array([1.37, -0.64])
    err_before = np.linalg.norm(start[out.refined] - truth[out.refined], axis=1)
    err_after = np.linalg.norm(out.reference_refined[out.refined] - truth[out.refined], axis=1)
    assert np.median(err_after) < limit, f"{strategy}: {np.median(err_after):.3f}"
    assert np.median(err_after) < np.median(err_before)


def test_refinement_keeps_initial_coordinates_separately(textured) -> None:
    source, reference = _shifted_pair(textured, 0.5, 0.5)
    src = _points()
    start = src + 0.2
    mask = np.ones(source.shape, np.uint8)
    out = refine_correspondences(source, mask, reference, mask, src, start, _identity, RefinementConfig())
    np.testing.assert_array_equal(out.reference_initial, start)
    np.testing.assert_array_equal(out.source, src)
    np.testing.assert_allclose(out.displacement, out.reference_refined - out.reference_initial)


def test_refinement_rejects_rather_than_guesses(textured) -> None:
    source, reference = _shifted_pair(textured, 0.0, 0.0)
    mask = np.ones(source.shape, np.uint8)
    cfg = RefinementConfig()

    outside = refine_correspondences(source, mask, reference, mask, np.array([[2.0, 2.0]]), np.array([[2.0, 2.0]]), _identity, cfg)
    assert outside.reason == [OUTSIDE] and not outside.refined[0]

    holes = mask.copy()
    holes[95:105, 95:105] = 0
    invalid = refine_correspondences(source, holes, reference, mask, np.array([[100.0, 100.0]]), np.array([[100.0, 100.0]]), _identity, cfg)
    assert invalid.reason == [INVALID]

    flat = np.full(source.shape, 120, np.uint8)
    no_texture = refine_correspondences(flat, mask, reference, mask, np.array([[100.0, 100.0]]), np.array([[100.0, 100.0]]), _identity, cfg)
    assert no_texture.reason == [FLAT]
    # A rejected point keeps exactly its initial coordinates.
    np.testing.assert_array_equal(no_texture.reference_refined, [[100.0, 100.0]])


def test_refinement_rejects_ambiguous_repeating_pattern() -> None:
    xx = np.arange(400, dtype=np.float32)
    pattern = np.tile(127 + 100 * np.sin(xx * 2 * np.pi / 8.0), (400, 1)).astype(np.uint8)
    mask = np.ones(pattern.shape, np.uint8)
    cfg = RefinementConfig(strategy="ncc", search_radius_px=12)
    out = refine_correspondences(pattern, mask, pattern, mask, np.array([[200.0, 200.0]]), np.array([[201.0, 200.0]]), _identity, cfg)
    assert not out.refined[0] and out.reason[0] != REFINED
    assert out.reason[0] in (AMBIGUOUS, "optimum at the search limit", "correlation surface has no clear peak")


def test_coherent_shift_is_flagged_and_noise_is_not() -> None:
    """Regression: a Sun change pulled points 4.5 px one way while correlation scores stayed high."""
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 0.4, (200, 2))
    assert _coherent_shift(noise)[2] is False
    pulled = noise + np.array([4.5, -0.1])
    x, y, flagged = _coherent_shift(pulled)
    assert flagged and x == pytest.approx(4.5, abs=0.1) and y == pytest.approx(-0.1, abs=0.1)
    assert _coherent_shift(pulled[:3]) == (None, None, False)  # too few to say


def test_refining_a_consistent_error_is_reported_as_systematic(textured) -> None:
    source, reference = _shifted_pair(textured, 2.0, 0.0)
    src = _points()
    mask = np.ones(source.shape, np.uint8)
    out = refine_correspondences(source, mask, reference, mask, src, src.copy(), _identity, RefinementConfig())
    assert out.summary.systematic_shift and out.summary.median_shift_x_px == pytest.approx(2.0, abs=0.2)


def test_low_correlation_is_not_refined(textured) -> None:
    other = (lunar_like((500, 500), 12) * 250 + 2).astype(np.uint8)
    source, _ = _shifted_pair(textured, 0.0, 0.0)
    mask = np.ones(source.shape, np.uint8)
    src = _points()
    out = refine_correspondences(source, mask, other, mask, src, src.copy(), _identity, RefinementConfig(min_score=0.6))
    assert out.summary.refined < 0.2 * len(src)


def test_refinement_uses_the_local_frame_for_rotated_pairs(textured) -> None:
    angle = np.radians(25.0)
    linear = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    forward = np.array([[linear[0, 0], linear[0, 1], 150.0], [linear[1, 0], linear[1, 1], -40.0]])
    source = np.clip(np.rint(textured), 0, 255).astype(np.uint8)
    reference = cv2.warpAffine(source, forward, (700, 700), flags=cv2.INTER_CUBIC)
    src = np.array([[200.0, 200.0], [260.0, 240.0], [300.0, 180.0]])
    truth = (forward[:, :2] @ src.T).T + forward[:, 2]
    start = truth + np.array([1.1, -0.8])
    mask_s, mask_r = np.ones(source.shape, np.uint8), np.ones(reference.shape, np.uint8)
    out = refine_correspondences(
        source, mask_s, reference, mask_r, src, start, lambda p: np.repeat(linear[None], len(p), axis=0), RefinementConfig()
    )
    assert out.refined.all(), out.reason
    assert np.linalg.norm(out.reference_refined - truth, axis=1).max() < 0.3


# --------------------------------------------------------------------------------- validation
def _candidates(n_in: int, n_out: int, seed: int = 0, noise: float = 0.4):
    rng = np.random.default_rng(seed)
    src_in = rng.uniform([0, 0], SIZE, (n_in, 2))
    dst_in = apply_transform(TRUE, src_in) + rng.normal(0, noise, (n_in, 2))
    src_out = rng.uniform([0, 0], SIZE, (n_out, 2))
    dst_out = rng.uniform([0, 0], (1600, 1800), (n_out, 2))
    return np.vstack([src_in, src_out]), np.vstack([dst_in, dst_out]), rng.uniform(0.1, 0.9, n_in + n_out)


def test_stratified_split_spans_the_image() -> None:
    rng = np.random.default_rng(1)
    points = rng.uniform([0, 0], SIZE, (2000, 2))
    held = stratified_split(points, SIZE, 8, 8, 0.3, np.random.default_rng(2))
    assert 0.25 < held.mean() < 0.35
    # Every part of the image contributes held-out points.
    rows = (points[held, 1] / SIZE[1] * 8).astype(int)
    columns = (points[held, 0] / SIZE[0] * 8).astype(int)
    assert len(np.unique(rows * 8 + columns)) == 64


def test_lone_point_in_a_cell_stays_in_the_fit() -> None:
    held = stratified_split(np.array([[10.0, 10.0]]), SIZE, 8, 8, 0.9, np.random.default_rng(0))
    assert not held.any()


def test_holdout_measures_points_outside_the_fit() -> None:
    src, dst, rank = _candidates(400, 200)
    final, log = verify_candidates(src, dst, rank, SIZE, (1600, 1800), GeometryConfig(), ValidationConfig(), None)
    assert final is not None, log
    result = holdout_validation(src, dst, rank, final, SIZE, (1600, 1800), GeometryConfig(), ValidationConfig(), GuidedSearchConfig(), HoldoutConfig(), 1600.0 * 1800.0)
    assert result.status == "ok" and result.repeats_completed == 5
    assert "not independent ground truth" in result.label
    assert result.heldout_transfer_rmse_px is not None and result.heldout_transfer_rmse_px < 1.5
    assert result.heldout_median_px < 1.0 and result.heldout_p90_px < 1.5
    assert result.heldout_within_gate > 0 and result.heldout_candidates > result.heldout_within_gate
    assert result.block_rmse_px is not None and result.block_count >= 4


def test_holdout_reports_when_it_cannot_measure() -> None:
    src, dst, rank = _candidates(10, 0)
    final, _ = verify_candidates(src, dst, rank, SIZE, (1600, 1800), GeometryConfig(), ValidationConfig(), None)
    assert final is not None
    result = holdout_validation(src, dst, rank, final, SIZE, (1600, 1800), GeometryConfig(), ValidationConfig(), None, HoldoutConfig(min_points=50), 1e6)
    assert result.status == "insufficient" and result.reason


def test_paired_comparison_detects_real_noise_reduction() -> None:
    rng = np.random.default_rng(5)
    src = rng.uniform([0, 0], SIZE, (600, 2))
    truth = apply_transform(TRUE, src)
    noisy = truth + rng.normal(0, 1.0, truth.shape)
    cleaner = truth + rng.normal(0, 0.2, truth.shape)

    def fit(a, b):
        matrix = fit_least_squares(a, b, "affine")
        return lambda p: apply_transform(matrix, p)

    report = paired_refinement_holdout(src, noisy, cleaner, "affine", SIZE, fit, HoldoutConfig())
    assert report["status"] == "ok" and report["improved"] is True
    assert report["refined_median_px"] < report["initial_median_px"]
    worse = paired_refinement_holdout(src, cleaner, noisy, "affine", SIZE, fit, HoldoutConfig())
    assert worse["improved"] is False


def test_local_check_sees_refinement_that_a_large_scale_misfit_hides() -> None:
    """Regression: on a real strip the global model's misfit swamped the change in point noise."""
    rng = np.random.default_rng(6)
    src = rng.uniform([0, 0], SIZE, (900, 2))
    truth = apply_transform(TRUE, src)
    truth[:, 1] += 25.0 * np.sin(src[:, 1] / 120.0)  # a deviation no global affine can follow
    noisy, cleaner = truth + rng.normal(0, 1.0, truth.shape), truth + rng.normal(0, 0.2, truth.shape)
    report = local_refinement_check(src, noisy, cleaner, SIZE, block_px=150.0)
    assert report["status"] == "ok" and report["blocks"] >= 4
    assert report["refined_median_px"] < report["initial_median_px"]
    assert local_refinement_check(src[:5], noisy[:5], cleaner[:5], SIZE)["status"] == "insufficient"


def test_control_points_measure_a_finished_model_without_fitting() -> None:
    points = [ControlPoint(source_x=x, source_y=y, reference_x=0.0, reference_y=0.0) for x, y in ((10, 10), (100, 500), (500, 1000))]
    for p in points:
        p.reference_x, p.reference_y = apply_transform(TRUE, np.array([[p.source_x, p.source_y]]))[0] + np.array([3.0, 4.0])
    result = evaluate_control_points(points, lambda q: apply_transform(TRUE, q), "test fixture", SIZE, (1600, 1800))
    assert result.rmse_px == pytest.approx(5.0) and result.median_px == pytest.approx(5.0)
    assert "independent" in result.label and result.provenance == "test fixture"
    assert any("little weight" in w for w in result.warnings)
    assert result.per_point[0]["error_px"] == pytest.approx(5.0)


def test_control_point_warnings_and_csv() -> None:
    parsed = parse_control_points("source_x,source_y,reference_x,reference_y,id,note\n1,2,3,4,a,first\n5000,6,7,8,,\n")
    assert len(parsed) == 2 and parsed[0].id == "a" and parsed[0].note == "first" and parsed[1].id is None
    result = evaluate_control_points(parsed, lambda q: q, "csv", SIZE, (1600, 1800))
    assert any("outside the source" in w for w in result.warnings)
    with pytest.raises(ValueError):
        parse_control_points("a,b\n1,2\n")
    with pytest.raises(ValueError, match="line 2"):
        parse_control_points("source_x,source_y,reference_x,reference_y\nx,2,3,4\n")
    with pytest.raises(ValueError):
        evaluate_control_points([], lambda q: q, "none")
