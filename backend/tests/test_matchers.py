"""The matchers themselves, the common interface and the guided search, on images with a known answer."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import lunar_like

from app.config import GeometryConfig, GuidedSearchConfig, PipelineConfig, ValidationConfig
from app.evaluation.synthetic import crater_terrain, render
from app.geometry.estimation import apply_transform
from app.matching import guided as guided_module
from app.matching.base import LOFTR, RIFT, SIFT, SUPERPOINT_LIGHTGLUE, MatcherStatus, PairMatcher, PairMatches
from app.matching.guided import (
    Hypothesis,
    ImageView,
    build_hypotheses,
    guided_search,
    match_tiles,
    plan_tile_centres,
    rotation_matrix_about,
    run_decoy_controls,
    working_scale,
)
from app.matching.models import ModelUnavailable, weights_present
from app.matching.registry import SiftPairMatcher, create_pair_matcher, matcher_statuses, runnable_matchers
from app.matching.selection import CRITERIA, Candidate, model_disagreement, select
from app.metadata.overlap import OverlapEstimate

needs_superpoint = pytest.mark.skipif(not (weights_present("superpoint") and weights_present("lightglue")), reason="SuperPoint / LightGlue weights not downloaded")
needs_loftr = pytest.mark.skipif(not weights_present("loftr"), reason="LoFTR weights not downloaded")


@pytest.fixture(scope="module")
def terrain():
    return crater_terrain((640, 640), 3)


@pytest.fixture(scope="module")
def image(terrain):
    return (render(terrain, 40.0, 25.0) * 254 + 1).astype(np.uint8)


def _shifted(image: np.ndarray, dx: int, dy: int) -> tuple[np.ndarray, np.ndarray]:
    """Two crops of one image: a point (x, y) in the first is (x + dx, y + dy) in the second."""
    size = 480
    return image[40 : 40 + size, 40 : 40 + size], image[40 - dy : 40 - dy + size, 40 - dx : 40 - dx + size]


# ---------------------------------------------------------------------------- registry and interface
def test_statuses_report_every_matcher_honestly() -> None:
    statuses = {s.name: s for s in matcher_statuses()}
    assert set(statuses) == {SIFT, SUPERPOINT_LIGHTGLUE, LOFTR, RIFT}
    assert statuses[SIFT].available and statuses[SIFT].weights == []
    assert statuses[RIFT].experimental
    for name in (SUPERPOINT_LIGHTGLUE, LOFTR):
        assert isinstance(statuses[name], MatcherStatus) and statuses[name].weights


def test_auto_never_includes_the_experimental_matcher() -> None:
    assert RIFT not in runnable_matchers() and SIFT in runnable_matchers()


def test_missing_weights_are_reported_not_faked(monkeypatch, tmp_path) -> None:
    from app import config

    monkeypatch.setattr(config, "MODELS_ROOT", tmp_path)
    cfg = PipelineConfig(allow_model_download=False)
    statuses = {s.name: s for s in matcher_statuses(cfg)}
    assert not statuses[LOFTR].available and "not downloaded" in statuses[LOFTR].reason
    assert not statuses[SUPERPOINT_LIGHTGLUE].available
    assert statuses[SIFT].available  # the classical baseline needs no weights


def test_missing_dependency_is_reported(monkeypatch) -> None:
    from app.matching import registry

    monkeypatch.setattr(registry, "dependency_missing", lambda *modules: "kornia")
    with pytest.raises(ModelUnavailable, match="kornia"):
        create_pair_matcher(LOFTR, PipelineConfig())
    assert not {s.name: s for s in matcher_statuses()}[LOFTR].available


def test_unknown_matcher_is_refused() -> None:
    with pytest.raises(ValueError):
        create_pair_matcher("NOT_A_MATCHER")


@pytest.mark.parametrize("name", [SIFT, RIFT, pytest.param(SUPERPOINT_LIGHTGLUE, marks=needs_superpoint), pytest.param(LOFTR, marks=needs_loftr)])
def test_every_matcher_follows_the_same_contract(name: str, image: np.ndarray) -> None:
    matcher = create_pair_matcher(name, PipelineConfig())
    assert isinstance(matcher, PairMatcher) and matcher.name == name and isinstance(matcher.device, str)
    a, b = _shifted(image, 37, -21)
    mask = np.ones(a.shape, np.uint8)
    out = matcher.match_pair(a, b, mask, mask)
    assert isinstance(out, PairMatches)
    assert out.points0.shape == (len(out), 2) and out.points1.shape == (len(out), 2) and out.confidence.shape == (len(out),)
    assert out.points0.dtype == np.float64 and out.candidates >= len(out)
    assert len(out) > 0, f"{name} found nothing on a pair that differs only by a translation"


@pytest.mark.parametrize(
    "name,minimum,limit",
    [
        (SIFT, 30, 1.0),
        pytest.param(SUPERPOINT_LIGHTGLUE, 30, 2.0, marks=needs_superpoint),
        pytest.param(LOFTR, 100, 1.5, marks=needs_loftr),
    ],
)
def test_matches_land_where_they_should(name: str, minimum: int, limit: float, image: np.ndarray) -> None:
    matcher = create_pair_matcher(name, PipelineConfig())
    a, b = _shifted(image, 37, -21)
    mask = np.ones(a.shape, np.uint8)
    out = matcher.match_pair(a, b, mask, mask)
    errors = np.linalg.norm(out.points0 + [37, -21] - out.points1, axis=1)
    assert len(out) >= minimum
    assert np.median(errors) < limit, f"{name}: median {np.median(errors):.2f} px over {len(out)} matches"


@needs_loftr
def test_loftr_half_pixel_correction_leaves_no_bias(image: np.ndarray) -> None:
    """Without the -0.5 conversion both images share the offset, so the check is on absolute position."""
    matcher = create_pair_matcher(LOFTR, PipelineConfig())
    a, b = _shifted(image, 16, 24)
    out = matcher.match_pair(a, b, np.ones(a.shape, np.uint8), np.ones(b.shape, np.uint8))
    signed = (out.points1 - out.points0) - np.array([16.0, 24.0])
    good = np.linalg.norm(signed, axis=1) < 3.0
    assert good.sum() > 100
    assert np.abs(np.median(signed[good], axis=0)).max() < 0.3
    # Coarse keypoints sit on the 1/8 grid minus the half-pixel conversion.
    np.testing.assert_allclose((out.points0[good] + 0.5) % 8.0, 0.0, atol=1e-6)


@needs_loftr
def test_loftr_masks_exclude_invalid_pixels(image: np.ndarray) -> None:
    matcher = create_pair_matcher(LOFTR, PipelineConfig())
    a, b = _shifted(image, 16, 24)
    mask = np.ones(a.shape, np.uint8)
    mask[:, :240] = 0
    out = matcher.match_pair(a, b, mask, np.ones(b.shape, np.uint8))
    assert len(out) > 0 and out.points0[:, 0].min() >= 239.0


@needs_loftr
def test_loftr_handles_different_sizes_and_too_small_input(image: np.ndarray) -> None:
    matcher = create_pair_matcher(LOFTR, PipelineConfig())
    out = matcher.match_pair(image[:300, :400], image[:520, :610])
    assert len(out) > 0
    assert len(matcher.match_pair(image[:32, :32], image[:400, :400])) == 0


@needs_superpoint
def test_superpoint_extraction_is_tiled_and_masked(image: np.ndarray) -> None:
    matcher = create_pair_matcher(SUPERPOINT_LIGHTGLUE, PipelineConfig())
    mask = np.zeros(image.shape, np.uint8)
    mask[100:400, 100:400] = 1
    points, scores, descriptors = matcher.extract(image, mask)
    assert len(points) > 20 and descriptors.shape == (len(points), 256) and scores.shape == (len(points),)
    assert points.min() >= 100 and points.max() < 400
    assert np.all(np.diff(scores) <= 1e-6)
    # Tiled extraction finds keypoints across a large image without duplicates at tile seams.
    tiled, _, _ = matcher.extract(np.tile(image, (2, 2)), None, max_keypoints=0)
    assert len(np.unique(tiled, axis=0)) == len(tiled)


def test_rift_matcher_recovers_a_translation(image: np.ndarray) -> None:
    out = create_pair_matcher(RIFT, PipelineConfig()).match_pair(*_shifted(image, 16, 24))
    errors = np.linalg.norm(out.points0 + [16, 24] - out.points1, axis=1)
    assert len(out) >= 20 and np.mean(errors < 3.0) > 0.5


def test_sift_pair_matcher_keeps_lowe_ratio_semantics(image: np.ndarray) -> None:
    a, b = _shifted(image, 37, -21)
    strict = PipelineConfig()
    strict.match.ratio_threshold = 0.5
    loose = PipelineConfig()
    loose.match.ratio_threshold = 0.95
    assert len(SiftPairMatcher(strict).match_pair(a, b)) < len(SiftPairMatcher(loose).match_pair(a, b))


# ---------------------------------------------------------------------------- search helpers
def test_rotation_about_the_centre_keeps_the_footprint() -> None:
    rotate = rotation_matrix_about(600, 5000, 180.0, False)
    corners = np.array([[0, 0], [599, 4999]], float)
    np.testing.assert_allclose(apply_transform(rotate, corners), corners[::-1], atol=1e-6)
    mirror = rotation_matrix_about(600, 5000, 0.0, True)
    assert np.linalg.det(mirror[:2, :2]) < 0


def test_metadata_gives_two_orientation_hypotheses_and_no_metadata_a_sweep() -> None:
    prior = OverlapEstimate(mode="metadata", reason="t", prior_affine=[[1, 0, 10], [0, 1, 20]], prior_mirrored=False)
    hypotheses = build_hypotheses(prior, (100, 500), (1000, 1000), GuidedSearchConfig())
    assert [h.from_metadata for h in hypotheses] == [True, True] and "180" in hypotheses[1].name
    sweep = build_hypotheses(OverlapEstimate(mode="full_image", reason="t"), (100, 500), (1000, 1000), GuidedSearchConfig())
    assert len(sweep) == 16 and not any(h.from_metadata for h in sweep) and sum(h.mirrored for h in sweep) == 8
    limited = build_hypotheses(OverlapEstimate(mode="full_image", reason="t"), (100, 500), (1000, 1000), GuidedSearchConfig(max_orientation_hypotheses=4))
    assert len(limited) == 4


def test_narrow_strips_are_enlarged_but_ordinary_ones_are_not() -> None:
    assert working_scale(0.5, 640.0, 160.0) == 0.5
    assert working_scale(0.5, 107.0, 160.0) == pytest.approx(160.0 / 107.0)
    assert working_scale(1.0, 640.0, 200.0) == 1.0
    assert working_scale(1.0, 20.0, 200.0) == 2.0


def test_decoy_windows_never_touch_true_ground() -> None:
    """Regression: clipped windows landed back on true ground, and 124 true matches posed as a null result."""
    assert guided_module._strict_window(np.array([50.0, 50.0]), 200.0, (1000, 1000)) is None
    assert guided_module._strict_window(np.array([500.0, 500.0]), 200.0, (1000, 1000)) == (400, 400, 600, 600)
    assert guided_module._window_touches((400, 400, 600, 600), np.array([500.0, 500.0]), 200.0)
    assert not guided_module._window_touches((400, 400, 600, 600), np.array([900.0, 900.0]), 200.0)

    reference = ImageView(np.zeros((2000, 2000), np.uint8), np.ones((2000, 2000), np.uint8))
    centres = np.array([[100.0, 200.0], [100.0, 800.0]])
    predict = lambda p: np.asarray(p) + 500.0
    shifts = guided_module._scattered_shifts(centres, predict, reference, 300.0, 400.0, 16.0, np.random.default_rng(0))
    assert np.all(np.isfinite(shifts))
    for centre, shift in zip(predict(centres), shifts):
        box = guided_module._strict_window(centre + shift, 432.0, reference.size)
        assert box is not None and not guided_module._window_touches(box, centre, 300.0)


def test_decoy_pass_skips_windows_outside_the_raster(image: np.ndarray) -> None:
    class Counting:
        name, device = "COUNT", "cpu"
        pairs = 0

        def match_pair(self, a, b, ma=None, mb=None):
            Counting.pairs += 1
            return PairMatches(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0), 0)

    view = ImageView(image, np.ones(image.shape, np.uint8))
    guided = GuidedSearchConfig(fine_tile_px=128, fine_margin_px=16, fine_jitter_px=0, appearance_filter=False)
    centres = np.array([[100.0, 100.0], [300.0, 300.0]])
    shifts = np.array([[5000.0, 0.0], [np.nan, np.nan]])
    out = match_tiles(Counting(), view, view, lambda p: np.asarray(p), lambda c: np.eye(2), centres, guided, scale=1.0, margin_px=16.0, rng=np.random.default_rng(0), shifts=shifts)
    assert Counting.pairs == 0 and all("skipped" in t for t in out.tiles) and len(out.points0) == 0


# ---------------------------------------------------------------------------- AUTO
def _cand(name: str, status: str, **values) -> Candidate:
    base = {"coverage_within_overlap": 0.3, "heldout_median_px": 1.0, "inlier_ratio": 0.2, "reprojection_rmse_px": 1.0, "seconds": 60.0}
    return Candidate(name, status, {**base, **values})


def test_auto_does_not_choose_by_match_count() -> None:
    many = _cand("MANY", "LOW_CONFIDENCE", coverage_within_overlap=0.2, inliers_for_information_only=5000)
    spread = _cand("SPREAD", "LOW_CONFIDENCE", coverage_within_overlap=0.5, inliers_for_information_only=60)
    assert select([many, spread]).selected == "SPREAD"
    assert all(name != "inliers_for_information_only" for name, _, _ in CRITERIA)


def test_status_outranks_everything_and_failures_are_never_selected() -> None:
    good = _cand("GOOD", "SUCCESS", coverage_within_overlap=0.3)
    better_looking = _cand("LOW", "LOW_CONFIDENCE", coverage_within_overlap=0.9)
    failed = _cand("BAD", "FAILED", coverage_within_overlap=1.0)
    summary = select([failed, better_looking, good])
    assert summary.selected == "GOOD" and any("status" in r for r in summary.reasons)
    assert select([failed]).selected is None


def test_small_differences_do_not_decide() -> None:
    a = _cand("A", "SUCCESS", coverage_within_overlap=0.50, heldout_median_px=1.0)
    b = _cand("B", "SUCCESS", coverage_within_overlap=0.52, heldout_median_px=0.5)
    summary = select([a, b])
    assert summary.selected == "B" and "heldout_median_px" in summary.reasons[0]


def test_unmeasured_properties_cannot_decide() -> None:
    a = _cand("A", "SUCCESS", heldout_median_px=None, inlier_ratio=0.1)
    b = _cand("B", "SUCCESS", heldout_median_px=0.5, inlier_ratio=0.5)
    assert "inlier_ratio" in select([a, b]).reasons[0]


def test_disagreement_is_measured_at_the_given_points() -> None:
    a = lambda p: np.asarray(p, float)
    b = lambda p: np.asarray(p, float) + [3.0, 4.0]
    report = model_disagreement(a, b, np.zeros((10, 2)))
    assert report["median_px"] == pytest.approx(5.0) and report["points"] == 10
    assert model_disagreement(a, b, np.zeros((0, 2))) is None


# ---------------------------------------------------------------------------- whole search, small scene
@pytest.fixture(scope="module")
def strip_scene():
    """A strip cut from a rendered terrain under the same Sun, 15 degrees and 180 px from its prior."""
    terrain = crater_terrain((1100, 1100), 8)
    image = (render(terrain, 40.0, 25.0) * 254 + 1).astype(np.uint8)
    w, h = 300, 800
    a = np.radians(12.0)
    truth = np.array([[np.cos(a), -np.sin(a), 330.0], [np.sin(a), np.cos(a), 110.0], [0.0, 0.0, 1.0]])
    import cv2

    source = cv2.warpAffine(image, truth[:2], (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    prior = truth.copy()
    prior[:2, 2] += np.array([150.0, -100.0])
    overlap = OverlapEstimate(
        mode="metadata", reason="test", prior_affine=prior[:2].tolist(), prior_mirrored=False, search_radius_px=260.0, reference_gsd_m=5.0
    )
    guided = GuidedSearchConfig(anchor_tile_px=256, anchor_scale=1.0, anchor_step_fraction=0.5, fine_tile_px=256, fine_margin_px=48, fine_jitter_px=8, fine_passes=2, decoy_controls=2, use_strip_model=False)
    return source, image, truth, overlap, guided


@needs_loftr
def test_guided_search_recovers_a_known_placement_and_decoys_stay_low(strip_scene) -> None:
    source, image, truth, overlap, guided = strip_scene
    matcher = create_pair_matcher(LOFTR, PipelineConfig())
    src_view = ImageView(source, np.ones(source.shape, np.uint8))
    ref_view = ImageView(image, np.ones(image.shape, np.uint8))
    outcome = guided_search(matcher, src_view, ref_view, overlap, guided, GeometryConfig(), ValidationConfig(), seed=1)
    assert outcome.geometry is not None, outcome.trace.get("anchor_log")
    assert outcome.geometry.valid and outcome.hypothesis.name == "metadata"
    grid = np.array([[0.0, 0.0], [299.0, 0.0], [299.0, 799.0], [0.0, 799.0], [150.0, 400.0]])
    errors = np.linalg.norm(outcome.geometry.apply(grid) - apply_transform(truth, grid), axis=1)
    assert errors.max() < 3.0, errors
    assert outcome.trace["anchor"]["offset_from_prior_px"] == pytest.approx([-150.0, 100.0], abs=20.0)
    # All coordinates came back in original pixels.
    r = outcome.result
    assert np.linalg.norm(apply_transform(truth, r.source_points[outcome.geometry.inlier_idx]) - r.reference_points[outcome.geometry.inlier_idx], axis=1).max() < 5.0

    decoys = run_decoy_controls(matcher, src_view, ref_view, outcome, guided, GeometryConfig(), ValidationConfig(), seed=1)
    assert len(decoys) == 2 and all(d.tiles_used > 0 for d in decoys)
    assert len(outcome.geometry.inlier_idx) > 10 * max(max(d.inliers for d in decoys), 1)


def test_sift_can_be_limited_to_the_search_region(image: np.ndarray) -> None:
    from app.ingestion.raster import RasterData, RasterInfo
    from app.registration.engine import _sift_branch

    info = RasterInfo(
        path="x", width=640, height=640, band_count=1, dtype="uint8", nodata=None, crs=None, crs_is_geographic=None,
        transform=[1, 0, 0, 0, 1, 0], has_geotransform=False, pixel_size=None, bounds=None, gcps=[], gcp_crs=None,
        crs_label_consistent=None, tags={},
    )  # fmt: skip
    raster = RasterData(info, image, np.ones(image.shape, bool))
    overlap = OverlapEstimate(mode="metadata", reason="t", search_bbox_px=[100, 100, 400, 400])
    cfg = PipelineConfig(preprocessing_modes=["clahe"], pool_modes=False)
    assert _sift_branch(raster, raster, cfg, lambda m, f: None, overlap).keypoints_reference[:, 0].max() > 400
    cfg.sift.use_search_region = True
    limited = _sift_branch(raster, raster, cfg, lambda m, f: None, overlap).keypoints_reference
    assert limited.min() >= 100 and limited.max() < 400


def test_hypothesis_shift_changes_only_the_translation() -> None:
    h = Hypothesis("x", np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 6.0], [0.0, 0.0, 1.0]]), False, True)
    moved = h.shifted(np.array([10.0, 20.0]))
    np.testing.assert_allclose(moved[:2, 2], [15.0, 26.0])
    np.testing.assert_allclose(h.matrix[:2, 2], [5.0, 6.0])


def test_tile_plan_handles_a_rotated_mapping() -> None:
    a = np.radians(118.0)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    centres = plan_tile_centres((600, 5000), rotation, 640.0, 480.0)
    assert len(centres) >= 10 and np.all(centres[:, 1] >= 0) and np.all(centres[:, 1] <= 4999)


def test_lunar_like_helper_is_available_for_fixtures() -> None:
    assert lunar_like((32, 32), 0).shape == (32, 32)
