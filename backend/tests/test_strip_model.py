"""Along-track strip model, including the defects found on official data."""
from __future__ import annotations

import numpy as np
import pytest

from app.config import GeometryConfig, GuidedSearchConfig, ValidationConfig
from app.geometry.estimation import apply_transform
from app.geometry.strip_model import (
    StripModel,
    balancing_weights,
    check_strip,
    choose_smoothness,
    fit_strip,
    leave_block_out_errors,
    make_knots,
    track_residuals,
)
from app.geometry.verification import verify_candidates

SIZE = (600, 5000)
REF_SIZE = (3000, 7000)
AFFINE = np.array([[0.97, -0.12, 1100.0], [0.12, 1.02, 1500.0], [0.0, 0.0, 1.0]])


def _drift(y: np.ndarray) -> np.ndarray:
    """A smooth along-track deviation of up to about 80 px, as measured on a real OHRC strip."""
    return 80.0 * np.sin(y / 5000.0 * 2.2) - 0.0


def _strip_points(n: int, seed: int = 0, noise: float = 0.3, y_range=(0, 5000)):
    rng = np.random.default_rng(seed)
    src = np.c_[rng.uniform(0, 600, n), rng.uniform(*y_range, n)]
    dst = apply_transform(AFFINE, src)
    dst[:, 1] += _drift(src[:, 1])
    return src, dst + rng.normal(0, noise, dst.shape)


def test_knots_are_uniform_so_the_penalty_reduces_to_a_straight_line() -> None:
    knots = make_knots(5000, 500)
    assert np.allclose(np.diff(knots), np.diff(knots)[0])
    assert knots[0] < 0 and knots[-1] > 1


def test_large_smoothness_recovers_an_affine_map() -> None:
    rng = np.random.default_rng(1)
    src = np.c_[rng.uniform(0, 600, 400), rng.uniform(0, 5000, 400)]
    dst = apply_transform(AFFINE, src)
    model = fit_strip(src, dst, SIZE, 450.0, 1e8)
    grid = np.c_[np.linspace(0, 600, 30), np.linspace(0, 5000, 30)]
    np.testing.assert_allclose(model.apply(grid), apply_transform(AFFINE, grid), atol=0.2)


def test_unsupported_ends_stay_plausible() -> None:
    """Regression: clamped knots gave a local scale of 3 at both ends, whatever the smoothness."""
    src, dst = _strip_points(500, y_range=(1200, 3800))
    for smoothness in (1.0, 100.0, 1e4, 1e6):
        model = fit_strip(src, dst, SIZE, 450.0, smoothness)
        ends = np.array([[300.0, 0.0], [300.0, 4999.0], [300.0, 400.0], [300.0, 4600.0]])
        singular = np.linalg.svd(model.jacobian(ends), compute_uv=False)
        assert singular.max() < 1.3, f"smoothness {smoothness}: {singular.max():.2f}"
        assert check_strip(model, REF_SIZE, ValidationConfig(), 1).valid


def test_fit_follows_a_curved_drift() -> None:
    src, dst = _strip_points(800)
    model = fit_strip(src, dst, SIZE, 450.0, 1.0)
    assert np.sqrt(np.mean(model.errors(src, dst) ** 2)) < 1.0


def test_inverse_round_trip() -> None:
    src, dst = _strip_points(800)
    model = fit_strip(src, dst, SIZE, 450.0, 1.0)
    points = np.c_[np.linspace(10, 590, 40), np.linspace(10, 4990, 40)]
    np.testing.assert_allclose(model.inverse(model.apply(points)), points, atol=0.05)


def test_serialisation_round_trip() -> None:
    src, dst = _strip_points(300)
    model = fit_strip(src, dst, SIZE, 450.0, 10.0)
    again = StripModel.from_dict(model.to_dict())
    points = np.c_[np.linspace(0, 600, 10), np.linspace(0, 5000, 10)]
    np.testing.assert_allclose(again.apply(points), model.apply(points))
    assert again.spacing_px == pytest.approx(model.spacing_px)


def test_dense_group_does_not_decide_the_fit() -> None:
    """Regression: one tile with hundreds of matches pulled the curve away from its neighbours."""
    weights = balancing_weights(np.r_[np.full(500, 100.0), np.full(5, 2000.0)], 5000.0, 450.0)
    assert weights[-1] / weights[0] > 3.0
    # Total group weight (squared) grows with the square root of the count, not with the count.
    assert (500 * weights[0] ** 2) / (5 * weights[-1] ** 2) == pytest.approx(10.0, rel=0.01)


def test_smoothness_choice_prefers_the_smoother_of_equals() -> None:
    rng = np.random.default_rng(3)
    src = np.c_[rng.uniform(0, 600, 300), rng.uniform(0, 5000, 300)]
    dst = apply_transform(AFFINE, src) + rng.normal(0, 0.5, (300, 2))
    best, scores = choose_smoothness(src, dst, SIZE, 450.0, [1.0, 100.0, 1e4, 1e6])
    assert best == 1e6 and set(scores) == {1.0, 100.0, 1e4, 1e6}


def test_track_ignores_false_peaks() -> None:
    src, dst = _strip_points(400, noise=0.5)
    rng = np.random.default_rng(5)
    false_src = np.c_[rng.uniform(0, 600, 400), rng.uniform(0, 5000, 400)]
    false_dst = apply_transform(AFFINE, false_src) + rng.uniform(-400, 400, (400, 2))
    all_src, all_dst = np.vstack([src, false_src]), np.vstack([dst, false_dst])
    track = track_residuals(all_src, all_dst - apply_transform(AFFINE, all_src), SIZE, 8.0, 6, 450.0)
    assert len(track) >= 8
    for along, value, _ in track:
        assert abs(value[1] - _drift(np.array([along]))[0]) < 25.0


def test_leave_block_out_reports_each_tested_block() -> None:
    src, dst = _strip_points(300)
    blocks = leave_block_out_errors(src, dst, SIZE, lambda a, b: fit_strip(a, b, SIZE, 450.0, 100.0).apply, blocks=6)
    assert len(blocks) == 6 and all(len(b) > 0 for b in blocks)


def test_verification_prefers_the_model_that_predicts_unseen_ground() -> None:
    src, dst = _strip_points(1200, noise=0.4)
    rng = np.random.default_rng(9)
    wrong_src = np.c_[rng.uniform(0, 600, 300), rng.uniform(0, 5000, 300)]
    wrong_dst = rng.uniform([0, 0], REF_SIZE, (300, 2))
    all_src, all_dst = np.vstack([src, wrong_src]), np.vstack([dst, wrong_dst])
    rank = rng.uniform(0.1, 0.9, len(all_src))
    geometry, log = verify_candidates(all_src, all_dst, rank, SIZE, REF_SIZE, GeometryConfig(), ValidationConfig(), GuidedSearchConfig())
    assert geometry is not None, log
    assert geometry.kind == "strip_spline", log
    assert geometry.valid and len(geometry.inlier_idx) > 800
    assert set(geometry.inlier_idx.tolist()) <= set(range(1200))


def test_verification_keeps_the_simple_model_when_it_is_enough() -> None:
    rng = np.random.default_rng(11)
    src = np.c_[rng.uniform(0, 600, 800), rng.uniform(0, 5000, 800)]
    dst = apply_transform(AFFINE, src) + rng.normal(0, 0.3, (800, 2))
    rank = rng.uniform(0.1, 0.9, 800)
    geometry, log = verify_candidates(src, dst, rank, SIZE, REF_SIZE, GeometryConfig(), ValidationConfig(), GuidedSearchConfig())
    assert geometry is not None, log
    np.testing.assert_allclose(geometry.apply(src[:50]), apply_transform(AFFINE, src[:50]), atol=1.0)


def test_mirrored_candidates_are_checked_in_the_mirrored_frame() -> None:
    flip = np.array([[-1.0, 0.0, 599.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    matrix = AFFINE @ flip
    rng = np.random.default_rng(13)
    src = np.c_[rng.uniform(0, 600, 500), rng.uniform(0, 5000, 500)]
    dst = apply_transform(matrix, src) + rng.normal(0, 0.3, (500, 2))
    geometry, log = verify_candidates(src, dst, rng.uniform(0.1, 0.9, 500), SIZE, REF_SIZE, GeometryConfig(), ValidationConfig(), None, mirrored=True)
    assert geometry is not None and geometry.valid, log
    assert np.linalg.det(geometry.matrix[:2, :2]) < 0
