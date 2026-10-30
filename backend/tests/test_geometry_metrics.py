"""Robust estimation, transform validation and metrics."""
from __future__ import annotations

import numpy as np
import pytest

from app.config import CoverageConfig, GeometryConfig, ValidationConfig
from app.evaluation.metrics import (
    cross_validated_rmse,
    expected_chance_inliers,
    reprojection_stats,
    spatial_coverage,
)
from app.geometry.consensus import find_consensus
from app.geometry.estimation import apply_transform, distinct_points, estimate_transform, reprojection_errors
from app.geometry.validation import check_transform

TRUE = np.array([[0.98, -0.17, 850.0], [0.17, 0.98, 420.0], [0.0, 0.0, 1.0]])


def _points(n_in: int, n_out: int, seed: int = 0, noise: float = 0.5):
    rng = np.random.default_rng(seed)
    src_in = rng.uniform([0, 0], [600, 5000], (n_in, 2))
    dst_in = apply_transform(TRUE, src_in) + rng.normal(0, noise, (n_in, 2))
    src_out = rng.uniform([0, 0], [600, 5000], (n_out, 2))
    dst_out = rng.uniform([0, 0], [3000, 6000], (n_out, 2))
    return np.vstack([src_in, src_out]), np.vstack([dst_in, dst_out])


@pytest.mark.parametrize("method", ["ransac", "magsac"])
@pytest.mark.parametrize("model", ["affine", "homography"])
def test_robust_estimation_recovers_transform(model: str, method: str) -> None:
    src, dst = _points(150, 150)
    est = estimate_transform(src, dst, model, GeometryConfig(method=method))
    assert est is not None and est.model == model
    assert est.inlier_mask[:150].mean() > 0.95 and est.inlier_mask[150:].mean() < 0.05
    corners = np.array([[0, 0], [600, 5000]], float)
    np.testing.assert_allclose(apply_transform(est.matrix, corners), apply_transform(TRUE, corners), atol=3.0)


def test_consensus_at_three_percent_inliers() -> None:
    src, dst = _points(30, 970, seed=4)
    ratios = np.r_[np.random.default_rng(1).uniform(0.5, 0.9, 30), np.random.default_rng(2).uniform(0.6, 0.9, 970)]
    consensus, log = find_consensus(src, dst, ratios, GeometryConfig(), ValidationConfig(), (600, 5000), (3000, 6000))
    assert consensus is not None, log
    assert set(consensus.inlier_idx.tolist()) <= set(range(30))
    assert len(consensus.inlier_idx) >= 25


def test_consensus_rejects_pure_noise() -> None:
    src, dst = _points(0, 800, seed=9)
    ratios = np.random.default_rng(3).uniform(0.5, 0.9, 800)
    consensus, _ = find_consensus(src, dst, ratios, GeometryConfig(), ValidationConfig(), (600, 5000), (3000, 6000))
    assert consensus is None


def test_validation_accepts_plausible_and_rejects_unstable() -> None:
    cfg = ValidationConfig()
    assert check_transform(TRUE, (600, 5000), (3000, 6000), cfg).valid
    mirrored = TRUE @ np.diag([-1.0, 1.0, 1.0])
    assert any("negative determinant" in r for r in check_transform(mirrored, (600, 5000), (3000, 6000), cfg).reasons)
    tiny = np.diag([0.01, 0.01, 1.0])
    assert any("scale" in r for r in check_transform(tiny, (600, 5000), (3000, 6000), cfg).reasons)
    far = TRUE.copy()
    far[:2, 2] = [1e6, 1e6]
    assert any("absurd translation" in r for r in check_transform(far, (600, 5000), (3000, 6000), cfg).reasons)
    persp = TRUE.copy()
    persp[2, 1] = 5e-4
    assert any("perspective" in r for r in check_transform(persp, (600, 5000), (3000, 6000), cfg).reasons)
    singular = np.array([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 1.0]])
    assert not check_transform(singular, (600, 5000), (3000, 6000), cfg).valid
    nan = TRUE.copy()
    nan[0, 0] = np.nan
    assert check_transform(nan, (600, 5000), (3000, 6000), cfg).reasons == ["transformation has non-finite values"]


def test_reprojection_error_and_stats() -> None:
    src = np.array([[0, 0], [10, 0], [0, 10]], float)
    dst = apply_transform(TRUE, src) + np.array([[3, 4], [0, 0], [0, 0]])
    np.testing.assert_allclose(reprojection_errors(TRUE, src, dst), [5, 0, 0], atol=1e-9)
    stats = reprojection_stats(TRUE, src, dst)
    assert stats.rmse_px == pytest.approx(np.sqrt(25 / 3)) and stats.median_px == pytest.approx(0) and stats.max_px == pytest.approx(5)


def test_cross_validated_rmse_is_not_optimistic() -> None:
    src, dst = _points(60, 0, noise=1.0)
    in_sample = reprojection_stats(estimate_transform(src, dst, "affine", GeometryConfig()).matrix, src, dst).rmse_px
    cv = cross_validated_rmse(src, dst, "affine")
    assert cv is not None and cv >= in_sample * 0.95


def test_spatial_coverage_counts_cells() -> None:
    valid = np.ones((800, 800), bool)
    valid[:, :100] = False  # first grid column is invalid
    pts = np.array([[150, 50], [160, 60], [750, 750], [50, 50]], float)  # last one sits in the invalid column
    cov = spatial_coverage(pts, valid, CoverageConfig(grid_rows=8, grid_cols=8))
    assert cov.valid_cells == 56
    assert cov.occupied_cells == 2
    assert cov.coverage == pytest.approx(2 / 56)
    overlap = valid.copy()
    overlap[400:] = False
    cov2 = spatial_coverage(pts, valid, CoverageConfig(), overlap)
    assert cov2.overlap_cells == 28 and cov2.coverage_within_overlap == pytest.approx(1 / 28)


def test_distinct_points_and_chance_expectation() -> None:
    pts = np.array([[10.0, 10.0], [10.5, 10.4], [30.0, 30.0]])
    keep = distinct_points(pts, np.array([0.8, 0.5, 0.7]), 2.0)
    assert keep.tolist() == [1, 2]  # the better of the two co-located points survives
    assert expected_chance_inliers(1000, 3.0, 10_000_000) == pytest.approx(1000 * np.pi * 9 / 1e7)
