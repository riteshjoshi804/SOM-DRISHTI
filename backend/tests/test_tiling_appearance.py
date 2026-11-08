"""Coordinate mapping through crops, resizes and rotated tiles; the matcher-independent appearance check."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from conftest import lunar_like

from app.config import GuidedSearchConfig
from app.matching.appearance import gradient_map, patch_zncc, summarise
from app.matching.base import MatcherResult, PairMatches, failed_result
from app.matching.guided import appearance_filter, plan_tile_centres, vote_translation
from app.matching.tiling import (
    PatchMap,
    compose,
    crop_map,
    crop_scaled,
    keep_inside_mask,
    rectified_tile,
    resize_map,
    spans,
)


def _gradient_image(h: int = 400, w: int = 500) -> np.ndarray:
    """Pixel value encodes position, so sampling errors show up as value errors."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return (xx * 0.3 + yy * 0.2).astype(np.float32)


def test_crop_map_round_trip() -> None:
    mapping = crop_map(100, 40)
    np.testing.assert_allclose(mapping.patch_to_original(np.array([[0, 0], [5, 7]])), [[100, 40], [105, 47]])
    np.testing.assert_allclose(mapping.original_to_patch(np.array([[105, 47]])), [[5, 7]])


@pytest.mark.parametrize("scale", [0.5, 0.25, 2.0, 0.37])
def test_resize_map_matches_how_cv2_samples(scale: float) -> None:
    image = _gradient_image()
    size = (round(image.shape[1] * scale), round(image.shape[0] * scale))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    mapping = resize_map(size[0] / image.shape[1], size[1] / image.shape[0])
    points = np.array([[10.0, 12.0], [40.0, 30.0], [size[0] // 2, size[1] // 2]])
    original = mapping.patch_to_original(points)
    expected = original[:, 0] * 0.3 + original[:, 1] * 0.2
    np.testing.assert_allclose(resized[points[:, 1].astype(int), points[:, 0].astype(int)], expected, atol=0.05)


def test_crop_scaled_maps_back_to_original_pixels() -> None:
    image = _gradient_image()
    patch = crop_scaled(image, np.ones(image.shape, np.uint8), 50, 60, 450, 360, scale=0.5)
    points = np.array([[10.0, 10.0], [100.0, 80.0]])
    original = patch.map.patch_to_original(points)
    expected = original[:, 0] * 0.3 + original[:, 1] * 0.2
    np.testing.assert_allclose(patch.image[points[:, 1].astype(int), points[:, 0].astype(int)], expected, atol=0.05)


@pytest.mark.parametrize("angle", [0.0, 17.0, 118.0, 180.0])
@pytest.mark.parametrize("mirror", [False, True])
def test_rectified_tile_maps_back_exactly(angle: float, mirror: bool) -> None:
    image = _gradient_image()
    a = np.radians(angle)
    linear = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    if mirror:
        linear = linear @ np.diag([-1.0, 1.0])
    patch = rectified_tile(image, np.ones(image.shape, np.uint8), linear, np.array([250.0, 200.0]), (101, 101))
    points = np.array([[50.0, 50.0], [20.0, 70.0], [80.0, 30.0]])
    original = patch.map.patch_to_original(points)
    expected = original[:, 0] * 0.3 + original[:, 1] * 0.2
    np.testing.assert_allclose(patch.image[points[:, 1].astype(int), points[:, 0].astype(int)], expected, atol=0.05)
    # The tile centre looks at the requested source point.
    np.testing.assert_allclose(patch.map.patch_to_original(np.array([[50.0, 50.0]])), [[250.0, 200.0]], atol=1e-9)


def test_rectified_tile_scale_is_undone_by_its_map() -> None:
    image = _gradient_image()
    linear = np.eye(2) * 0.5
    patch = rectified_tile(image, np.ones(image.shape, np.uint8), linear, np.array([250.0, 200.0]), (81, 81))
    original = patch.map.patch_to_original(np.array([[40.0, 40.0], [60.0, 40.0]]))
    np.testing.assert_allclose(original[1] - original[0], [40.0, 0.0])


def test_rectified_tile_outside_image_is_empty_not_a_crash() -> None:
    image = _gradient_image()
    patch = rectified_tile(image, np.ones(image.shape, np.uint8), np.eye(2), np.array([5000.0, 5000.0]), (64, 64))
    assert patch.mask.sum() == 0


def test_compose_applies_inner_first() -> None:
    outer, inner = crop_map(10, 20), resize_map(0.5, 0.5)
    composed = compose(outer, inner)
    p = np.array([[3.0, 4.0]])
    np.testing.assert_allclose(composed.patch_to_original(p), outer.patch_to_original(inner.patch_to_original(p)))
    assert isinstance(composed, PatchMap)


def test_keep_inside_mask() -> None:
    mask = np.zeros((10, 10), np.uint8)
    mask[5, 5] = 1
    kept = keep_inside_mask(np.array([[5, 5], [0, 0], [100, 100], [-1, 3]]), mask)
    assert kept.tolist() == [True, False, False, False]


def test_spans_cover_the_length_and_respect_the_tile_size() -> None:
    result = spans(1000, 256, 200)
    assert result[0][0] == 0 and result[-1][1] == 1000
    assert all(b - a == 256 for a, b in result)
    assert all(result[i + 1][0] <= result[i][1] for i in range(len(result) - 1))
    assert spans(100, 256, 200) == [(0, 100)]


def test_tile_plan_covers_a_strip_along_its_length() -> None:
    centres = plan_tile_centres((600, 5000), np.eye(2), 640.0, 480.0)
    assert np.all(centres[:, 0] == pytest.approx(299.5))
    assert centres[:, 1].min() < 400 and centres[:, 1].max() > 4600
    assert np.all(np.diff(centres[:, 1]) <= 480.0 + 1e-6)


def test_vote_finds_common_translation_and_runner_up() -> None:
    rng = np.random.default_rng(0)
    p0 = rng.uniform(0, 500, (60, 2))
    p1 = p0 + np.array([33.0, -12.0])
    p1[40:] = rng.uniform(0, 500, (20, 2))
    peak = vote_translation(p0, p1, 4.0)
    assert peak.votes >= 40
    np.testing.assert_allclose(peak.displacement, [33.0, -12.0], atol=0.5)
    assert peak.second <= 3


def test_vote_on_nothing() -> None:
    peak = vote_translation(np.zeros((0, 2)), np.zeros((0, 2)), 4.0)
    assert peak.votes == 0 and peak.second == 0


def test_same_structure_scores_high_and_different_structure_low() -> None:
    a = (lunar_like((300, 300), 1) * 255).astype(np.uint8)
    b = (lunar_like((300, 300), 2) * 255).astype(np.uint8)
    ga, gb = gradient_map(a), gradient_map(b)
    points = np.array([[80, 80], [150, 150], [220, 120], [100, 210]], dtype=float)
    assert np.nanmin(patch_zncc(ga, ga, points, points, 15)) > 0.99
    assert abs(np.nanmedian(patch_zncc(ga, gb, points, points, 15))) < 0.35


def test_contrast_reversal_keeps_the_gradient_similarity() -> None:
    """A changed Sun can reverse local contrast; the same structure must still be recognised."""
    a = (lunar_like((300, 300), 3) * 255).astype(np.uint8)
    ga, gb = gradient_map(a), gradient_map(255 - a)
    points = np.array([[80, 80], [150, 150], [220, 120]], dtype=float)
    assert np.nanmin(patch_zncc(ga, gb, points, points, 15)) > 0.99


def test_flat_and_out_of_bounds_patches_cannot_pass() -> None:
    """Regression: black shadow patches paired with dark terrain were among the false inliers."""
    flat = np.zeros((100, 100), np.float32)
    textured = gradient_map((lunar_like((100, 100), 4) * 255).astype(np.uint8))
    scores = patch_zncc(flat, textured, np.array([[50.0, 50.0]]), np.array([[50.0, 50.0]]), 10)
    assert np.isnan(scores[0])
    edge = patch_zncc(textured, textured, np.array([[2.0, 2.0]]), np.array([[50.0, 50.0]]), 10)
    assert np.isnan(edge[0])


def test_appearance_filter_removes_look_alikes_but_keeps_true_matches() -> None:
    image = (lunar_like((400, 400), 5) * 255).astype(np.uint8)
    other = (lunar_like((400, 400), 6) * 255).astype(np.uint8)
    true_points = np.array([[100.0, 100.0], [200.0, 150.0], [300.0, 250.0]])
    pair = PairMatches(true_points, true_points.copy(), np.full(3, 0.9), 3)
    kept, _ = appearance_filter(pair, image, image, 15, GuidedSearchConfig())
    assert len(kept) == 3
    dropped, scores = appearance_filter(pair, image, other, 15, GuidedSearchConfig())
    assert len(dropped) < 3 and len(scores) == 3


def test_filter_can_be_disabled() -> None:
    image = (lunar_like((200, 200), 7) * 255).astype(np.uint8)
    points = np.array([[100.0, 100.0]])
    pair = PairMatches(points, points.copy(), np.array([0.5]), 1)
    kept, scores = appearance_filter(pair, image, image, 10, GuidedSearchConfig(appearance_filter=False))
    assert len(kept) == 1 and np.isnan(scores[0])


def test_summary_counts_unmeasurable_points() -> None:
    summary = summarise(np.array([0.5, 0.1, np.nan, 0.9]), 0.3, 15)
    assert (summary.measured, summary.unmeasurable, summary.passed) == (3, 1, 2)
    assert summary.median_zncc == pytest.approx(0.5)


def test_matcher_result_is_standardised() -> None:
    result = MatcherResult(
        matcher_name="X",
        source_points=[[1, 2], [3, 4]],
        reference_points=[[5, 6], [7, 8]],
        confidence_scores=[0.5, 0.6],
        rank=[0.5, 0.4],
        candidate_count=9,
        accepted_count=2,
        runtime_seconds=1.0,
        device="cpu",
    )
    assert len(result) == 2 and result.source_points.dtype == np.float64 and result.status == "OK"
    with pytest.raises(ValueError):
        MatcherResult("X", [[1, 2]], [[1, 2], [3, 4]], [0.5], [0.5], 1, 1, 0.1, "cpu")


def test_failed_result_says_why() -> None:
    result = failed_result("LOFTR", "weights missing", status="UNAVAILABLE")
    assert len(result) == 0 and result.status == "UNAVAILABLE" and result.failure_reason == "weights missing"
