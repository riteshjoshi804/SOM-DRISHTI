"""Normalisation, SIFT extraction and descriptor matching."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from conftest import lunar_like

from app.config import MatchConfig, SiftConfig
from app.matching.matcher import match_features
from app.matching.sift import extract_sift, to_root_sift
from app.preprocessing.pipeline import percentile_normalize, preprocess


def test_float32_percentile_normalisation_is_not_a_cast() -> None:
    rng = np.random.default_rng(3)
    data = (rng.normal(600, 80, (200, 200))).astype(np.float32)  # IIRS-like radiance
    data[0, 0] = np.nan
    data[0, 1] = np.inf
    valid = np.isfinite(data)
    out, lo, hi = percentile_normalize(data, valid, 1, 99)
    assert out.dtype == np.uint8
    finite = data[valid]
    assert lo == pytest.approx(np.percentile(finite, 1)) and hi == pytest.approx(np.percentile(finite, 99))
    # ~1% of pixels at or below the low percentile map to 0, ~1% saturate at 255.
    assert 0.005 < (out[valid] == 0).mean() < 0.02
    assert 0.005 < (out[valid] == 255).mean() < 0.02
    assert out[0, 0] == 0 and out[0, 1] == 0  # invalid pixels
    # A naive cast would saturate almost everything.
    assert (np.clip(finite, 0, 255).astype(np.uint8) == 255).mean() > 0.9


def test_normalisation_does_not_modify_input() -> None:
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    before = data.copy()
    percentile_normalize(data, np.ones_like(data, bool))
    np.testing.assert_array_equal(data, before)


@pytest.mark.parametrize("mode", ["raw_normalized", "clahe", "gradient"])
def test_preprocessing_modes(mode: str) -> None:
    data = (lunar_like((300, 200), 5) * 900).astype(np.float32)
    valid = np.ones(data.shape, bool)
    valid[:, :10] = False
    pre = preprocess(data, valid, mode)
    assert pre.image.dtype == np.uint8 and pre.image.shape == data.shape
    assert (pre.image[~valid] == 0).all()
    # The detection mask is eroded away from invalid pixels and the border.
    assert pre.detect_mask[:, :10].max() == 0 and pre.detect_mask[:, 10:15].max() == 0
    assert pre.detect_mask[150, 100] == 255


def test_sift_extraction_respects_mask_and_tiles() -> None:
    img = (lunar_like((600, 500), 7) * 255).astype(np.uint8)
    mask = np.full(img.shape, 255, np.uint8)
    mask[:, :250] = 0
    feats = extract_sift(img, mask, SiftConfig(tile_size=256, nfeatures=500))
    assert len(feats) > 100
    assert feats.descriptors.shape == (len(feats), 128)
    assert feats.points[:, 0].min() >= 250 - 1  # nothing detected in the masked half
    # Tiled extraction must not duplicate keypoints along tile seams. (SIFT itself
    # may emit one location with several orientations, so compare with the angle.)
    keys = np.c_[np.round(feats.points, 2), np.round(feats.angles, 1)]
    assert len(np.unique(keys, axis=0)) == len(feats)


def test_root_sift_is_unit_norm() -> None:
    d = np.abs(np.random.default_rng(0).normal(size=(10, 128))).astype(np.float32)
    np.testing.assert_allclose(np.linalg.norm(to_root_sift(d), axis=1), 1.0, rtol=1e-5)


def test_matching_ratio_test_and_one_to_one() -> None:
    ref = (lunar_like((500, 500), 11) * 255).astype(np.uint8)
    rot = cv2.getRotationMatrix2D((250, 250), 25, 1.0)
    src = cv2.warpAffine(ref, rot, (500, 500))
    full = np.full(ref.shape, 255, np.uint8)
    fs = extract_sift(src, cv2.warpAffine(full, rot, (500, 500)), SiftConfig(nfeatures=800))
    fr = extract_sift(ref, full, SiftConfig(nfeatures=800))
    loose = match_features(fs, fr, MatchConfig(ratio_threshold=0.95))
    strict = match_features(fs, fr, MatchConfig(ratio_threshold=0.6))
    assert loose.raw_candidates == len(fs)
    assert len(strict) < len(loose)
    assert len(np.unique(loose.reference_idx)) == len(loose)  # one-to-one
    assert (strict.ratios < 0.6).all()
    # Most strict matches must be geometrically right (reference = inverse rotation).
    inv = cv2.invertAffineTransform(rot)
    pred = fs.points[strict.source_idx] @ inv[:, :2].T + inv[:, 2]
    err = np.linalg.norm(pred - fr.points[strict.reference_idx], axis=1)
    assert (err < 3).mean() > 0.8
