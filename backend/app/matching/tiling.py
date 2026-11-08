"""Pixel bookkeeping for crops, resizes and rotated tiles.

Learned matchers see tiles, not the original rasters. Each tile carries a
``PatchMap`` recording exactly how its pixels relate to the original image, so
every correspondence can be converted back. Images are never stretched to make
them agree: a rectified tile only changes the *frame* a matcher looks through, and
that frame is undone before anything is measured.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PatchMap:
    """``to_original`` maps patch pixels (x, y) to original image pixels (3x3)."""

    to_original: np.ndarray

    def patch_to_original(self, points: np.ndarray) -> np.ndarray:
        return apply_h(self.to_original, points)

    def original_to_patch(self, points: np.ndarray) -> np.ndarray:
        return apply_h(np.linalg.inv(self.to_original), points)


def apply_h(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homog = np.c_[pts, np.ones(len(pts))] @ np.asarray(matrix, dtype=np.float64).T
    return homog[:, :2] / homog[:, 2:3]


def crop_map(x0: float, y0: float) -> PatchMap:
    """Patch pixel (0, 0) is original pixel (x0, y0)."""
    return PatchMap(np.array([[1.0, 0.0, x0], [0.0, 1.0, y0], [0.0, 0.0, 1.0]]))


def resize_map(scale_x: float, scale_y: float) -> PatchMap:
    """Pixel-centre-aligned resize: patch = (original + 0.5) * scale - 0.5, as cv2.resize samples it."""
    return PatchMap(
        np.array(
            [
                [1.0 / scale_x, 0.0, 0.5 / scale_x - 0.5],
                [0.0, 1.0 / scale_y, 0.5 / scale_y - 0.5],
                [0.0, 0.0, 1.0],
            ]
        )
    )


def compose(outer: PatchMap, inner: PatchMap) -> PatchMap:
    """``outer`` maps intermediate -> original, ``inner`` maps patch -> intermediate."""
    return PatchMap(outer.to_original @ inner.to_original)


@dataclass
class Patch:
    image: np.ndarray  # uint8
    mask: np.ndarray  # uint8, non-zero = usable
    map: PatchMap


def crop_scaled(
    image: np.ndarray, mask: np.ndarray, x0: int, y0: int, x1: int, y1: int, scale: float = 1.0
) -> Patch:
    """Axis-aligned crop, optionally resized with one isotropic factor."""
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(image.shape[1], int(x1)), min(image.shape[0], int(y1))
    crop, crop_mask = image[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    mapping = crop_map(x0, y0)
    if scale != 1.0 and crop.size:
        w, h = max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        crop = cv2.resize(crop, (w, h), interpolation=interp)
        crop_mask = cv2.resize(crop_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        # The factor actually applied can differ slightly from ``scale`` after rounding.
        applied_x = w / (x1 - x0)
        applied_y = h / (y1 - y0)
        mapping = compose(mapping, resize_map(applied_x, applied_y))
    return Patch(crop, crop_mask, mapping)


def rectified_tile(
    image: np.ndarray,
    mask: np.ndarray,
    linear: np.ndarray,
    centre: np.ndarray,
    size: tuple[int, int],
) -> Patch:
    """A ``size = (w, h)`` window looking at ``image`` through the 2x2 ``linear`` map.

    ``linear`` takes source pixel offsets to tile pixel offsets, so the tile is the
    source as it would look in the target frame (typically rotated into the
    reference orientation and brought to the reference pixel size).
    """
    w, h = size
    linear = np.asarray(linear, dtype=np.float64)
    centre = np.asarray(centre, dtype=np.float64)
    t = np.array([(w - 1) / 2.0, (h - 1) / 2.0]) - linear @ centre
    forward = np.c_[linear, t]
    inverse = np.linalg.inv(np.vstack([forward, [0.0, 0.0, 1.0]]))
    mapping = PatchMap(inverse)

    # Only the part of the raster under the tile is touched, which keeps long strips cheap.
    corners = mapping.patch_to_original(np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float))
    scale = float(np.sqrt(abs(np.linalg.det(linear))))
    sigma = 0.5 / max(scale, 1e-3) if scale < 0.75 else 0.0  # anti-aliasing when downsampling
    pad = int(np.ceil(3 * sigma)) + 2
    x0 = int(np.clip(np.floor(corners[:, 0].min()) - pad, 0, image.shape[1]))
    y0 = int(np.clip(np.floor(corners[:, 1].min()) - pad, 0, image.shape[0]))
    x1 = int(np.clip(np.ceil(corners[:, 0].max()) + pad + 1, x0, image.shape[1]))
    y1 = int(np.clip(np.ceil(corners[:, 1].max()) + pad + 1, y0, image.shape[0]))
    if x1 <= x0 or y1 <= y0:
        return Patch(np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8), mapping)

    roi, roi_mask = image[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    if sigma > 0:
        roi = cv2.GaussianBlur(roi, (0, 0), sigma)
    shifted = forward.copy()
    shifted[:, 2] += linear @ np.array([x0, y0], dtype=np.float64)
    patch = cv2.warpAffine(roi, shifted, (w, h), flags=cv2.INTER_LINEAR)
    patch_mask = cv2.warpAffine(roi_mask, shifted, (w, h), flags=cv2.INTER_NEAREST)
    return Patch(patch, patch_mask, mapping)


def erode(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Shrink a mask, treating everything outside the array as unusable."""
    if radius_px <= 0:
        return mask
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1))
    return cv2.erode(padded, kernel)[1:-1, 1:-1]


def keep_inside_mask(points: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Boolean per point: inside the array and on a usable mask pixel."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if mask is None:
        return np.ones(len(pts), bool)
    h, w = mask.shape
    xi, yi = np.rint(pts[:, 0]).astype(np.int64), np.rint(pts[:, 1]).astype(np.int64)
    inside = (xi >= 0) & (yi >= 0) & (xi < w) & (yi < h)
    keep = np.zeros(len(pts), bool)
    keep[inside] = mask[yi[inside], xi[inside]] > 0
    return keep


def spans(length: int, tile: int, step: int) -> list[tuple[int, int]]:
    """Overlapping ``[start, end)`` spans of at most ``tile`` covering ``length`` exactly."""
    if length <= tile:
        return [(0, length)]
    count = int(np.ceil((length - tile) / step)) + 1
    starts = np.linspace(0, length - tile, count).round().astype(int)
    return [(int(s), int(s) + tile) for s in starts]
