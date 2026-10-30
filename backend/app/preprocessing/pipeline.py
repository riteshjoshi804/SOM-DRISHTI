"""Turn a scientific raster into an 8-bit image suitable for SIFT.

Three representations are available. None of them *solves* Sun-angle
differences - on the Moon a different Sun geometry moves real shadows, which
no contrast operation can undo. They are alternatives the registration engine
can test, nothing more:

* ``raw_normalized`` - percentile stretch only.
* ``clahe``          - percentile stretch + local contrast equalisation.
* ``gradient``       - gradient magnitude of the stretched image (insensitive
                       to the sign of the local contrast).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.config import PreprocessConfig, PreprocessMode


@dataclass
class Preprocessed:
    image: np.ndarray  # uint8, the representation SIFT runs on
    display: np.ndarray  # uint8, plain percentile stretch for visualisation
    valid_mask: np.ndarray  # bool, original valid pixels
    detect_mask: np.ndarray  # uint8 0/255, eroded - where keypoints may be detected
    mode: str
    stretch_low: float
    stretch_high: float


def percentile_normalize(
    data: np.ndarray, valid_mask: np.ndarray, p_low: float = 1.0, p_high: float = 99.0
) -> tuple[np.ndarray, float, float]:
    """Linear stretch of ``[p_low, p_high]`` percentiles to 0..255.

    Percentiles are taken over valid pixels only. Values outside the range are
    clipped; invalid pixels (NoData / NaN / Inf) become 0. The input array is
    never modified, and float data is never cast straight to uint8.
    """
    out = np.zeros(data.shape, dtype=np.uint8)
    values = data[valid_mask]
    if values.size == 0:
        return out, float("nan"), float("nan")
    lo, hi = (float(v) for v in np.percentile(values.astype(np.float64), (p_low, p_high)))
    if not hi > lo:  # constant image
        hi = lo + 1.0
    scaled = (np.where(valid_mask, data, lo).astype(np.float32) - lo) * (255.0 / (hi - lo))
    np.clip(scaled, 0, 255, out=scaled)
    out[valid_mask] = np.rint(scaled[valid_mask]).astype(np.uint8)
    return out, lo, hi


def apply_clahe(image: np.ndarray, clip_limit: float, tile_px: int) -> np.ndarray:
    grid = (max(1, round(image.shape[1] / tile_px)), max(1, round(image.shape[0] / tile_px)))
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid).apply(image)


def gradient_magnitude(image: np.ndarray, valid_mask: np.ndarray, sigma: float) -> np.ndarray:
    img = image.astype(np.float32)
    if sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    gx = cv2.Scharr(img, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(img, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    # The data/no-data boundary is not terrain; drop its gradient response.
    inner = cv2.erode(valid_mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    out, _, _ = percentile_normalize(mag, inner, 1.0, 99.5)
    return out


def erode_mask(valid_mask: np.ndarray, radius_px: int) -> np.ndarray:
    """0/255 detection mask, shrunk so descriptors do not straddle invalid pixels."""
    mask = valid_mask.astype(np.uint8) * 255
    if radius_px <= 0:
        return mask
    # Pad with invalid pixels so the image border is eroded as well.
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1))
    return cv2.erode(padded, kernel)[1:-1, 1:-1]


def preprocess(
    data: np.ndarray, valid_mask: np.ndarray, mode: PreprocessMode, cfg: PreprocessConfig | None = None
) -> Preprocessed:
    cfg = cfg or PreprocessConfig()
    if data.ndim != 2:
        raise ValueError(f"Expected a single-band 2-D array, got shape {data.shape}")

    display, lo, hi = percentile_normalize(data, valid_mask, cfg.percentile_low, cfg.percentile_high)
    base = cv2.GaussianBlur(display, (0, 0), cfg.gaussian_sigma) if cfg.gaussian_sigma > 0 else display

    if mode == "raw_normalized":
        image = base
    elif mode == "clahe":
        image = apply_clahe(base, cfg.clahe_clip_limit, cfg.clahe_tile_px)
    elif mode == "gradient":
        image = gradient_magnitude(apply_clahe(base, cfg.clahe_clip_limit, cfg.clahe_tile_px), valid_mask, cfg.gradient_sigma)
    else:
        raise ValueError(f"Unknown preprocessing mode: {mode}")

    image = np.where(valid_mask, image, 0).astype(np.uint8)
    return Preprocessed(
        image=image,
        display=display,
        valid_mask=valid_mask,
        detect_mask=erode_mask(valid_mask, cfg.mask_erosion_px),
        mode=mode,
        stretch_low=lo,
        stretch_high=hi,
    )
