"""Turn a scientific raster into an 8-bit image suitable for matching.

Several representations are available. None of them *solves* Sun-angle
differences - on the Moon a different Sun geometry moves real shadows, which
no contrast operation can undo. They are alternatives the registration engine
can test, nothing more:

* ``raw_normalized``   - percentile stretch only.
* ``clahe``            - percentile stretch + local contrast equalisation.
* ``gradient``         - gradient magnitude of the stretched image (insensitive
                         to the sign of the local contrast).
* ``structural``       - gradient magnitude divided by its local mean, so that weak
                         texture in dim terrain counts as much as strong texture
                         in bright terrain.
* ``phase_congruency`` - local phase agreement across scales (see phase_congruency.py),
                         which depends on feature shape rather than contrast.
* ``shadow_aware``     - CLAHE with deep shadows neutralised and kept out of the
                         detection mask, because shadow edges move with the Sun.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.config import PreprocessConfig, PreprocessMode
from app.preprocessing.phase_congruency import phase_congruency


@dataclass
class Preprocessed:
    image: np.ndarray  # uint8, the representation the matcher runs on
    display: np.ndarray  # uint8, plain percentile stretch for visualisation
    valid_mask: np.ndarray  # bool, original valid pixels
    detect_mask: np.ndarray  # uint8 0/255, eroded - where keypoints may be detected
    mode: str
    stretch_low: float
    stretch_high: float
    shadow_mask: np.ndarray | None = None  # bool, deep shadow; only computed by ``shadow_aware``


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


def structural_map(image: np.ndarray, valid_mask: np.ndarray, sigma: float, window_px: int) -> np.ndarray:
    """Gradient magnitude divided by its local mean (masked, so the swath edge does not leak in)."""
    img = image.astype(np.float32)
    if sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    magnitude = cv2.magnitude(cv2.Scharr(img, cv2.CV_32F, 1, 0), cv2.Scharr(img, cv2.CV_32F, 0, 1))
    inner = cv2.erode(valid_mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
    magnitude *= inner
    spread = window_px / 3.0
    local = cv2.GaussianBlur(magnitude, (0, 0), spread)
    weight = cv2.GaussianBlur(inner, (0, 0), spread)
    mean = local / np.maximum(weight, 1e-3)
    floor = float(np.percentile(magnitude[inner > 0], 50)) * 0.1 if inner.any() else 1.0
    normalised = magnitude / (mean + max(floor, 1e-3))
    out, _, _ = percentile_normalize(normalised, inner > 0, 1.0, 99.5)
    return out


def shadow_mask(display: np.ndarray, valid_mask: np.ndarray, threshold: int, min_area_px: int) -> np.ndarray:
    """Deep shadow: connected dark regions of real size. Dark specks are left alone."""
    smooth = cv2.GaussianBlur(display, (0, 0), 1.5)
    dark = ((smooth <= threshold) & valid_mask).astype(np.uint8)
    if not dark.any():
        return np.zeros(display.shape, dtype=bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area_px
    keep[0] = False
    return keep[labels]


def shadow_aware(
    base: np.ndarray, valid_mask: np.ndarray, cfg: PreprocessConfig
) -> tuple[np.ndarray, np.ndarray]:
    """CLAHE image with deep shadows replaced by a neutral level, and the shadow mask.

    This hides shadows from the matcher; it is not a physical correction and recovers
    nothing inside them. The transition is feathered so the shadow edge itself does
    not become the strongest feature in the image.
    """
    shadow = shadow_mask(base, valid_mask, cfg.shadow_threshold, cfg.shadow_min_area_px)
    equalised = apply_clahe(base, cfg.clahe_clip_limit, cfg.clahe_tile_px).astype(np.float32)
    lit = valid_mask & ~shadow
    level = float(np.median(equalised[lit])) if lit.any() else 128.0
    blend = cv2.GaussianBlur(shadow.astype(np.float32), (0, 0), cfg.shadow_feather_px)
    # Inside a shadow the weight stays 1; the feather only reaches outward.
    blend = np.maximum(blend * 2.0, shadow.astype(np.float32)).clip(0.0, 1.0)
    out = equalised * (1.0 - blend) + level * blend
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), shadow


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
    elif mode == "structural":
        image = structural_map(apply_clahe(base, cfg.clahe_clip_limit, cfg.clahe_tile_px), valid_mask, cfg.gradient_sigma, cfg.structural_window_px)
    elif mode == "phase_congruency":
        response = phase_congruency(base, cfg.phase_scales, cfg.phase_orientations)
        inner = cv2.erode(valid_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        image, _, _ = percentile_normalize(response, inner, 1.0, 99.5)
    elif mode == "shadow_aware":
        image, shadows = shadow_aware(base, valid_mask, cfg)
    else:
        raise ValueError(f"Unknown preprocessing mode: {mode}")

    image = np.where(valid_mask, image, 0).astype(np.uint8)
    detect_mask = erode_mask(valid_mask, cfg.mask_erosion_px)
    if mode == "shadow_aware":
        grown = cv2.dilate(shadows.astype(np.uint8), np.ones((2 * cfg.shadow_margin_px + 1,) * 2, np.uint8)) > 0
        detect_mask = np.where(grown, 0, detect_mask).astype(np.uint8)
    return Preprocessed(
        image=image,
        display=display,
        valid_mask=valid_mask,
        detect_mask=detect_mask,
        mode=mode,
        stretch_low=lo,
        stretch_high=hi,
        shadow_mask=shadows if mode == "shadow_aware" else None,
    )
