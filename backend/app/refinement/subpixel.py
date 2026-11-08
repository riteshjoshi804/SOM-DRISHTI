"""Sub-pixel refinement of verified correspondences.

Fractional coordinates are not sub-pixel accuracy. Each estimator here measures a local
shift between image patches, and a point is only moved when the evidence for that shift
holds up: the patch must be fully valid and textured, the optimum must lie inside the
search range, and it must stand out from the alternatives.

The source point stays fixed and the reference point moves. A source patch is resampled
into the reference frame through the local linear part of the verified model, so the two
patches are compared at the same scale and orientation. Only the *shape* of the model is
used here; the position comes from the images. Whether refinement actually helps is
measured separately, on synthetic ground truth and on held-out points (see
``evaluation/validation.py``), and the original coordinates are always kept.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
from pydantic import BaseModel

from app.config import RefinementConfig

# Rejection reasons, in the order they are checked.
OUTSIDE = "patch outside the image"
INVALID = "patch contains invalid pixels"
FLAT = "patch has too little texture"
LIMIT = "optimum at the search limit"
NOT_CONCAVE = "correlation surface has no clear peak"
LOW_SCORE = "correlation below the minimum"
AMBIGUOUS = "a second peak is nearly as strong"
NO_CONVERGENCE = "estimator did not converge"
TOO_FAR = "shift larger than the search radius"
INCONSISTENT = "forward and backward estimates disagree"
REFINED = "refined"

_SYSTEMATIC_SHIFT_PX = 0.5  # a common shift this large is not measurement noise


class RefinementSummary(BaseModel):
    strategy: str
    representation: str
    attempted: int
    refined: int
    rejected: int
    rejected_by_reason: dict[str, int]
    median_displacement_px: float | None = None
    p90_displacement_px: float | None = None
    max_displacement_px: float | None = None
    median_score: float | None = None
    median_shift_x_px: float | None = None
    median_shift_y_px: float | None = None
    systematic_shift: bool = False  # the points moved together, see _coherent_shift
    # Filled in by the validation step when it has been measured.
    model_refit: bool = False
    note: str = (
        "Initial coordinates are kept alongside refined ones. Refined coordinates are not, by themselves, "
        "evidence of sub-pixel accuracy; see the held-out and synthetic validation figures."
    )


@dataclass
class RefinedPoints:
    source: np.ndarray  # (N, 2) unchanged
    reference_initial: np.ndarray  # (N, 2)
    reference_refined: np.ndarray  # (N, 2), equal to the initial value where not refined
    displacement: np.ndarray  # (N, 2), refined - initial
    score: np.ndarray  # (N,), correlation at the optimum; NaN when it could not be measured
    refined: np.ndarray  # (N,) bool
    reason: list[str]
    summary: RefinementSummary


def _sample_reference(image: np.ndarray, mask: np.ndarray, centre: np.ndarray, half: int):
    """float32 patch of side 2*half+1 centred exactly on ``centre``, with its valid fraction."""
    size = 2 * half + 1
    x0, y0 = int(np.floor(centre[0])) - half - 1, int(np.floor(centre[1])) - half - 1
    x1, y1 = x0 + size + 3, y0 + size + 3
    if x0 < 0 or y0 < 0 or x1 > image.shape[1] or y1 > image.shape[0]:
        return None, 0.0
    warp = np.array([[1.0, 0.0, half - (centre[0] - x0)], [0.0, 1.0, half - (centre[1] - y0)]])
    roi = image[y0:y1, x0:x1].astype(np.float32)
    patch = cv2.warpAffine(roi, warp, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    valid = cv2.warpAffine(mask[y0:y1, x0:x1], warp, (size, size), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    return patch, float((valid > 0).mean())


def _sample_source(image: np.ndarray, mask: np.ndarray, centre: np.ndarray, linear: np.ndarray, half: int):
    """Source patch resampled into the reference frame through ``linear`` (source -> reference)."""
    size = 2 * half + 1
    try:
        inverse = np.linalg.inv(linear)
    except np.linalg.LinAlgError:
        return None, 0.0
    reach = half * float(np.abs(inverse).sum(axis=1).max()) + 3.0
    x0, y0 = int(np.floor(centre[0] - reach)), int(np.floor(centre[1] - reach))
    x1, y1 = int(np.ceil(centre[0] + reach)) + 1, int(np.ceil(centre[1] + reach)) + 1
    if x0 < 0 or y0 < 0 or x1 > image.shape[1] or y1 > image.shape[0]:
        return None, 0.0
    offset = np.array([half, half], dtype=np.float64) - linear @ (centre - np.array([x0, y0], dtype=np.float64))
    warp = np.c_[linear, offset]
    roi = image[y0:y1, x0:x1].astype(np.float32)
    patch = cv2.warpAffine(roi, warp, (size, size), flags=cv2.INTER_LINEAR)
    valid = cv2.warpAffine(mask[y0:y1, x0:x1], warp, (size, size), flags=cv2.INTER_NEAREST)
    return patch, float((valid > 0).mean())


def _zncc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denominator) if denominator > 1e-6 else float("nan")


def _parabola(minus: float, centre: float, plus: float) -> float | None:
    """Sub-sample position of a peak from three samples, or None if they do not form a peak."""
    curvature = minus - 2.0 * centre + plus
    if curvature >= -1e-9:
        return None
    return float(0.5 * (minus - plus) / curvature)


def _estimate_ncc(template, window, search: int, cfg: RefinementConfig):
    """Normalised cross-correlation surface with a parabolic peak fit."""
    surface = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
    _, peak_value, _, (px, py) = cv2.minMaxLoc(surface)
    if px == 0 or py == 0 or px == surface.shape[1] - 1 or py == surface.shape[0] - 1:
        return None, None, LIMIT
    dx = _parabola(surface[py, px - 1], surface[py, px], surface[py, px + 1])
    dy = _parabola(surface[py - 1, px], surface[py, px], surface[py + 1, px])
    if dx is None or dy is None or abs(dx) > 1.0 or abs(dy) > 1.0:
        return None, None, NOT_CONCAVE
    if peak_value < cfg.min_score:
        return None, float(peak_value), LOW_SCORE
    # The runner-up is searched outside the peak's own lobe.
    masked = surface.copy()
    y0, y1 = max(0, py - 2), min(surface.shape[0], py + 3)
    x0, x1 = max(0, px - 2), min(surface.shape[1], px + 3)
    masked[y0:y1, x0:x1] = -1.0
    second = float(masked.max())
    if second > 0 and peak_value / second < cfg.min_peak_ratio:
        return None, float(peak_value), AMBIGUOUS
    return np.array([px - search + dx, py - search + dy]), float(peak_value), REFINED


def _estimate_lk(source_window, window, search: int, half: int, cfg: RefinementConfig):
    """Pyramidal Lucas-Kanade from the patch centre, with a forward-backward consistency check."""
    centre = np.array([[[search + half, search + half]]], dtype=np.float32)
    size = (2 * half + 1, 2 * half + 1)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.005)
    a = cv2.normalize(source_window, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    b = cv2.normalize(window, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    forward, status, _ = cv2.calcOpticalFlowPyrLK(a, b, centre, None, winSize=size, maxLevel=1, criteria=criteria)
    if status is None or not status[0, 0]:
        return None, None, NO_CONVERGENCE
    backward, status2, _ = cv2.calcOpticalFlowPyrLK(b, a, forward, None, winSize=size, maxLevel=1, criteria=criteria)
    if status2 is None or not status2[0, 0]:
        return None, None, NO_CONVERGENCE
    if np.linalg.norm(backward - centre) > 0.5:
        return None, None, INCONSISTENT
    shift = (forward - centre).reshape(2).astype(np.float64)
    if np.linalg.norm(shift) > search:
        return None, None, TOO_FAR
    return shift, None, REFINED


def _estimate_ecc(template, centre_patch, search: int, cfg: RefinementConfig):
    """Enhanced correlation coefficient maximisation for a pure translation."""
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-4)
    try:
        score, warp = cv2.findTransformECC(template, centre_patch, warp, cv2.MOTION_TRANSLATION, criteria, None, 3)
    except cv2.error:
        return None, None, NO_CONVERGENCE
    shift = np.array([warp[0, 2], warp[1, 2]], dtype=np.float64)
    if np.linalg.norm(shift) > search:
        return None, float(score), TOO_FAR
    if score < cfg.min_score:
        return None, float(score), LOW_SCORE
    return shift, float(score), REFINED


def _estimate_phase(template, centre_patch, search: int, cfg: RefinementConfig):
    """Windowed phase correlation."""
    window = cv2.createHanningWindow((template.shape[1], template.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(template, centre_patch, window)
    shift = np.array([dx, dy], dtype=np.float64)
    if np.linalg.norm(shift) > search:
        return None, float(response), TOO_FAR
    return shift, float(response), REFINED


def _coherent_shift(displacement: np.ndarray) -> tuple[float | None, float | None, bool]:
    """Median shift of the refined points, and whether they all moved the same way.

    Correcting measurement noise moves points in scattered directions. A correlation peak that
    follows shading instead of terrain moves them together: on rendered terrain with the Sun
    turned by 90 degrees, refinement on plain intensity shifted points by 4.5 px in one direction
    and made them much worse, while its correlation scores looked fine. Unlike error, this can be
    seen without knowing the truth.
    """
    if len(displacement) < 5:
        return None, None, False
    median = np.median(displacement, axis=0)
    size = float(np.linalg.norm(median))
    typical = float(np.median(np.linalg.norm(displacement, axis=1)))
    return float(median[0]), float(median[1]), bool(size > _SYSTEMATIC_SHIFT_PX and size > 0.5 * typical)


def refine_correspondences(
    source_image: np.ndarray,
    source_mask: np.ndarray,
    reference_image: np.ndarray,
    reference_mask: np.ndarray,
    source_points: np.ndarray,
    reference_points: np.ndarray,
    local_linear: Callable[[np.ndarray], np.ndarray],
    cfg: RefinementConfig,
    representation: str = "",
) -> RefinedPoints:
    """Refine each correspondence; ``local_linear`` gives (N, 2, 2) source -> reference maps."""
    src = np.asarray(source_points, dtype=np.float64).reshape(-1, 2)
    ref = np.asarray(reference_points, dtype=np.float64).reshape(-1, 2)
    n = len(src)
    half, search = cfg.patch_radius_px, cfg.search_radius_px
    refined_pts = ref.copy()
    score = np.full(n, np.nan)
    flags = np.zeros(n, dtype=bool)
    reasons: list[str] = []
    linear = local_linear(src) if n else np.zeros((0, 2, 2))

    for i in range(n):
        reason = _refine_one(i, src, ref, linear, source_image, source_mask, reference_image, reference_mask, half, search, cfg, refined_pts, score, flags)
        reasons.append(reason)

    displacement = refined_pts - ref
    magnitude = np.linalg.norm(displacement[flags], axis=1)
    by_reason: dict[str, int] = {}
    for reason in reasons:
        if reason != REFINED:
            by_reason[reason] = by_reason.get(reason, 0) + 1
    finite = score[flags & np.isfinite(score)]
    shift_x, shift_y, systematic = _coherent_shift(displacement[flags])
    summary = RefinementSummary(
        strategy=cfg.strategy,
        representation=representation,
        attempted=n,
        refined=int(flags.sum()),
        rejected=int(n - flags.sum()),
        rejected_by_reason=by_reason,
        median_displacement_px=float(np.median(magnitude)) if len(magnitude) else None,
        p90_displacement_px=float(np.percentile(magnitude, 90)) if len(magnitude) else None,
        max_displacement_px=float(magnitude.max()) if len(magnitude) else None,
        median_score=float(np.median(finite)) if len(finite) else None,
        median_shift_x_px=shift_x,
        median_shift_y_px=shift_y,
        systematic_shift=systematic,
    )
    return RefinedPoints(src, ref, refined_pts, displacement, score, flags, reasons, summary)


def _refine_one(
    i, src, ref, linear, source_image, source_mask, reference_image, reference_mask, half, search, cfg, refined_pts, score, flags
) -> str:
    strategy = cfg.strategy
    outer = half + search
    template, valid_a = _sample_source(source_image, source_mask, src[i], linear[i], half)
    window, valid_b = _sample_reference(reference_image, reference_mask, ref[i], outer)
    if template is None or window is None:
        return OUTSIDE
    if valid_a < cfg.min_valid_fraction or valid_b < cfg.min_valid_fraction:
        return INVALID
    if template.std() < cfg.min_texture_std or window.std() < cfg.min_texture_std:
        return FLAT

    if strategy == "ncc":
        shift, value, reason = _estimate_ncc(template, window, search, cfg)
    elif strategy == "lucas_kanade":
        source_window, _ = _sample_source(source_image, source_mask, src[i], linear[i], outer)
        if source_window is None:
            return OUTSIDE
        shift, value, reason = _estimate_lk(source_window, window, search, half, cfg)
    elif strategy == "ecc":
        centre_patch = window[search:search + 2 * half + 1, search:search + 2 * half + 1]
        shift, value, reason = _estimate_ecc(template, centre_patch, search, cfg)
    elif strategy == "phase":
        centre_patch = window[search:search + 2 * half + 1, search:search + 2 * half + 1]
        shift, value, reason = _estimate_phase(template, centre_patch, search, cfg)
    else:
        raise ValueError(f"Unknown refinement strategy: {strategy}")

    if value is not None:
        score[i] = value
    if shift is None:
        return reason

    candidate = ref[i] + shift
    # Every estimator is judged by the same measure at the point it proposes.
    moved, valid_c = _sample_reference(reference_image, reference_mask, candidate, half)
    if moved is None or valid_c < cfg.min_valid_fraction:
        return OUTSIDE
    measured = _zncc(template, moved)
    score[i] = measured
    if not np.isfinite(measured) or measured < cfg.min_score:
        return LOW_SCORE
    refined_pts[i] = candidate
    flags[i] = True
    return REFINED
