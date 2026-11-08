"""Do the two patches at a claimed correspondence actually look alike?

A check that is independent of the matcher. Learned matchers happily pair a crater with a
different crater of similar size, or a black shadow with dark terrain, and such matches can
agree with some geometry by chance. Comparing the local *gradient magnitude* of the two
patches exposes them: it ignores the sign of the contrast, which a changed Sun can reverse,
while still requiring the same structure to be present.

Measured on official scene 2470502 (zero-mean normalised correlation, 31 px patches): real
inliers had a median of +0.66, while matches on deliberately wrong ground had +0.05. Black
patches have no gradient to compare, so they can never pass. A threshold of 0.25 kept 86 % of
the matches in clear translation peaks and 11 % of the rest; the separation peaks at 0.20-0.25.
"""
from __future__ import annotations

import cv2
import numpy as np
from pydantic import BaseModel


class AppearanceSummary(BaseModel):
    label: str = "gradient patch similarity at matched points; independent of the matcher"
    threshold: float
    patch_radius_px: int
    measured: int
    unmeasurable: int  # flat patches, or too close to the image edge
    passed: int
    median_zncc: float | None = None  # over every measurable match, including the rejected ones
    p10_zncc: float | None = None
    p90_zncc: float | None = None
    median_zncc_passed: float | None = None  # over the matches that were kept


def gradient_map(image: np.ndarray) -> np.ndarray:
    """Gradient magnitude (float32) of a slightly smoothed image."""
    smooth = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 1.0)
    return cv2.magnitude(cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3))


def patch_zncc(
    map0: np.ndarray, map1: np.ndarray, points0: np.ndarray, points1: np.ndarray, half: int, min_std: float = 1.0
) -> np.ndarray:
    """ZNCC per point pair, NaN where a patch leaves the image or has no structure to compare.

    Both images must already share scale and orientation; the guided search provides that.
    """
    p0 = np.rint(np.asarray(points0, dtype=np.float64)).astype(np.int64).reshape(-1, 2)
    p1 = np.rint(np.asarray(points1, dtype=np.float64)).astype(np.int64).reshape(-1, 2)
    h0, w0 = map0.shape
    h1, w1 = map1.shape
    out = np.full(len(p0), np.nan)
    size = 2 * half + 1
    for i in range(len(p0)):
        x0, y0, x1, y1 = p0[i, 0] - half, p0[i, 1] - half, p1[i, 0] - half, p1[i, 1] - half
        if x0 < 0 or y0 < 0 or x1 < 0 or y1 < 0 or x0 + size > w0 or y0 + size > h0 or x1 + size > w1 or y1 + size > h1:
            continue
        a = map0[y0 : y0 + size, x0 : x0 + size]
        b = map1[y1 : y1 + size, x1 : x1 + size]
        sa, sb = float(a.std()), float(b.std())
        if sa < min_std or sb < min_std:
            continue
        out[i] = float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))
    return out


def summarise(scores: np.ndarray, threshold: float, half: int) -> AppearanceSummary:
    finite = scores[np.isfinite(scores)]
    kept = finite[finite >= threshold]
    return AppearanceSummary(
        threshold=threshold,
        patch_radius_px=half,
        measured=len(finite),
        unmeasurable=int(len(scores) - len(finite)),
        passed=int((finite >= threshold).sum()),
        median_zncc=float(np.median(finite)) if len(finite) else None,
        p10_zncc=float(np.percentile(finite, 10)) if len(finite) else None,
        p90_zncc=float(np.percentile(finite, 90)) if len(finite) else None,
        median_zncc_passed=float(np.median(kept)) if len(kept) else None,
    )
