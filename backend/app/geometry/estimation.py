"""Robust estimation of the source -> reference transformation.

All transforms are handled as 3x3 matrices (affine models have last row
``0 0 1``) mapping source pixel (x, y) to reference pixel (x, y).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.config import GeometryConfig

MIN_POINTS = {"similarity": 2, "affine": 3, "homography": 4}


@dataclass
class TransformEstimate:
    matrix: np.ndarray  # 3x3 float64
    model: str  # similarity | affine | homography
    method: str  # robust estimator actually used
    inlier_mask: np.ndarray  # bool over the points passed in

    @property
    def inlier_count(self) -> int:
        return int(self.inlier_mask.sum())


def to_3x3(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    return m if m.shape == (3, 3) else np.vstack([m, [0.0, 0.0, 1.0]])


def apply_transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Map (N, 2) points through a 3x3 matrix (with perspective division)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homog = np.c_[pts, np.ones(len(pts))] @ matrix.T
    w = homog[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return homog[:, :2] / w


def reprojection_errors(matrix: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    return np.linalg.norm(apply_transform(matrix, src) - np.asarray(dst, dtype=np.float64), axis=1)


def _robust_flag(method: str) -> tuple[int, str]:
    if method == "magsac" and hasattr(cv2, "USAC_MAGSAC"):
        return cv2.USAC_MAGSAC, "USAC_MAGSAC"
    return cv2.RANSAC, "RANSAC"


def estimate_transform(
    src: np.ndarray,
    dst: np.ndarray,
    model: str,
    cfg: GeometryConfig,
    *,
    method: str | None = None,
    threshold: float | None = None,
    max_iters: int | None = None,
) -> TransformEstimate | None:
    """Fit ``model`` robustly. Returns None when no model can be estimated."""
    src = np.ascontiguousarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.ascontiguousarray(dst, dtype=np.float64).reshape(-1, 2)
    if len(src) < MIN_POINTS[model]:
        return None
    flag, name = _robust_flag(method or cfg.method)
    thr = threshold or cfg.reproj_threshold_px
    iters = max_iters or cfg.max_iters

    def run(flag_: int):
        if model == "homography":
            return cv2.findHomography(src, dst, flag_, thr, maxIters=iters, confidence=cfg.confidence)
        fn = cv2.estimateAffinePartial2D if model == "similarity" else cv2.estimateAffine2D
        return fn(src, dst, method=flag_, ransacReprojThreshold=thr, maxIters=iters, confidence=cfg.confidence)

    try:
        matrix, mask = run(flag)
    except cv2.error:
        matrix, mask = None, None
    if matrix is None and flag != cv2.RANSAC:
        # Not every OpenCV build accepts USAC flags for every model.
        name = "RANSAC"
        try:
            matrix, mask = run(cv2.RANSAC)
        except cv2.error:
            matrix, mask = None, None
    if matrix is None or mask is None or not np.all(np.isfinite(matrix)):
        return None
    return TransformEstimate(
        matrix=to_3x3(matrix), model=model, method=name, inlier_mask=mask.ravel().astype(bool)
    )


def fit_least_squares(src: np.ndarray, dst: np.ndarray, model: str) -> np.ndarray | None:
    """Non-robust fit on points already believed to be inliers."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if len(src) < MIN_POINTS[model]:
        return None
    if model == "homography":
        matrix, _ = cv2.findHomography(src, dst, 0)
        return None if matrix is None else to_3x3(matrix)
    if model == "similarity":
        matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        return None if matrix is None else to_3x3(matrix)
    sol, *_ = np.linalg.lstsq(np.c_[src, np.ones(len(src))], dst, rcond=None)
    return to_3x3(sol.T)


def distinct_points(points: np.ndarray, priority: np.ndarray, min_distance_px: float = 2.0) -> np.ndarray:
    """Indices of spatially distinct points, keeping the best ``priority`` (lowest) per cell.

    SIFT emits several keypoints at one location (different orientations or
    scales). Counting each as a separate tie point would overstate support.
    """
    if len(points) == 0:
        return np.zeros(0, np.int64)
    cells = np.floor(np.asarray(points) / min_distance_px).astype(np.int64)
    order = np.argsort(priority, kind="stable")
    _, first = np.unique(cells[order], axis=0, return_index=True)
    return np.sort(order[first])
