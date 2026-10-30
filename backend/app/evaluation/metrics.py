"""Metrics computed from the actual registration result. Nothing here is a constant.

Important: every error figure below is derived from the *matches themselves*.
None of them is an independent, ground-truth registration accuracy.
"""
from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel

from app.config import CoverageConfig
from app.geometry.estimation import apply_transform, fit_least_squares, reprojection_errors


class ReprojectionStats(BaseModel):
    rmse_px: float | None = None
    median_px: float | None = None
    max_px: float | None = None
    count: int = 0


class SpatialCoverage(BaseModel):
    grid_rows: int
    grid_cols: int
    valid_cells: int
    occupied_cells: int
    coverage: float | None  # occupied / valid  ("Image-space Spatial Coverage")
    overlap_cells: int | None = None  # valid cells that can overlap the reference at all
    coverage_within_overlap: float | None = None
    cell_counts: list[list[int]]
    cell_valid: list[list[bool]]
    cell_in_overlap: list[list[bool]] | None = None


class MetadataConsistency(BaseModel):
    centre_offset_px: float
    centre_offset_m: float | None
    mean_corner_offset_px: float
    max_corner_offset_px: float
    mean_corner_offset_m: float | None
    rotation_difference_deg: float
    scale_ratio: float
    mirror_agrees: bool
    orientation_agrees: bool
    note: str = (
        "Comparison of the image-based result with the footprint predicted from XML/GCP "
        "metadata. The metadata is system-level pointing knowledge, not ground truth."
    )


def reprojection_stats(matrix: np.ndarray, src: np.ndarray, dst: np.ndarray) -> ReprojectionStats:
    if len(src) == 0:
        return ReprojectionStats()
    err = reprojection_errors(matrix, src, dst)
    return ReprojectionStats(
        rmse_px=float(np.sqrt(np.mean(err**2))),
        median_px=float(np.median(err)),
        max_px=float(err.max()),
        count=int(len(err)),
    )


def cross_validated_rmse(
    src: np.ndarray, dst: np.ndarray, model: str, folds: int = 5, seed: int = 0
) -> float | None:
    """k-fold reprojection RMSE: fit on k-1 folds of the inliers, test on the rest.

    Less optimistic than the in-sample RMSE because every error is measured on
    points the model was not fitted to. It is still *not* ground-truth accuracy:
    the points were selected as inliers in the first place.
    """
    n = len(src)
    min_pts = {"similarity": 2, "affine": 3, "homography": 4}[model]
    if n < max(folds, 2 * min_pts + 2):
        return None
    order = np.random.default_rng(seed).permutation(n)
    sq_errors: list[np.ndarray] = []
    for k in range(folds):
        test = order[k::folds]
        train = np.setdiff1d(order, test, assume_unique=True)
        matrix = fit_least_squares(src[train], dst[train], model)
        if matrix is None:
            return None
        sq_errors.append(reprojection_errors(matrix, src[test], dst[test]) ** 2)
    return float(np.sqrt(np.mean(np.concatenate(sq_errors))))


def spatial_coverage(
    inlier_points: np.ndarray,
    valid_mask: np.ndarray,
    cfg: CoverageConfig,
    overlap_mask: np.ndarray | None = None,
) -> SpatialCoverage:
    """Image-space Spatial Coverage of the accepted tie points over the source image.

    The source is divided into ``grid_rows x grid_cols`` cells. A cell is *valid*
    when enough of its pixels hold data, and *occupied* when at least one
    accepted tie point falls in it. ``coverage = occupied / valid``.

    With partial overlap some valid cells can never be matched. When
    ``overlap_mask`` (source pixels that land on valid reference pixels) is
    given, ``coverage_within_overlap`` restricts the denominator to those cells.

    This is coverage of the *image grid*, not of any geographic area.
    """
    h, w = valid_mask.shape
    rows, cols = cfg.grid_rows, cfg.grid_cols
    y_edges = np.linspace(0, h, rows + 1)
    x_edges = np.linspace(0, w, cols + 1)

    def cell_fraction(mask: np.ndarray) -> np.ndarray:
        out = np.zeros((rows, cols))
        for r in range(rows):
            y0, y1 = int(round(y_edges[r])), max(int(round(y_edges[r + 1])), int(round(y_edges[r])) + 1)
            for c in range(cols):
                x0, x1 = int(round(x_edges[c])), max(int(round(x_edges[c + 1])), int(round(x_edges[c])) + 1)
                out[r, c] = mask[y0:y1, x0:x1].mean()
        return out

    valid = cell_fraction(valid_mask) >= cfg.min_valid_fraction
    counts = np.zeros((rows, cols), dtype=int)
    if len(inlier_points):
        pts = np.asarray(inlier_points, dtype=np.float64)
        ci = np.clip((pts[:, 0] / w * cols).astype(int), 0, cols - 1)
        ri = np.clip((pts[:, 1] / h * rows).astype(int), 0, rows - 1)
        np.add.at(counts, (ri, ci), 1)
    occupied = (counts > 0) & valid
    n_valid = int(valid.sum())

    in_overlap = None
    n_overlap = None
    within = None
    if overlap_mask is not None:
        in_overlap = valid & (cell_fraction(overlap_mask) >= cfg.min_valid_fraction)
        n_overlap = int(in_overlap.sum())
        within = float((occupied & in_overlap).sum() / n_overlap) if n_overlap else None

    return SpatialCoverage(
        grid_rows=rows,
        grid_cols=cols,
        valid_cells=n_valid,
        occupied_cells=int(occupied.sum()),
        coverage=float(occupied.sum() / n_valid) if n_valid else None,
        overlap_cells=n_overlap,
        coverage_within_overlap=within,
        cell_counts=counts.tolist(),
        cell_valid=valid.tolist(),
        cell_in_overlap=None if in_overlap is None else in_overlap.tolist(),
    )


def expected_chance_inliers(candidates: int, threshold_px: float, reference_valid_px: int) -> float | None:
    """Tie points expected to agree with a *given* model purely by chance.

    Each candidate is a global nearest-neighbour match, so a wrong one lands
    anywhere in the valid reference area; the probability that it falls within
    ``threshold_px`` of where the model predicts is ``pi*t^2 / area``.
    """
    if reference_valid_px <= 0:
        return None
    return float(candidates * math.pi * threshold_px**2 / reference_valid_px)


def metadata_consistency(
    matrix: np.ndarray,
    predicted_affine: list[list[float]],
    source_size: tuple[int, int],
    reference_gsd_m: float | None,
) -> MetadataConsistency:
    w, h = source_size
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float64)
    pred = np.vstack([np.asarray(predicted_affine, dtype=np.float64), [0, 0, 1]])
    est_corners, pred_corners = apply_transform(matrix, corners), apply_transform(pred, corners)
    offsets = np.linalg.norm(est_corners - pred_corners, axis=1)
    # The footprint centre is insensitive to how the array is oriented inside it.
    centre = float(np.linalg.norm(est_corners.mean(axis=0) - pred_corners.mean(axis=0)))

    def rot_scale_det(m: np.ndarray) -> tuple[float, float, float]:
        lin = (m / m[2, 2])[:2, :2]
        det = float(np.linalg.det(lin))
        return float(np.degrees(np.arctan2(lin[1, 0], lin[0, 0]))), float(np.sqrt(abs(det))), det

    r_est, s_est, d_est = rot_scale_det(matrix)
    r_pred, s_pred, d_pred = rot_scale_det(pred)
    rot_diff = float((r_est - r_pred + 180.0) % 360.0 - 180.0)
    mirror_agrees = bool(np.sign(d_est) == np.sign(d_pred))
    mean_px = float(offsets.mean())
    gsd = reference_gsd_m
    return MetadataConsistency(
        centre_offset_px=centre,
        centre_offset_m=None if gsd is None else centre * gsd,
        mean_corner_offset_px=mean_px,
        max_corner_offset_px=float(offsets.max()),
        mean_corner_offset_m=None if gsd is None else mean_px * gsd,
        rotation_difference_deg=rot_diff,
        scale_ratio=float(s_est / s_pred) if s_pred > 0 else float("nan"),
        mirror_agrees=mirror_agrees,
        orientation_agrees=mirror_agrees and abs(rot_diff) < 45.0,
    )
