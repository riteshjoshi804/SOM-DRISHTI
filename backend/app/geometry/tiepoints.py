"""Spatially balanced tie points, chosen from verified inliers only.

Verified matches cluster where the imagery happens to be matchable. Selection thins dense
clusters so that no single region dominates a later fit or the published tie-point list.
It never adds anything: a grid cell without a verified match stays empty, and if too few
reliable points exist the result says so instead of inventing some.

Steps: (1) duplicate suppression, (2) confidence-aware ranking, (3) Adaptive Non-Maximal
Suppression (Brown, Szeliski & Winder 2005), (4) a per-cell cap, so that the final set
is limited by what the terrain offers rather than by what the strongest cluster offers.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from pydantic import BaseModel

from app.config import CoverageConfig, TiePointConfig
from app.evaluation.metrics import spatial_coverage

_ANMS_MAX_POINTS = 6000  # the suppression radius is quadratic in the number of points


class TiePointSummary(BaseModel):
    verified_inliers: int
    after_duplicate_suppression: int
    selected: int
    below_minimum: bool
    minimum_requested: int
    maximum_requested: int
    grid_rows: int
    grid_cols: int
    max_per_cell: int
    # Over the whole valid source image
    valid_cells: int
    occupied_cells: int
    source_coverage: float | None
    inlier_source_coverage: float | None  # same grid, all verified inliers before selection
    # Over the estimated common overlap
    overlap_cells: int | None = None
    occupied_overlap_cells: int | None = None
    overlap_coverage: float | None = None
    inlier_overlap_coverage: float | None = None
    # Distribution
    tie_points_per_km2: float | None = None
    tie_points_per_megapixel: float | None = None
    distribution_score: float | None = None
    distribution_definition: str = (
        "Normalised entropy of the tie-point counts over the valid grid cells of the overlap: 1 means equally "
        "many points in every cell, 0 means all points in one cell. Empty cells lower it."
    )
    hull_fraction_of_overlap: float | None = None
    note: str = "Selection only removes verified points; empty grid cells are never filled."


def score_points(confidence: np.ndarray, errors: np.ndarray, threshold_px: float) -> np.ndarray:
    """Higher is better: matcher confidence, discounted by up to half at the inlier threshold."""
    fit = 1.0 - 0.5 * np.clip(np.asarray(errors, dtype=np.float64) / threshold_px, 0.0, 1.0)
    return np.asarray(confidence, dtype=np.float64) * fit


def suppress_duplicates(points: np.ndarray, scores: np.ndarray, radius_px: float) -> np.ndarray:
    """Indices kept after greedy suppression: the best point in each ``radius_px`` neighbourhood wins."""
    n = len(points)
    if n == 0 or radius_px <= 0:
        return np.arange(n)
    order = np.argsort(-scores, kind="stable")
    # A hash grid limits each lookup to the neighbouring cells.
    cell = radius_px
    occupied: dict[tuple[int, int], list[int]] = {}
    kept: list[int] = []
    radius_sq = radius_px * radius_px
    for i in order:
        cx, cy = int(math.floor(points[i, 0] / cell)), int(math.floor(points[i, 1] / cell))
        clash = False
        for ax in (cx - 1, cx, cx + 1):
            for ay in (cy - 1, cy, cy + 1):
                for j in occupied.get((ax, ay), ()):
                    if (points[i, 0] - points[j, 0]) ** 2 + (points[i, 1] - points[j, 1]) ** 2 < radius_sq:
                        clash = True
                        break
                if clash:
                    break
            if clash:
                break
        if not clash:
            occupied.setdefault((cx, cy), []).append(int(i))
            kept.append(int(i))
    return np.sort(np.array(kept, dtype=np.int64))


def anms_radii(points: np.ndarray, scores: np.ndarray, robust: float = 0.9) -> np.ndarray:
    """Suppression radius per point: distance to the nearest clearly stronger point.

    A point survives to a large radius when nothing clearly stronger is nearby, so sorting by
    radius yields a set spread over the image while still favouring strong points. Points close
    to the top score have nothing "clearly stronger" at all; they are measured against the
    points ranked above them, or a dense cluster of near-equal scores could never be thinned
    (LoFTR confidences pile up just below 1).
    """
    n = len(points)
    radii = np.full(n, np.inf)
    if n < 2:
        return radii
    order = np.argsort(-scores, kind="stable")
    pts, sc = points[order], scores[order]
    for i in range(1, n):
        stronger = robust * sc[:i] > sc[i]
        d = (pts[:i][stronger] if stronger.any() else pts[:i]) - pts[i]
        radii[order[i]] = math.sqrt(float((d * d).sum(axis=1).min()))
    return radii


def _distribution(counts: np.ndarray, cells: np.ndarray) -> float | None:
    usable = counts[cells]
    total = int(usable.sum())
    n = int(cells.sum())
    if total == 0 or n < 2:
        return None
    p = usable[usable > 0] / total
    return float(-(p * np.log(p)).sum() / math.log(n))


def select_tie_points(
    source_points: np.ndarray,
    confidence: np.ndarray,
    errors: np.ndarray,
    threshold_px: float,
    valid_mask: np.ndarray,
    cfg: TiePointConfig,
    overlap_mask: np.ndarray | None = None,
    gsd_m: float | None = None,
) -> tuple[np.ndarray, TiePointSummary]:
    """Indices (into the given arrays) of the selected tie points, and their measurements."""
    points = np.asarray(source_points, dtype=np.float64).reshape(-1, 2)
    scores = score_points(confidence, errors, threshold_px)
    grid = CoverageConfig(grid_rows=cfg.grid_rows, grid_cols=cfg.grid_cols)
    h, w = valid_mask.shape

    kept = suppress_duplicates(points, scores, cfg.min_separation_px)
    pts, sc = points[kept], scores[kept]

    # Points beyond the ANMS budget can only be the weakest ones, in a much denser set than the output.
    if len(pts) > _ANMS_MAX_POINTS:
        strongest = np.argsort(-sc, kind="stable")[:_ANMS_MAX_POINTS]
        kept, pts, sc = kept[strongest], pts[strongest], sc[strongest]

    radii = anms_radii(pts, sc, cfg.anms_robust)
    order = np.lexsort((-sc, -radii))
    rows = np.clip((pts[:, 1] / h * cfg.grid_rows).astype(int), 0, cfg.grid_rows - 1)
    cols = np.clip((pts[:, 0] / w * cfg.grid_cols).astype(int), 0, cfg.grid_cols - 1)
    per_cell = np.zeros((cfg.grid_rows, cfg.grid_cols), dtype=int)
    chosen: list[int] = []
    for i in order:
        if len(chosen) >= cfg.max_tie_points:
            break
        if per_cell[rows[i], cols[i]] < cfg.max_per_cell:
            per_cell[rows[i], cols[i]] += 1
            chosen.append(int(i))
    selected = kept[np.array(chosen, dtype=np.int64)] if chosen else np.zeros(0, np.int64)
    selected = np.sort(selected)

    all_cov = spatial_coverage(points, valid_mask, grid, overlap_mask)
    sel_cov = spatial_coverage(points[selected], valid_mask, grid, overlap_mask)

    counts = np.asarray(sel_cov.cell_counts)
    in_overlap = np.asarray(sel_cov.cell_in_overlap) if sel_cov.cell_in_overlap is not None else np.asarray(sel_cov.cell_valid)
    area_mask = overlap_mask if overlap_mask is not None else valid_mask
    area_px = float(area_mask.sum())
    density_km2 = None
    if gsd_m and area_px > 0:
        density_km2 = len(selected) / (area_px * (gsd_m / 1000.0) ** 2)
    hull_fraction = None
    if len(selected) >= 3 and area_px > 0:
        hull = cv2.convexHull(points[selected].astype(np.float32))
        hull_fraction = float(min(1.0, cv2.contourArea(hull) / area_px))

    occupied_overlap = None
    if sel_cov.cell_in_overlap is not None:
        occupied_overlap = int(((counts > 0) & in_overlap).sum())

    return selected, TiePointSummary(
        verified_inliers=len(points),
        after_duplicate_suppression=len(kept),
        selected=len(selected),
        below_minimum=len(selected) < cfg.min_tie_points,
        minimum_requested=cfg.min_tie_points,
        maximum_requested=cfg.max_tie_points,
        grid_rows=cfg.grid_rows,
        grid_cols=cfg.grid_cols,
        max_per_cell=cfg.max_per_cell,
        valid_cells=sel_cov.valid_cells,
        occupied_cells=sel_cov.occupied_cells,
        source_coverage=sel_cov.coverage,
        inlier_source_coverage=all_cov.coverage,
        overlap_cells=sel_cov.overlap_cells,
        occupied_overlap_cells=occupied_overlap,
        overlap_coverage=sel_cov.coverage_within_overlap,
        inlier_overlap_coverage=all_cov.coverage_within_overlap,
        tie_points_per_km2=density_km2,
        tie_points_per_megapixel=(len(selected) / (area_px / 1e6)) if area_px > 0 else None,
        distribution_score=_distribution(counts, in_overlap),
        hull_fraction_of_overlap=hull_fraction,
    )
