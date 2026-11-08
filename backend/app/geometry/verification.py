"""The geometric verification every matcher branch goes through.

Matcher-specific filtering happens before this point. Here every branch gets the
same two-stage robust consensus (seed + expansion, MAGSAC with RANSAC fallback),
the same plausibility checks, and, for elongated images, the same test of whether
a smooth along-track model explains clearly more of the candidates than one global
transformation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.config import GeometryConfig, GuidedSearchConfig, ValidationConfig
from app.geometry.consensus import Consensus, find_consensus
from app.geometry.estimation import apply_transform, distinct_points, fit_least_squares
from app.geometry.strip_model import (
    StripCheck,
    StripModel,
    check_strip,
    choose_smoothness,
    fit_strip,
    leave_block_out_errors,
    track_residuals,
)
from app.geometry.validation import TransformCheck, check_transform


def mirror_matrix(width: int) -> np.ndarray:
    """Maps pixel x of the left-right flipped image back to the original image (its own inverse)."""
    return np.array([[-1.0, 0.0, width - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class VerifiedGeometry:
    """Source -> reference mapping accepted by robust estimation (validity is in ``check``)."""

    kind: str  # similarity | affine | homography | strip_spline
    matrix: np.ndarray  # 3x3; for a strip model, its best affine approximation
    method: str
    inlier_idx: np.ndarray  # indices into the candidate arrays
    check: TransformCheck  # checks of ``matrix``, always in the unmirrored frame
    mirrored: bool
    strip: StripModel | None = None
    strip_check: StripCheck | None = None
    seed_pool: int = 0
    seed_inliers: int = 0
    log: list[str] = field(default_factory=list)
    global_inliers: int = 0  # support of the best single global transformation

    @property
    def valid(self) -> bool:
        if self.strip is not None:
            return self.strip_check is not None and self.strip_check.valid
        return self.check.valid

    @property
    def reasons(self) -> list[str]:
        if self.strip is not None:
            return list(self.strip_check.reasons) if self.strip_check else ["strip model not checked"]
        return list(self.check.reasons)

    def apply(self, points: np.ndarray) -> np.ndarray:
        if self.strip is not None:
            return self.strip.apply(points)
        return apply_transform(self.matrix, points)

    def errors(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        return np.linalg.norm(self.apply(src) - np.asarray(dst, dtype=np.float64), axis=1)


def _distinct(idx: np.ndarray, src: np.ndarray, dst: np.ndarray, rank: np.ndarray, sep: float) -> np.ndarray:
    if len(idx) == 0:
        return idx
    idx = idx[distinct_points(src[idx], rank[idx], sep)]
    return idx[distinct_points(dst[idx], rank[idx], sep)]


def grow_strip_model(
    src: np.ndarray,
    dst: np.ndarray,
    rank: np.ndarray,
    start: np.ndarray,
    source_size: tuple[int, int],
    guided: GuidedSearchConfig,
    geometry: GeometryConfig,
    log: list[str],
) -> tuple[StripModel, np.ndarray] | None:
    """Grow from a global model ``start`` to a strip model and its distinct tie points."""
    residual = dst - apply_transform(start, src)
    track = track_residuals(
        src, residual, source_size, guided.tile_vote_bin_px, guided.tile_min_votes, guided.strip_knot_spacing_px
    )
    if len(track) < guided.strip_min_tiles:
        log.append(f"strip: only {len(track)} along-track groups agree (need {guided.strip_min_tiles})")
        return None
    axis = 1 if source_size[1] >= source_size[0] else 0

    # First approximation: the affine plus the tracked deviation, interpolated along-track.
    along = np.array([t[0] for t in track])
    values = np.array([t[1] for t in track])
    predicted = np.c_[
        np.interp(src[:, axis], along, values[:, 0]),
        np.interp(src[:, axis], along, values[:, 1]),
    ]
    errors = np.linalg.norm(residual - predicted, axis=1)
    log.append(
        f"strip: track over {len(track)} groups, deviation up to {np.abs(values).max():.0f} px, "
        f"{int(sum(t[2] for t in track))} votes"
    )

    sep = geometry.tie_point_min_separation_px
    radii = [max(r, geometry.reproj_threshold_px) for r in guided.strip_radii_px] + [geometry.reproj_threshold_px] * 2
    model: StripModel | None = None
    lam = guided.strip_smoothness_candidates[len(guided.strip_smoothness_candidates) // 2]
    for radius in radii:
        idx = np.flatnonzero(errors < radius)
        idx = _distinct(idx, src, dst, rank, sep)
        if len(idx) < 8:
            log.append(f"strip: only {len(idx)} candidates within {radius:g} px")
            return None
        lam, scores = choose_smoothness(
            src[idx], dst[idx], source_size, guided.strip_knot_spacing_px, guided.strip_smoothness_candidates
        )
        model = fit_strip(src[idx], dst[idx], source_size, guided.strip_knot_spacing_px, lam)
        if model is None:
            log.append("strip: fit failed")
            return None
        errors = model.errors(src, dst)

    final = _distinct(np.flatnonzero(errors < geometry.reproj_threshold_px), src, dst, rank, sep)
    if len(final) < 8:
        log.append(f"strip: only {len(final)} tie points at the final threshold")
        return None
    log.append(f"strip: {len(final)} tie points within {geometry.reproj_threshold_px:g} px, smoothness {lam:g}")
    return model, final


def _strip_predicts_better(
    src: np.ndarray,
    dst: np.ndarray,
    tie: np.ndarray,
    result: VerifiedGeometry,
    strip: StripModel,
    source_size: tuple[int, int],
    guided: GuidedSearchConfig,
    geometry_threshold: float,
    log: list[str],
) -> bool:
    """Decide between the models by how well each predicts ground left out of its fit.

    Counting inliers would be the wrong test: a single strong tile can give one global model
    hundreds of tightly fitting points while the rest of the strip is wrong by tens of pixels.
    Both models are tested on the same points: every candidate close to either of them. The
    block under test never takes part in the fit, so a point cannot help the model that
    happened to pass near it.
    """
    reach = 4.0 * geometry_threshold
    near_strip = strip.errors(src, dst) < reach
    near_global = result.errors(src, dst) < reach
    union = np.flatnonzero(near_strip | near_global)
    frame_src, frame_dst = src[union], dst[union]

    def fit_global(train_src, train_dst):
        matrix = fit_least_squares(train_src, train_dst, "affine" if result.kind == "similarity" else result.kind)
        return None if matrix is None else (lambda points: apply_transform(matrix, points))

    def fit_flexible(train_src, train_dst):
        model = fit_strip(train_src, train_dst, source_size, guided.strip_knot_spacing_px, strip.smoothness)
        return None if model is None else model.apply

    errors_global = leave_block_out_errors(frame_src, frame_dst, source_size, fit_global)
    errors_strip = leave_block_out_errors(frame_src, frame_dst, source_size, fit_flexible)
    if len(errors_global) < 2 or len(errors_strip) < 2:
        adopt = len(tie) >= guided.strip_min_gain * len(result.inlier_idx)
        log.append(f"strip: too few blocks to compare ({len(errors_strip)}); by tie points: {len(tie)} vs {len(result.inlier_idx)}")
        return adopt

    def mean_block_rmse(blocks: list[np.ndarray]) -> float:
        return float(np.mean([np.sqrt(np.mean(b**2)) for b in blocks]))  # each block counts the same

    rmse_global, rmse_strip = mean_block_rmse(errors_global), mean_block_rmse(errors_strip)
    log.append(
        f"strip: unseen-block RMSE {rmse_strip:.1f} px vs {rmse_global:.1f} px for the global model "
        f"({len(errors_strip)} blocks, {len(union)} points)"
    )
    if rmse_strip >= rmse_global:
        log.append("strip: not adopted, it does not predict unseen ground better")
        return False
    return True


def verify_candidates(
    src: np.ndarray,
    dst: np.ndarray,
    rank: np.ndarray,
    source_size: tuple[int, int],
    reference_size: tuple[int, int],
    geometry: GeometryConfig,
    validation: ValidationConfig,
    guided: GuidedSearchConfig | None = None,
    *,
    mirrored: bool = False,
    start: np.ndarray | None = None,
) -> tuple[VerifiedGeometry | None, list[str]]:
    """Verified geometry for candidates in ORIGINAL source pixels, or None.

    With ``mirrored`` the consensus runs in the left-right flipped source frame, where a
    valid transform preserves orientation, and the result is converted back. ``start``
    is an optional prior (3x3, original frame) to grow the strip model from when no
    global consensus exists on its own.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    rank = np.asarray(rank, dtype=np.float64).reshape(-1)
    w, h = source_size
    log: list[str] = []

    frame = src.copy()
    if mirrored:
        frame[:, 0] = (w - 1) - frame[:, 0]
    mirror = mirror_matrix(w)

    consensus: Consensus | None = None
    if len(src) >= 3:
        consensus, consensus_log = find_consensus(frame, dst, rank, geometry, validation, source_size, reference_size)
        log.extend(consensus_log)

    result: VerifiedGeometry | None = None
    if consensus is not None:
        check = check_transform(consensus.matrix, source_size, reference_size, validation)
        matrix = consensus.matrix @ mirror if mirrored else consensus.matrix
        result = VerifiedGeometry(
            kind=consensus.model,
            matrix=matrix,
            method=consensus.method,
            inlier_idx=consensus.inlier_idx,
            check=check,
            mirrored=mirrored,
            seed_pool=consensus.seed_pool,
            seed_inliers=consensus.seed_inliers,
            log=log,
            global_inliers=len(consensus.inlier_idx),
        )

    use_strip = (
        guided is not None
        and guided.use_strip_model
        and max(w, h) / max(min(w, h), 1) >= guided.strip_min_aspect
        and len(src) >= guided.tile_min_votes
    )
    if not use_strip:
        return result, log

    origin = result.matrix if result is not None and result.check.valid else start
    if origin is None:
        return result, log
    grown = grow_strip_model(src, dst, rank, origin, source_size, guided, geometry, log)
    if grown is None:
        return result, log
    strip, tie = grown
    expected = -1 if mirrored else 1
    strip_check = check_strip(strip, reference_size, validation, expected)
    if not strip_check.valid:
        log.append("strip: rejected: " + "; ".join(strip_check.reasons))
        return result, log

    global_support = len(result.inlier_idx) if result is not None and result.check.valid else 0
    if global_support and not _strip_predicts_better(
        src, dst, tie, result, strip, source_size, guided, geometry.reproj_threshold_px, log
    ):
        return result, log

    affine = strip.best_affine()
    frame_affine = affine @ mirror if mirrored else affine
    log.append(f"strip: adopted ({len(tie)} tie points vs {global_support} for the global model)")
    return (
        VerifiedGeometry(
            kind="strip_spline",
            matrix=affine,
            method=result.method if result is not None else "P-spline with iterative trimming",
            inlier_idx=tie,
            check=check_transform(frame_affine, source_size, reference_size, validation),
            mirrored=mirrored,
            strip=strip,
            strip_check=strip_check,
            seed_pool=result.seed_pool if result is not None else 0,
            seed_inliers=result.seed_inliers if result is not None else 0,
            log=log,
            global_inliers=global_support,
        ),
        log,
    )
