"""Two-stage geometric verification for very low inlier ratios.

On the official OHRC / LRO-NAC pairs only a few percent of ratio-test
survivors are correct (the Sun geometry differs, so most small craters match
*some other* small crater). Plain RANSAC over the whole set is unreliable
there, so verification is split in two:

1. **Seed** - affine RANSAC on the most distinctive matches (lowest Lowe
   ratio) first. The pool is widened only if no consensus appears.
2. **Expansion** - every ratio-filtered match lying near the seed model is
   re-fitted, tightening to the final reprojection threshold.

The candidates are *global* nearest-neighbour matches: a wrong one lands
anywhere in the reference, so the number that agree with the model by chance
is tiny and can be stated analytically (see ``expected_chance_inliers``).
No locally-searched "guided" matches are added - with keypoints this dense,
those agree with almost any model and would inflate the inlier count.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.config import GeometryConfig, ValidationConfig
from app.geometry.estimation import (
    TransformEstimate,
    distinct_points,
    estimate_transform,
    reprojection_errors,
)
from app.geometry.validation import check_transform


@dataclass
class Consensus:
    matrix: np.ndarray  # 3x3, source -> reference
    model: str
    method: str
    inlier_idx: np.ndarray  # indices into the match arrays, spatially distinct
    seed_pool: int
    seed_inliers: int
    log: list[str] = field(default_factory=list)


def _tie_points(idx: np.ndarray, src: np.ndarray, dst: np.ndarray, ratios: np.ndarray, sep: float) -> np.ndarray:
    """Inlier indices that are distinct in both the source and the reference image."""
    if len(idx) == 0:
        return idx
    idx = idx[distinct_points(src[idx], ratios[idx], sep)]
    return idx[distinct_points(dst[idx], ratios[idx], sep)]


def _find_seed(src, dst, ratios, cfg: GeometryConfig, log: list[str]) -> tuple[np.ndarray, int, int] | None:
    n_all = len(src)
    order = np.argsort(ratios, kind="stable")
    pools = sorted({min(p, n_all) for p in cfg.seed_pool_sizes} | {n_all})
    for pool in pools:
        idx = order[:pool]
        est = estimate_transform(src[idx], dst[idx], "affine", cfg, method="ransac")
        if est is None:
            log.append(f"seed pool {pool}: no model")
            continue
        inl = idx[est.inlier_mask]
        # A seed must be supported by points that are genuinely apart; several
        # keypoints on one crater agreeing with each other prove nothing.
        spread = _tie_points(inl, src, dst, ratios, cfg.seed_min_separation_px)
        log.append(f"seed pool {pool}: {len(inl)} inliers, {len(spread)} spatially distinct")
        if len(spread) >= cfg.min_seed_inliers:
            return est.matrix, pool, len(spread)
    return None


def _refit(src, dst, matrix, radius: float, model: str, cfg: GeometryConfig) -> TransformEstimate | None:
    near = np.flatnonzero(reprojection_errors(matrix, src, dst) < radius)
    est = estimate_transform(src[near], dst[near], model, cfg)
    if est is None:
        return None
    mask = np.zeros(len(src), bool)
    mask[near[est.inlier_mask]] = True
    return TransformEstimate(est.matrix, est.model, est.method, mask)


def find_consensus(
    src: np.ndarray,
    dst: np.ndarray,
    ratios: np.ndarray,
    cfg: GeometryConfig,
    validation: ValidationConfig,
    source_size: tuple[int, int],
    reference_size: tuple[int, int],
) -> tuple[Consensus | None, list[str]]:
    """Verified transform for the matches ``src[i] -> dst[i]`` or None.

    ``ratios`` ranks the matches (lower = more distinctive). Sizes are (w, h).
    """
    log: list[str] = []
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if len(src) < 3:
        log.append("fewer than 3 matches")
        return None, log

    seed = _find_seed(src, dst, ratios, cfg, log)
    if seed is None:
        return None, log
    matrix, seed_pool, seed_inliers = seed

    radii = [max(r, cfg.reproj_threshold_px) for r in cfg.expansion_radii_px] or [3 * cfg.reproj_threshold_px]
    best: TransformEstimate | None = None
    for radius in [*radii, radii[-1]]:  # last radius twice so the fit settles
        est = _refit(src, dst, matrix, radius, "affine", cfg)
        if est is None or est.inlier_count < 3:
            break
        best, matrix = est, est.matrix
        log.append(f"expansion within {radius:g} px: {est.inlier_count} affine inliers")
    if best is None:
        log.append("expansion produced no model")
        return None, log

    sep = cfg.tie_point_min_separation_px
    chosen = best
    chosen_idx = _tie_points(np.flatnonzero(best.inlier_mask), src, dst, ratios, sep)

    want = cfg.model
    if want in ("homography", "similarity", "auto"):
        alt_model = "homography" if want == "auto" else want
        alt = _refit(src, dst, best.matrix, radii[-1], alt_model, cfg)
        if alt is not None:
            alt_idx = _tie_points(np.flatnonzero(alt.inlier_mask), src, dst, ratios, sep)
            log.append(f"{alt_model}: {len(alt_idx)} tie points (affine: {len(chosen_idx)})")
            if want != "auto":
                chosen, chosen_idx = alt, alt_idx
            else:
                # Eight degrees of freedom need clear evidence: enough support,
                # a real gain over the affine model, and a sane geometry.
                ok = check_transform(alt.matrix, source_size, reference_size, validation).valid
                if (
                    ok
                    and len(alt_idx) >= cfg.homography_min_inliers
                    and len(alt_idx) >= cfg.homography_min_gain * len(chosen_idx)
                ):
                    chosen, chosen_idx = alt, alt_idx
                    log.append("auto: homography selected")
                else:
                    log.append("auto: affine kept")
        elif want != "auto":
            log.append(f"{alt_model}: estimation failed")
            return None, log

    return (
        Consensus(
            matrix=chosen.matrix,
            model=chosen.model,
            method=chosen.method,
            inlier_idx=chosen_idx,
            seed_pool=seed_pool,
            seed_inliers=seed_inliers,
            log=log,
        ),
        log,
    )
