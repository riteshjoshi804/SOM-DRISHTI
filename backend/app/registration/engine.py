"""Registration engine: matcher branches -> one verification -> one measurement and report.

A *branch* turns a source / reference pair into correspondences in original pixels plus a
verified geometry, and everything after that is the same for every matcher.

The SIFT branch is the classical MVP: it tries each configured preprocessing representation
and, only when a representation yields no geometry at all, the same representation with the
source mirrored left-right (SIFT is not mirror-invariant and one official scene is delivered
mirrored). The strongest *valid* geometry wins.

The learned branches run the guided search (``matching/guided.py``). AUTO runs every matcher
that is available and compares them by the transparent rules in ``matching/selection.py``.

If nothing valid is found the result is FAILED with the reasons - never a guess.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from app.config import PipelineConfig
from app.evaluation.metrics import (
    ReprojectionStats,
    SpatialCoverage,
    cross_validated_rmse,
    expected_chance_inliers,
    metadata_consistency,
    reprojection_stats,
    spatial_coverage,
)
from app.evaluation.validation import HoldoutResult, holdout_validation, paired_refinement_holdout
from app.geometry.consensus import Consensus, find_consensus
from app.geometry.estimation import apply_transform, fit_least_squares
from app.geometry.strip_model import check_strip, fit_strip
from app.geometry.tiepoints import TiePointSummary, select_tie_points
from app.geometry.validation import TransformCheck, check_transform
from app.geometry.verification import VerifiedGeometry
from app.ingestion.indexer import SceneRecord
from app.ingestion.raster import RasterData, compute_stats, load_raster
from app.matching.base import LOFTR, RIFT, SIFT, SUPERPOINT_LIGHTGLUE, MatcherResult, failed_result
from app.matching.guided import GuidedResult, ImageView, guided_search, run_decoy_controls
from app.matching.matcher import MatchSet, match_features
from app.matching.models import ModelUnavailable, device_label
from app.matching.registry import create_pair_matcher, matcher_statuses
from app.matching.selection import Candidate, SelectionSummary, model_disagreement, select
from app.matching.sift import Features, extract_sift
from app.metadata.georef import PredictedFootprint, predict_footprint, reference_gsd_m
from app.metadata.overlap import OverlapEstimate, estimate_overlap
from app.metadata.xml_parser import SceneMetadata, parse_scene_xml
from app.preprocessing.illumination import IlluminationSummary, analyse_illumination, axis_difference_in_reference_frame
from app.preprocessing.pipeline import Preprocessed, preprocess
from app.refinement.subpixel import RefinedPoints, RefinementSummary, refine_correspondences
from app.registration import exports
from app.registration.schemas import (
    AttemptSummary,
    DecoySummary,
    EvidenceSummary,
    ImageSummary,
    MatcherSummary,
    Metrics,
    RegistrationResult,
    Status,
    Transformation,
)
from app.registration.warp import (
    source_overlap_mask,
    source_overlap_mask_smooth,
    warp_mask_to_reference,
    warp_to_reference,
    warp_with_inverse,
    write_registered_geotiff,
)
from app.visualization import render

ProgressFn = Callable[[str, float], None]
_RANK = {"SUCCESS": 2, "LOW_CONFIDENCE": 1, "FAILED": 0}

RMSE_NOTE = (
    "Reprojection RMSE is computed on the same matches that were used to estimate the "
    "transformation. It measures internal consistency, not independent ground-truth accuracy."
)


@dataclass
class _Attempt:
    mode: str
    mirrored: bool
    source_pre: Preprocessed
    reference_pre: Preprocessed
    source_features: Features  # points in ORIGINAL (un-mirrored) source pixels
    reference_features: Features
    matches: MatchSet
    consensus: Consensus | None  # matrix maps ORIGINAL source pixels -> reference pixels
    check: TransformCheck | None
    reprojection: ReprojectionStats | None
    coverage: SpatialCoverage | None
    overlap_fraction: float | None
    status: Status
    failures: list[str]
    warnings: list[str]
    summary: AttemptSummary

    @property
    def accepted(self) -> bool:
        return self.status != "FAILED"

    def tie_points(self) -> tuple[np.ndarray, np.ndarray]:
        """All ratio-filtered candidate matches as (source px, reference px)."""
        return (
            self.source_features.points[self.matches.source_idx].astype(np.float64),
            self.reference_features.points[self.matches.reference_idx].astype(np.float64),
        )


@dataclass
class BranchOutcome:
    """Everything one matcher produced, in a form that does not depend on the matcher."""

    matcher: str
    result: MatcherResult  # candidates in original pixels
    geometry: VerifiedGeometry | None
    status: Status
    failures: list[str]
    warnings: list[str]
    representation: str | None
    source_pre: Preprocessed | None
    reference_pre: Preprocessed | None
    attempts: list[AttemptSummary]
    seconds: float
    availability: str = "OK"
    keypoints_source: np.ndarray | None = None
    keypoints_reference: np.ndarray | None = None
    reprojection: ReprojectionStats | None = None
    coverage: SpatialCoverage | None = None
    overlap_mask: np.ndarray | None = None
    overlap_fraction: float | None = None
    along_track_support: float | None = None
    search_area_px: float = 0.0
    decoys: list[DecoySummary] = field(default_factory=list)
    holdout: HoldoutResult | None = None
    tie_summary: TiePointSummary | None = None
    selected_idx: np.ndarray | None = None  # into the inlier list
    refined: RefinedPoints | None = None
    refinement_summary: RefinementSummary | None = None
    refinement_validation: dict | None = None
    initial_geometry: VerifiedGeometry | None = None
    trace: dict | None = None
    guided: GuidedResult | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.status != "FAILED"

    @property
    def inliers(self) -> int:
        return len(self.geometry.inlier_idx) if self.geometry is not None else 0

    @property
    def inlier_ratio(self) -> float | None:
        return (self.inliers / len(self.result)) if len(self.result) else None


def _mirror_matrix(width: int) -> np.ndarray:
    """Maps pixel x of the left-right flipped image back to the original image."""
    return np.array([[-1.0, 0.0, width - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def _source_corners(width: int, height: int) -> np.ndarray:
    return np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)


def _image_summary(raster: RasterData) -> ImageSummary:
    info = raster.info
    return ImageSummary(
        path=info.path,
        width=info.width,
        height=info.height,
        dtype=info.dtype,
        nodata=info.nodata,
        has_georeferencing=info.has_geotransform,
        stats=compute_stats(raster.data, raster.valid_mask),
    )


def grade(
    cfg: PipelineConfig,
    n_src: int | None,
    n_ref: int | None,
    n_matches: int,
    inliers: int | None,
    check: TransformCheck | None,
    coverage: float | None,
    rmse: float | None,
    *,
    learned: bool = False,
    invalid_reasons: list[str] | None = None,
) -> tuple[Status, list[str], list[str]]:
    """(status, failure reasons, low-confidence warnings) from measured quantities.

    ``inliers`` is None when no geometric consensus was found at all. Keypoint counts are
    None for detector-free matchers. Learned matchers face stricter hard limits, because they
    return matches for any input.
    """
    v = cfg.validation
    fail: list[str] = []
    warn: list[str] = []
    if n_src is not None and n_src < v.min_source_keypoints:
        fail.append(f"insufficient source keypoints ({n_src} < {v.min_source_keypoints})")
    if n_ref is not None and n_ref < v.min_reference_keypoints:
        fail.append(f"insufficient reference keypoints ({n_ref} < {v.min_reference_keypoints})")
    if not fail and n_matches < v.min_matches:
        fail.append(f"insufficient matches ({n_matches} {'accepted' if learned else 'ratio-filtered'} < {v.min_matches})")
    if fail:
        return "FAILED", fail, warn
    if inliers is None:
        return "FAILED", ["too few inliers: no transformation is supported by enough spatially distinct matches"], warn

    min_inliers = v.learned_min_inliers if learned else v.min_inliers
    min_ratio = v.learned_min_inlier_ratio if learned else v.min_inlier_ratio
    ratio = inliers / n_matches if n_matches else 0.0
    if inliers < min_inliers:
        fail.append(f"too few inliers ({inliers} < {min_inliers})")
    if ratio < min_ratio:
        fail.append(f"low inlier ratio ({ratio:.2%} < {min_ratio:.2%})")
    if check is not None and not check.valid:
        fail.append("unstable transformation: " + "; ".join(check.reasons))
    if invalid_reasons:
        fail.append("unstable transformation: " + "; ".join(invalid_reasons))
    if coverage is not None and coverage < v.min_spatial_coverage:
        fail.append(f"low spatial coverage ({coverage:.1%} < {v.min_spatial_coverage:.1%})")
    if fail:
        return "FAILED", fail, warn

    if inliers < v.good_inliers:
        warn.append(f"few inliers ({inliers} < {v.good_inliers})")
    if ratio < v.good_inlier_ratio:
        warn.append(f"low inlier ratio ({ratio:.2%} < {v.good_inlier_ratio:.2%})")
    if coverage is not None and coverage < v.good_spatial_coverage:
        warn.append(f"tie points cover little of the overlap ({coverage:.1%} < {v.good_spatial_coverage:.1%})")
    if rmse is not None and rmse > v.good_rmse_px:
        warn.append(f"high reprojection RMSE ({rmse:.2f} px > {v.good_rmse_px} px)")
    return ("LOW_CONFIDENCE" if warn else "SUCCESS"), fail, warn


def _extract_and_match(
    mirrored: bool, src_pre: Preprocessed, ref_features: Features, cfg: PipelineConfig
) -> tuple[Features, MatchSet]:
    """SIFT on the (optionally mirrored) source, matched to the reference.

    Keypoint coordinates are returned in ORIGINAL source pixels either way.
    """
    image, mask = src_pre.image, src_pre.detect_mask
    if mirrored:
        image, mask = np.ascontiguousarray(image[:, ::-1]), np.ascontiguousarray(mask[:, ::-1])
    features = extract_sift(image, mask, cfg.sift)
    matches = match_features(features, ref_features, cfg.match)
    if mirrored:
        features.points = features.points.copy()
        features.points[:, 0] = (image.shape[1] - 1) - features.points[:, 0]
    return features, matches


def _verify(
    mode: str,
    mirrored: bool,
    source: RasterData,
    reference: RasterData,
    src_pre: Preprocessed,
    ref_pre: Preprocessed,
    src_features: Features,
    ref_features: Features,
    matches: MatchSet,
    cfg: PipelineConfig,
    t0: float,
) -> _Attempt:
    """Geometric verification, measurement and grading of one set of SIFT matches."""
    h, w = src_pre.image.shape
    ref_size = (ref_pre.image.shape[1], ref_pre.image.shape[0])

    consensus, log, check = None, [], None
    if len(matches) >= cfg.validation.min_matches:
        # Geometry is estimated in the frame SIFT saw (mirrored or not), where a
        # valid transform always preserves orientation; it is converted to
        # original source pixels right after.
        frame_pts = src_features.points[matches.source_idx].astype(np.float64)
        if mirrored:
            frame_pts[:, 0] = (w - 1) - frame_pts[:, 0]
        consensus, log = find_consensus(
            frame_pts,
            ref_features.points[matches.reference_idx],
            matches.ratios,
            cfg.geometry,
            cfg.validation,
            (w, h),
            ref_size,
        )
        if consensus is not None:
            check = check_transform(consensus.matrix, (w, h), ref_size, cfg.validation)
            if mirrored:
                consensus.matrix = consensus.matrix @ _mirror_matrix(w)

    reprojection = coverage = overlap_fraction = None
    if consensus is not None:
        idx = consensus.inlier_idx
        in_src = src_features.points[matches.source_idx[idx]].astype(np.float64)
        in_ref = ref_features.points[matches.reference_idx[idx]].astype(np.float64)
        reprojection = reprojection_stats(consensus.matrix, in_src, in_ref)
        overlap = None
        if check is not None and check.valid:
            overlap = source_overlap_mask(source.valid_mask, reference.valid_mask, consensus.matrix)
            valid_px = int(source.valid_mask.sum())
            overlap_fraction = float(overlap.sum() / valid_px) if valid_px else None
        coverage = spatial_coverage(in_src, source.valid_mask, cfg.coverage, overlap)

    graded_coverage = None
    if coverage is not None:
        # With partial overlap, cells outside the overlap can never hold a tie point.
        graded_coverage = (
            coverage.coverage_within_overlap if coverage.coverage_within_overlap is not None else coverage.coverage
        )
    inliers = len(consensus.inlier_idx) if consensus is not None else None
    status, fail, warn = grade(
        cfg,
        len(src_features),
        len(ref_features),
        len(matches),
        inliers,
        check,
        graded_coverage,
        reprojection.rmse_px if reprojection else None,
    )
    summary = AttemptSummary(
        preprocessing_mode=mode,
        mirrored_source=mirrored,
        source_keypoints=len(src_features),
        reference_keypoints=len(ref_features),
        raw_matches=matches.raw_candidates,
        ratio_filtered_matches=len(matches),
        inliers=inliers or 0,
        inlier_ratio=((inliers or 0) / len(matches)) if len(matches) else None,
        transformation_type=consensus.model if consensus is not None else None,
        status=status,
        reasons=fail or warn,
        log=log,
        seconds=round(time.perf_counter() - t0, 3),
    )
    return _Attempt(
        mode, mirrored, src_pre, ref_pre, src_features, ref_features, matches, consensus, check,
        reprojection, coverage, overlap_fraction, status, fail, warn, summary,
    )  # fmt: skip


def _pooled(group: list[_Attempt], source: RasterData, reference: RasterData, cfg: PipelineConfig) -> _Attempt:
    """One verification over the candidate matches of several representations.

    Different representations recover different correct matches. All of them
    must agree with a single transformation, so pooling adds support without
    lowering the bar: every tie point is still a global nearest-neighbour match
    that passed the ratio test in its own representation.
    """
    t0 = time.perf_counter()
    src_off = np.cumsum([0] + [len(a.source_features) for a in group])
    ref_off = np.cumsum([0] + [len(a.reference_features) for a in group])

    def concat(features: list[Features]) -> Features:
        return Features(
            points=np.concatenate([f.points for f in features]),
            sizes=np.concatenate([f.sizes for f in features]),
            angles=np.concatenate([f.angles for f in features]),
            responses=np.concatenate([f.responses for f in features]),
            descriptors=np.zeros((0, 128), np.float32),  # not needed after matching
        )

    matches = MatchSet(
        source_idx=np.concatenate([a.matches.source_idx + src_off[i] for i, a in enumerate(group)]),
        reference_idx=np.concatenate([a.matches.reference_idx + ref_off[i] for i, a in enumerate(group)]),
        distances=np.concatenate([a.matches.distances for a in group]),
        ratios=np.concatenate([a.matches.ratios for a in group]),
        raw_candidates=sum(a.matches.raw_candidates for a in group),
        ratio_filtered=sum(a.matches.ratio_filtered for a in group),
        matcher_used=group[0].matches.matcher_used,
    )
    first = group[0]
    return _verify(
        "pooled(" + "+".join(a.mode for a in group) + ")",
        first.mirrored,
        source,
        reference,
        first.source_pre,
        first.reference_pre,
        concat([a.source_features for a in group]),
        concat([a.reference_features for a in group]),
        matches,
        cfg,
        t0,
    )


def _better(a: _Attempt, b: _Attempt | None) -> bool:
    if b is None:
        return True

    def key(t: _Attempt) -> tuple[int, int, int]:
        return (_RANK[t.status], t.summary.inliers, t.summary.ratio_filtered_matches)

    return key(a) > key(b)


# ---------------------------------------------------------------------------------------- branches


def _sift_branch(
    source: RasterData, reference: RasterData, cfg: PipelineConfig, say: ProgressFn, overlap: OverlapEstimate | None = None
) -> BranchOutcome:
    """The classical pipeline, unchanged; only its result is wrapped in the common form."""
    t_start = time.perf_counter()
    cv2.setRNGSeed(cfg.random_seed)  # makes OpenCV's RANSAC sampling repeatable
    modes = list(dict.fromkeys(cfg.preprocessing_modes))
    attempts: list[_Attempt] = []
    best: _Attempt | None = None
    for i, mode in enumerate(modes):
        lo, step = 0.05 + 0.75 * i / len(modes), 0.75 / len(modes)
        say(f"[{mode}] preprocessing", lo)
        src_pre = preprocess(source.data, source.valid_mask, mode, cfg.preprocess)
        ref_pre = preprocess(reference.data, reference.valid_mask, mode, cfg.preprocess)
        say(f"[{mode}] SIFT on reference", lo + 0.2 * step)
        ref_mask = ref_pre.detect_mask
        if cfg.sift.use_search_region and overlap is not None and overlap.mode == "metadata" and overlap.search_bbox_px:
            x0, y0, x1, y1 = overlap.search_bbox_px
            ref_mask = np.zeros_like(ref_pre.detect_mask)
            ref_mask[y0:y1, x0:x1] = ref_pre.detect_mask[y0:y1, x0:x1]
        ref_features = extract_sift(ref_pre.image, ref_mask, cfg.sift)

        for mirrored in (False, True):
            if mirrored:
                # Only worth trying when the source as delivered gave no acceptable geometry
                # (a rejected, orientation-flipping consensus is itself a hint of mirroring).
                if attempts[-1].accepted or not cfg.geometry.allow_mirrored_source:
                    break
                say(f"[{mode}] retry with mirrored source", lo + 0.75 * step)
            else:
                say(f"[{mode}] matching and geometric verification", lo + 0.5 * step)
            t0 = time.perf_counter()
            src_features, matches = _extract_and_match(mirrored, src_pre, ref_features, cfg)
            attempt = _verify(
                mode, mirrored, source, reference, src_pre, ref_pre, src_features, ref_features, matches, cfg, t0
            )
            attempts.append(attempt)
            if _better(attempt, best):
                best = attempt
        if best is not None and best.status == "SUCCESS" and not cfg.evaluate_all_modes:
            break

    assert best is not None
    if cfg.pool_modes and best.status != "SUCCESS":
        say("pooling candidates from all representations", 0.80)
        for mirrored in (False, True):
            group = [a for a in attempts if a.mirrored == mirrored]
            if len(group) < 2:
                continue
            pooled = _pooled(group, source, reference, cfg)
            attempts.append(pooled)
            if _better(pooled, best):
                best = pooled
            if pooled.accepted:
                break

    src_pts, ref_pts = best.tie_points()
    ratios = best.matches.ratios.astype(np.float64)
    result = MatcherResult(
        matcher_name=SIFT,
        source_points=src_pts,
        reference_points=ref_pts,
        confidence_scores=1.0 - ratios,
        rank=ratios,
        candidate_count=best.matches.raw_candidates,
        accepted_count=len(best.matches),
        runtime_seconds=time.perf_counter() - t_start,
        device="cpu",
        source_keypoints=len(best.source_features),
        reference_keypoints=len(best.reference_features),
        extras={"matcher_used": best.matches.matcher_used},
    )
    geometry = None
    if best.consensus is not None and best.check is not None:
        consensus = best.consensus
        geometry = VerifiedGeometry(
            kind=consensus.model,
            matrix=consensus.matrix,
            method=consensus.method,
            inlier_idx=consensus.inlier_idx,
            check=best.check,
            mirrored=best.mirrored,
            seed_pool=consensus.seed_pool,
            seed_inliers=consensus.seed_inliers,
            log=consensus.log,
            global_inliers=len(consensus.inlier_idx),
        )
    outcome = BranchOutcome(
        matcher=SIFT,
        result=result,
        geometry=geometry,
        status=best.status,
        failures=list(best.failures),
        warnings=list(best.warnings),
        representation=best.mode,
        source_pre=best.source_pre,
        reference_pre=best.reference_pre,
        attempts=[a.summary for a in attempts],
        seconds=time.perf_counter() - t_start,
        keypoints_source=best.source_features.points,
        keypoints_reference=best.reference_features.points,
        reprojection=best.reprojection,
        coverage=best.coverage,
        overlap_fraction=best.overlap_fraction,
        search_area_px=float(reference.valid_mask.sum()),
    )
    return outcome


def _unavailable_branch(name: str, reason: str, source: RasterData, reference: RasterData) -> BranchOutcome:
    return BranchOutcome(
        matcher=name,
        result=failed_result(name, reason, status="UNAVAILABLE"),
        geometry=None,
        status="FAILED",
        failures=[f"{name} is unavailable: {reason}"],
        warnings=[],
        representation=None,
        source_pre=None,
        reference_pre=None,
        attempts=[],
        seconds=0.0,
        availability="UNAVAILABLE",
    )


def _measure_geometry(o: BranchOutcome, source: RasterData, reference: RasterData, cfg: PipelineConfig) -> None:
    """Residuals, overlap and coverage of a branch's geometry, for any model family."""
    g = o.geometry
    if g is None:
        return
    idx = g.inlier_idx
    in_src, in_ref = o.result.source_points[idx], o.result.reference_points[idx]
    errors = g.errors(in_src, in_ref)
    o.reprojection = ReprojectionStats(
        rmse_px=float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
        median_px=float(np.median(errors)) if len(errors) else None,
        max_px=float(errors.max()) if len(errors) else None,
        count=int(len(errors)),
    )
    overlap = None
    if g.valid:
        if g.strip is not None:
            overlap = source_overlap_mask_smooth(source.valid_mask, reference.valid_mask, g.strip.apply)
        else:
            overlap = source_overlap_mask(source.valid_mask, reference.valid_mask, g.matrix)
        valid_px = int(source.valid_mask.sum())
        o.overlap_fraction = float(overlap.sum() / valid_px) if valid_px else None
        o.overlap_mask = overlap
    o.coverage = spatial_coverage(in_src, source.valid_mask, cfg.coverage, overlap)


def _graded_coverage(o: BranchOutcome) -> float | None:
    if o.coverage is None:
        return None
    within = o.coverage.coverage_within_overlap
    return within if within is not None else o.coverage.coverage


def _prior_disagreement(o: BranchOutcome, overlap: OverlapEstimate, cfg: PipelineConfig) -> list[str]:
    """Metadata knows orientation and scale far better than position, so a result far from them is suspect."""
    g = o.geometry
    if g is None or o.guided is None or o.guided.hypothesis is None or not o.guided.hypothesis.from_metadata:
        return []
    relative = g.matrix[:2, :2] @ np.linalg.inv(o.guided.hypothesis.linear)
    rotation = float(np.degrees(np.arctan2(relative[1, 0] - relative[0, 1], relative[0, 0] + relative[1, 1])))
    singular = np.linalg.svd(relative, compute_uv=False)
    reasons = []
    if abs(rotation) > cfg.validation.max_prior_rotation_deg:
        reasons.append(f"rotation differs from the metadata by {rotation:.1f} deg (limit {cfg.validation.max_prior_rotation_deg})")
    deviation = max(abs(singular.max() - 1.0), abs(singular.min() - 1.0))
    if deviation > cfg.validation.max_prior_scale_deviation:
        reasons.append(f"scale differs from the metadata by {deviation:.1%} (limit {cfg.validation.max_prior_scale_deviation:.0%})")
    return reasons


def _guided_branch(
    name: str, source: RasterData, reference: RasterData, cfg: PipelineConfig, overlap: OverlapEstimate, say: ProgressFn
) -> BranchOutcome:
    """A learned (or experimental) matcher, through the guided search and the shared verification."""
    t_start = time.perf_counter()
    try:
        say(f"{name}: loading model", 0.01)
        matcher = create_pair_matcher(name, cfg)
    except ModelUnavailable as exc:
        return _unavailable_branch(name, str(exc), source, reference)
    cv2.setRNGSeed(cfg.random_seed)
    representation = cfg.guided.representation
    say(f"{name}: preparing the {representation} representation", 0.02)
    src_pre = preprocess(source.data, source.valid_mask, representation, cfg.preprocess)
    ref_pre = preprocess(reference.data, reference.valid_mask, representation, cfg.preprocess)
    src_view = ImageView(src_pre.image, src_pre.detect_mask)
    ref_view = ImageView(ref_pre.image, ref_pre.detect_mask)

    guided = guided_search(
        matcher, src_view, ref_view, overlap, cfg.guided, cfg.geometry, cfg.validation,
        seed=cfg.random_seed, say=lambda message, fraction: say(f"{name}: {message}", 0.03 + 0.77 * fraction),
    )  # fmt: skip
    result = guided.result
    geometry = guided.geometry
    side = cfg.guided.fine_tile_px / max(guided.scale, 1e-6) + 2.0 * cfg.guided.fine_margin_px
    tiles = guided.trace.get("tile_count") or 1
    o = BranchOutcome(
        matcher=name,
        result=result,
        geometry=geometry,
        status="FAILED",
        failures=[],
        warnings=[],
        representation=representation,
        source_pre=src_pre,
        reference_pre=ref_pre,
        attempts=[],
        seconds=0.0,
        search_area_px=float(min(reference.valid_mask.sum(), tiles * side * side)),
        trace=guided.trace,
        guided=guided,
    )
    _measure_geometry(o, source, reference, cfg)

    if geometry is None:
        o.failures = [result.failure_reason or "no transformation is supported by enough spatially distinct matches"]
    else:
        o.status, o.failures, o.warnings = grade(
            cfg,
            result.source_keypoints,
            result.reference_keypoints,
            len(result),
            len(geometry.inlier_idx),
            None,
            _graded_coverage(o),
            o.reprojection.rmse_px if o.reprojection else None,
            learned=True,
            invalid_reasons=geometry.reasons if not geometry.valid else None,
        )
        disagreement = _prior_disagreement(o, overlap, cfg)
        if disagreement and o.status != "FAILED":
            o.status = "FAILED"
            o.failures = ["implausible against the metadata: " + "; ".join(disagreement)]

    # The null control is only worth its cost for a result that would otherwise stand.
    if o.status != "FAILED" and cfg.guided.decoy_controls > 0:
        say(f"{name}: decoy controls on known-wrong ground", 0.82)
        decoys = run_decoy_controls(
            matcher, src_view, ref_view, guided, cfg.guided, cfg.geometry, cfg.validation,
            seed=cfg.random_seed, say=lambda message, fraction: say(f"{name}: {message}", 0.82 + 0.12 * fraction),
        )  # fmt: skip
        o.decoys = [DecoySummary(**vars(d)) for d in decoys]
        _apply_decoy_rule(o, cfg)
    elif o.status != "FAILED":
        o.warnings.append("no decoy control was run, so the null level of this result is unknown")
        if o.status == "SUCCESS":
            o.status = "LOW_CONFIDENCE"

    o.seconds = time.perf_counter() - t_start
    result.runtime_seconds = o.seconds
    h = guided.hypothesis
    o.attempts = [
        AttemptSummary(
            preprocessing_mode=representation,
            mirrored_source=bool(h.mirrored) if h is not None else False,
            source_keypoints=result.source_keypoints,
            reference_keypoints=result.reference_keypoints,
            raw_matches=result.candidate_count,
            ratio_filtered_matches=len(result),
            inliers=o.inliers,
            inlier_ratio=o.inlier_ratio,
            transformation_type=geometry.kind if geometry is not None else None,
            status=o.status,
            reasons=o.failures or o.warnings,
            log=(geometry.log if geometry is not None else []) or guided.trace.get("anchor_log", [])[-8:],
            seconds=round(o.seconds, 3),
            matcher=name,
        )
    ]
    return o


def _apply_decoy_rule(o: BranchOutcome, cfg: PipelineConfig) -> None:
    """A result must stand clearly above what the same pipeline finds on known-wrong ground."""
    measured = [d for d in o.decoys if d.tiles_used > 0]
    if not measured:
        o.warnings.append("no wrong ground was available for a decoy control, so the null level of this result is unknown")
        if o.status == "SUCCESS":
            o.status = "LOW_CONFIDENCE"
        return
    worst = max(d.inliers for d in measured)
    limit = cfg.guided.decoy_factor * worst
    if o.inliers < limit:
        o.status = "FAILED"
        o.failures = [f"not above the null control: {o.inliers} inliers vs {worst} on known-wrong ground (factor {cfg.guided.decoy_factor:g} required)"]
        o.warnings = []


def _along_track_support(o: BranchOutcome, source: RasterData, cfg: PipelineConfig) -> float | None:
    """Share of the strip length that has verified tie points, for any model family.

    A stretch counts only with enough points to constrain it, so that a stray point cannot decide a status.
    """
    if o.geometry is None or not len(o.geometry.inlier_idx):
        return None
    axis = o.geometry.strip.axis if o.geometry.strip is not None else int(source.info.height >= source.info.width)
    along = source.info.height if axis == 1 else source.info.width
    pts = o.result.source_points[o.geometry.inlier_idx][:, axis]
    bins = 20
    counts = np.bincount(np.clip((pts / along * bins).astype(int), 0, bins - 1), minlength=bins)
    return float((counts >= cfg.validation.along_track_support_min_points).mean())


def _apply_support_rule(o: BranchOutcome, source: RasterData, cfg: PipelineConfig) -> None:
    """A mapping extrapolated over most of the strip is not a clean success, whichever matcher or model made it."""
    support = _along_track_support(o, source, cfg) if o.accepted else None
    o.along_track_support = support
    if support is None or support >= cfg.validation.min_along_track_support:
        return
    o.warnings.append(f"tie points support only {support:.0%} of the strip length; the rest of the mapping is extrapolated")
    if o.status == "SUCCESS":
        o.status = "LOW_CONFIDENCE"


def _sift_decoy(o: BranchOutcome, source: RasterData, reference: RasterData, cfg: PipelineConfig) -> None:
    """SIFT on a reference with the true ground removed: the same null question as for the learned matchers."""
    g = o.geometry
    if g is None or o.reference_pre is None or o.source_pre is None:
        return
    t0 = time.perf_counter()
    h, w = reference.data.shape
    corners = _source_corners(source.info.width, source.info.height)
    quad = np.round(g.apply(corners)).astype(np.int32)
    keep = np.full((h, w), 255, np.uint8)
    cv2.fillConvexPoly(keep, quad, 0)
    # Generous margin, so that no window of true ground can survive.
    keep = cv2.erode(keep, cv2.getStructuringElement(cv2.MORPH_RECT, (257, 257)))
    mask = cv2.bitwise_and(o.reference_pre.detect_mask, keep)
    if int((mask > 0).sum()) < 10000:
        o.decoys = [DecoySummary(kind="masked_reference", model="no wrong ground available in the reference")]
        return
    ref_features = extract_sift(o.reference_pre.image, mask, cfg.sift)
    features, matches = _extract_and_match(g.mirrored, o.source_pre, ref_features, cfg)
    attempt = _verify(
        o.representation or "clahe", g.mirrored, source, reference, o.source_pre, o.reference_pre,
        features, ref_features, matches, cfg, t0,
    )  # fmt: skip
    consensus = attempt.consensus
    o.decoys = [
        DecoySummary(
            kind="masked_reference",
            tiles_used=1,
            candidates=len(matches),
            candidates_before_appearance=len(matches),
            inliers=len(consensus.inlier_idx) if consensus is not None else 0,
            valid=bool(attempt.check.valid) if attempt.check is not None else False,
            model=consensus.model if consensus is not None else None,
            seconds=round(time.perf_counter() - t0, 1),
        )
    ]


def _run_branch(
    name: str, source: RasterData, reference: RasterData, cfg: PipelineConfig, overlap: OverlapEstimate, say: ProgressFn
) -> BranchOutcome:
    if name == SIFT:
        outcome = _sift_branch(source, reference, cfg, say, overlap)
        if cfg.sift_decoy_controls > 0 and outcome.accepted:
            say("SIFT: decoy control on known-wrong ground", 0.85)
            _sift_decoy(outcome, source, reference, cfg)
            _apply_decoy_rule(outcome, cfg)
        return outcome
    return _guided_branch(name, source, reference, cfg, overlap, say)


# ---------------------------------------------------------------------------------- measurements


def _local_linear(g: VerifiedGeometry) -> Callable[[np.ndarray], np.ndarray]:
    """(N, 2, 2) local linear maps of the geometry at source points."""
    if g.strip is not None:
        return g.strip.jacobian
    m = g.matrix / g.matrix[2, 2]
    if np.allclose(m[2], [0.0, 0.0, 1.0], atol=1e-12):
        return lambda points: np.repeat(m[:2, :2][None], len(points), axis=0)

    def jacobian(points: np.ndarray, step: float = 1.0) -> np.ndarray:
        dx = (apply_transform(m, points + [step, 0]) - apply_transform(m, points - [step, 0])) / (2 * step)
        dy = (apply_transform(m, points + [0, step]) - apply_transform(m, points - [0, step])) / (2 * step)
        return np.stack([dx, dy], axis=2)

    return jacobian


def _measure_holdout(o: BranchOutcome, source: RasterData, reference: RasterData, cfg: PipelineConfig) -> None:
    if not cfg.holdout.enabled or o.geometry is None or not o.accepted:
        return
    o.holdout = holdout_validation(
        o.result.source_points,
        o.result.reference_points,
        o.result.rank,
        o.geometry,
        (source.info.width, source.info.height),
        (reference.info.width, reference.info.height),
        cfg.geometry,
        cfg.validation,
        cfg.guided,
        cfg.holdout,
        o.search_area_px,
        seed=cfg.random_seed,
    )


def _apply_unseen_ground_rule(o: BranchOutcome, cfg: PipelineConfig) -> None:
    """A model that predicts ground left out of its fit poorly is not a clean success, whatever else holds."""
    h = o.holdout
    limit = cfg.validation.good_unseen_block_rmse_px
    if h is None or h.block_rmse_px is None or h.block_rmse_px <= limit or not o.accepted:
        return
    o.warnings.append(
        f"predicts along-track blocks left out of the fit only to {h.block_rmse_px:.1f} px RMSE (target {limit:g} px), "
        "so alignment away from the tie points is uncertain"
    )
    if o.status == "SUCCESS":
        o.status = "LOW_CONFIDENCE"


def _post_process(o: BranchOutcome, source: RasterData, reference: RasterData, cfg: PipelineConfig, say: ProgressFn) -> None:
    """Tie-point selection and sub-pixel refinement of an accepted branch."""
    g = o.geometry
    if g is None or not o.accepted:
        return
    idx = g.inlier_idx
    src, ref = o.result.source_points[idx], o.result.reference_points[idx]
    conf = o.result.confidence_scores[idx]
    errors = g.errors(src, ref)
    gsd = reference_gsd_m(reference.info)
    source_gsd = None
    if gsd:
        source_gsd = gsd * float(np.sqrt(abs(np.linalg.det(g.matrix[:2, :2]))))
    o.selected_idx, o.tie_summary = select_tie_points(
        src, conf, errors, cfg.geometry.reproj_threshold_px, source.valid_mask, cfg.tie_points, o.overlap_mask, source_gsd
    )

    r = cfg.refinement
    if not r.enabled or r.strategy == "none":
        return
    say(f"{o.matcher}: sub-pixel refinement", 0.0)
    if r.representation == o.representation and o.source_pre is not None and o.reference_pre is not None:
        src_pre, ref_pre = o.source_pre, o.reference_pre
    else:
        src_pre = preprocess(source.data, source.valid_mask, r.representation, cfg.preprocess)
        ref_pre = preprocess(reference.data, reference.valid_mask, r.representation, cfg.preprocess)
    # Refinement may use every valid pixel; the detection mask is only eroded for the matcher.
    src_mask = source.valid_mask.astype(np.uint8)
    ref_mask = reference.valid_mask.astype(np.uint8)
    o.refined = refine_correspondences(src_pre.image, src_mask, ref_pre.image, ref_mask, src, ref, _local_linear(g), r, r.representation)
    o.refinement_summary = o.refined.summary

    flags = o.refined.refined
    size = (source.info.width, source.info.height)
    if int(flags.sum()) < 12:
        o.refinement_validation = {"status": "insufficient", "label": "same held-out splits, initial vs refined coordinates", "points": int(flags.sum())}
        return

    def fit(train_src: np.ndarray, train_dst: np.ndarray):
        if g.strip is not None:
            model = fit_strip(train_src, train_dst, size, g.strip.spacing_px, g.strip.smoothness)
            return None if model is None else model.apply
        matrix = fit_least_squares(train_src, train_dst, "affine" if g.kind == "similarity" else g.kind)
        return None if matrix is None else (lambda points: apply_transform(matrix, points))

    o.refinement_validation = paired_refinement_holdout(
        src[flags], ref[flags], o.refined.reference_refined[flags], g.kind, size, fit, cfg.holdout, seed=cfg.random_seed
    )

    if o.refinement_summary.systematic_shift:
        o.notes.append(
            "Refinement moved the points coherently "
            f"({o.refinement_summary.median_shift_x_px:.2f}, {o.refinement_summary.median_shift_y_px:.2f} px), which suggests the correlation "
            "followed shading rather than terrain. The refined coordinates should not be trusted, and the model was not refitted on them."
        )
        return
    # The classical baseline stays exactly as it was. Elsewhere the model is refitted only
    # when the held-out comparison shows that the refined points predict each other better.
    if o.matcher == SIFT or not r.refit_transformation or not o.refinement_validation.get("improved"):
        return
    _refit_on_refined(o, src, source, reference, cfg)


def _refit_on_refined(o: BranchOutcome, src: np.ndarray, source: RasterData, reference: RasterData, cfg: PipelineConfig) -> None:
    g = o.geometry
    refined = o.refined.reference_refined
    size = (source.info.width, source.info.height)
    rsize = (reference.info.width, reference.info.height)
    new = None
    if g.strip is not None:
        strip = fit_strip(src, refined, size, g.strip.spacing_px, g.strip.smoothness)
        if strip is not None:
            check = check_strip(strip, rsize, cfg.validation, -1 if g.mirrored else 1)
            if check.valid:
                new = VerifiedGeometry(
                    kind=g.kind, matrix=strip.best_affine(), method=g.method, inlier_idx=g.inlier_idx, check=g.check,
                    mirrored=g.mirrored, strip=strip, strip_check=check, seed_pool=g.seed_pool, seed_inliers=g.seed_inliers,
                    log=g.log, global_inliers=g.global_inliers,
                )  # fmt: skip
    else:
        matrix = fit_least_squares(src, refined, "affine" if g.kind == "similarity" else g.kind)
        if matrix is not None:
            frame = matrix @ _mirror_matrix(size[0]) if g.mirrored else matrix
            check = check_transform(frame, size, rsize, cfg.validation)
            if check.valid:
                new = VerifiedGeometry(
                    kind=g.kind, matrix=matrix, method=f"{g.method}, refitted on refined points", inlier_idx=g.inlier_idx,
                    check=check, mirrored=g.mirrored, seed_pool=g.seed_pool, seed_inliers=g.seed_inliers, log=g.log,
                    global_inliers=g.global_inliers,
                )  # fmt: skip
    if new is not None:
        o.initial_geometry = g
        o.geometry = new
        o.refinement_summary.model_refit = True


# ------------------------------------------------------------------------------------ the run


def _branches(cfg: PipelineConfig) -> list[str]:
    if cfg.matcher != "AUTO":
        return [cfg.matcher]
    usable = [s.name for s in matcher_statuses(cfg) if (s.available or s.needs_download) and not s.experimental]
    return usable or [SIFT]


def _select(outcomes: list[BranchOutcome], cfg: PipelineConfig) -> tuple[BranchOutcome, SelectionSummary]:
    def values(o: BranchOutcome) -> dict:
        return {
            "coverage_within_overlap": _graded_coverage(o),
            "heldout_median_px": o.holdout.heldout_median_px if o.holdout else None,
            "inlier_ratio": o.inlier_ratio,
            "reprojection_rmse_px": o.reprojection.rmse_px if o.reprojection else None,
            "seconds": o.seconds,
            "inliers_for_information_only": o.inliers,
        }

    summary = select([Candidate(o.matcher, o.status, values(o)) for o in outcomes], cfg.matcher)
    if summary.selected is not None:
        chosen = next(o for o in outcomes if o.matcher == summary.selected)
    else:
        # Nothing was accepted: report the branch that got furthest, for its diagnostics.
        chosen = max(outcomes, key=lambda o: (o.availability != "UNAVAILABLE", o.inliers, len(o.result)))
    return chosen, summary


def _cross_check(chosen: BranchOutcome, outcomes: list[BranchOutcome], summary: SelectionSummary, cfg: PipelineConfig) -> None:
    """Do other matchers' verified tie points fall where the selected model puts them?

    These tie points were found independently of the selected matcher, though not independently
    of the imagery. Measuring at the other matcher's own points avoids blaming a model for
    how another one extrapolates. Large disagreement means at least one of them is wrong.
    """
    if chosen.geometry is None or not chosen.accepted:
        return
    for other in outcomes:
        if other is chosen or other.geometry is None or not other.accepted:
            continue
        idx = other.geometry.inlier_idx
        points = other.result.source_points[idx]
        targets = other.result.reference_points[idx]
        distance = np.linalg.norm(chosen.geometry.apply(points) - targets, axis=1)
        summary.agreement.append(
            {
                "selected": chosen.matcher,
                "other": other.matcher,
                "points": int(len(points)),
                "median_px": float(np.median(distance)),
                "p90_px": float(np.percentile(distance, 90)),
                "within_threshold": int((distance <= cfg.geometry.reproj_threshold_px).sum()),
                "models_differ_at_selected_inliers": model_disagreement(
                    chosen.geometry.apply,
                    other.geometry.apply,
                    chosen.result.source_points[chosen.geometry.inlier_idx],
                ),
            }
        )
        if float(np.median(distance)) > cfg.validation.max_matcher_disagreement_px:
            message = (
                f"{other.matcher} tie points lie a median {np.median(distance):.1f} px from the selected {chosen.matcher} "
                "model: at least one of the two is wrong."
            )
            summary.warnings.append(message)
            chosen.warnings.append(message)
            if chosen.status == "SUCCESS":
                chosen.status = "LOW_CONFIDENCE"


def _summary(o: BranchOutcome) -> MatcherSummary:
    availability = o.availability
    if availability != "UNAVAILABLE" and o.result.status == "FAILED":
        availability = "FAILED"
    return MatcherSummary(
        matcher=o.matcher,
        availability=availability,
        registration_status=o.status,
        reasons=o.failures,
        warnings=o.warnings,
        device=o.result.device,
        representation=o.representation,
        mirrored_source=o.geometry.mirrored if o.geometry is not None else None,
        source_keypoints=o.result.source_keypoints,
        reference_keypoints=o.result.reference_keypoints,
        candidate_count=o.result.candidate_count,
        accepted_count=len(o.result),
        inliers=o.inliers,
        inlier_ratio=o.inlier_ratio,
        reprojection_rmse_px=o.reprojection.rmse_px if o.reprojection else None,
        median_reprojection_error_px=o.reprojection.median_px if o.reprojection else None,
        image_space_spatial_coverage=o.coverage.coverage if o.coverage else None,
        coverage_within_overlap=o.coverage.coverage_within_overlap if o.coverage else None,
        estimated_overlap_fraction=o.overlap_fraction,
        along_track_support=o.along_track_support,
        transformation_type=o.geometry.kind if o.geometry is not None and o.accepted else None,
        holdout=o.holdout,
        tie_points=o.tie_summary,
        refinement=o.refinement_summary,
        refinement_validation=o.refinement_validation,
        decoys=o.decoys,
        appearance=(o.result.extras or {}).get("appearance"),
        processing_time_seconds=round(o.seconds, 3),
    )


def _illumination(
    source: RasterData, reference: RasterData, metadata: SceneMetadata | None, chosen: BranchOutcome, cfg: PipelineConfig
) -> IlluminationSummary | None:
    src_pre, ref_pre = chosen.source_pre, chosen.reference_pre
    if src_pre is None or ref_pre is None:
        return None
    summary = analyse_illumination(
        src_pre.display,
        source.valid_mask,
        ref_pre.display,
        reference.valid_mask,
        sun_azimuth_deg=metadata.sun_azimuth_deg if metadata else None,
        sun_elevation_deg=metadata.sun_elevation_deg if metadata else None,
        solar_incidence_deg=metadata.solar_incidence_angle_deg if metadata else None,
        shadow_threshold=cfg.preprocess.shadow_threshold,
        shadow_min_area_px=cfg.preprocess.shadow_min_area_px,
    )
    if chosen.geometry is not None and chosen.accepted:
        summary.shading_axis_difference_deg = axis_difference_in_reference_frame(summary, chosen.geometry.matrix[:2, :2])
    return summary


def register(
    source: RasterData,
    reference: RasterData,
    cfg: PipelineConfig | None = None,
    *,
    scene: SceneRecord | None = None,
    metadata: SceneMetadata | None = None,
    out_dir: str | Path | None = None,
    progress: ProgressFn | None = None,
    overlap: OverlapEstimate | None = None,
) -> RegistrationResult:
    """Register ``source`` onto ``reference``. Writes artifacts to ``out_dir`` if given.

    ``overlap`` replaces the metadata-derived search prior; synthetic tests use it.
    """
    cfg = cfg or PipelineConfig()
    t_start = time.perf_counter()
    say: ProgressFn = progress or (lambda message, fraction: None)
    notes = [RMSE_NOTE]

    footprint = predict_footprint(source.info, reference.info, metadata) if metadata is not None else None
    if overlap is None:
        overlap = estimate_overlap(source.info, reference.info, metadata, reference.valid_mask, cfg.guided, cfg.validation)

    names = _branches(cfg)
    outcomes: list[BranchOutcome] = []
    for i, name in enumerate(names):
        low, span = 0.75 * i / len(names), 0.75 / len(names)

        def branch_progress(message: str, fraction: float, low: float = low, span: float = span) -> None:
            say(message, low + span * min(max(fraction, 0.0), 1.0))

        outcomes.append(_run_branch(name, source, reference, cfg, overlap, branch_progress))

    say("validating on held-out correspondences", 0.76)
    for o in outcomes:
        _measure_holdout(o, source, reference, cfg)
        # The SIFT-only run stays the unchanged classical baseline. Anywhere results are compared,
        # the same standard applies to every branch.
        if cfg.matcher != SIFT:
            _apply_support_rule(o, source, cfg)
            _apply_unseen_ground_rule(o, cfg)
    chosen, selection = _select(outcomes, cfg)
    if cfg.matcher == "AUTO":
        _cross_check(chosen, outcomes, selection, cfg)

    for o in outcomes:
        _post_process(o, source, reference, cfg, lambda message, fraction: say(message, 0.8))

    say("computing metrics", 0.82)
    consensus_geometry = chosen.geometry
    src_pts, ref_pts = chosen.result.source_points, chosen.result.reference_points
    src_size = (source.info.width, source.info.height)
    metrics = Metrics(
        source_keypoints=chosen.result.source_keypoints,
        reference_keypoints=chosen.result.reference_keypoints,
        raw_matches=chosen.result.candidate_count,
        ratio_filtered_matches=len(chosen.result),
        inliers=chosen.inliers,
        inlier_ratio=chosen.inlier_ratio,
        registration_status=chosen.status,
        matcher=chosen.matcher,
        device=device_text(chosen),
        candidate_count=chosen.result.candidate_count,
        accepted_count=len(chosen.result),
    )

    consistency = None
    transformation = None
    if consensus_geometry is not None and chosen.reprojection is not None:
        g = consensus_geometry
        idx = g.inlier_idx
        metrics.reprojection_rmse_px = chosen.reprojection.rmse_px
        metrics.median_reprojection_error_px = chosen.reprojection.median_px
        metrics.max_reprojection_error_px = chosen.reprojection.max_px
        metrics.global_model_inliers = g.global_inliers
        if g.strip is not None:
            metrics.cross_validated_rmse_px = _strip_cross_validated(src_pts[idx], ref_pts[idx], g, src_size, cfg.random_seed)
        else:
            metrics.cross_validated_rmse_px = cross_validated_rmse(src_pts[idx], ref_pts[idx], g.kind, seed=cfg.random_seed)
        metrics.expected_chance_inliers = expected_chance_inliers(
            len(chosen.result), cfg.geometry.reproj_threshold_px, int(chosen.search_area_px)
        )
        metrics.estimated_overlap_fraction = chosen.overlap_fraction
        if chosen.coverage is not None:
            metrics.image_space_spatial_coverage = chosen.coverage.coverage
            metrics.occupied_cells = chosen.coverage.occupied_cells
            metrics.valid_cells = chosen.coverage.valid_cells
            metrics.coverage_within_overlap = chosen.coverage.coverage_within_overlap
        if footprint is not None:
            consistency = metadata_consistency(g.matrix, footprint.affine, src_size, reference_gsd_m(reference.info))
        if chosen.accepted:
            # A rejected transformation is never returned as a result.
            metrics.transformation_type = g.kind
            transformation = Transformation(
                model=g.kind,
                method=g.method,
                matrix=g.matrix.tolist(),
                mirrored_source=g.mirrored,
                seed_pool=g.seed_pool,
                seed_inliers=g.seed_inliers,
                properties=g.check.properties,
                strip_model=g.strip.to_dict() if g.strip is not None else None,
                strip_check=g.strip_check,
                refit_on_refined_points=chosen.initial_geometry is not None,
            )

    if chosen.accepted and chosen.geometry is not None:
        metrics.supported_along_track_fraction = _along_track_support(chosen, source, cfg)
    if chosen.holdout is not None:
        metrics.heldout_transfer_rmse_px = chosen.holdout.heldout_transfer_rmse_px
        metrics.heldout_median_px = chosen.holdout.heldout_median_px
        metrics.heldout_p90_px = chosen.holdout.heldout_p90_px
        metrics.block_heldout_rmse_px = chosen.holdout.block_rmse_px
    if chosen.tie_summary is not None:
        metrics.selected_tie_points = chosen.tie_summary.selected
        metrics.spatial_distribution_score = chosen.tie_summary.distribution_score
    if chosen.refined is not None:
        metrics.refined_tie_points = chosen.refinement_summary.refined
        metrics.median_refinement_px = chosen.refinement_summary.median_displacement_px
        flags = chosen.refined.refined
        if flags.any() and chosen.geometry is not None:
            errors = chosen.geometry.errors(chosen.refined.source[flags], chosen.refined.reference_refined[flags])
            metrics.refined_reprojection_rmse_px = float(np.sqrt(np.mean(errors**2)))
            metrics.refined_reprojection_median_px = float(np.median(errors))
    measured_decoys = [d.inliers for d in chosen.decoys if d.tiles_used > 0]
    if measured_decoys:
        metrics.decoy_max_inliers = max(measured_decoys)
    appearance = (chosen.result.extras or {}).get("appearance")
    if appearance:
        metrics.appearance_median_zncc = appearance.get("median_zncc_passed")

    if chosen.accepted and chosen.geometry is not None and chosen.geometry.mirrored:
        notes.append(
            "The source had to be mirrored left-right to match the reference; the reported matrix "
            "includes that mirror and therefore has a negative determinant."
        )
    if chosen.accepted and consistency is not None:
        if consistency.centre_offset_m is not None and consistency.centre_offset_m > cfg.validation.metadata_offset_warn_m:
            notes.append(
                f"The registered footprint centre lies {consistency.centre_offset_m / 1000:.2f} km from the one "
                "predicted by the XML/GCP metadata. That metadata is system-level pointing knowledge, so this is "
                "reported, not penalised."
            )
        if not consistency.orientation_agrees:
            notes.append(
                "The image content fixes an array orientation that differs from the one implied by the XML/GCP "
                f"metadata (rotation difference {consistency.rotation_difference_deg:.1f} deg, mirror "
                f"{'agrees' if consistency.mirror_agrees else 'differs'})."
            )
    if chosen.accepted and chosen.overlap_fraction is not None and chosen.overlap_fraction < 0.999:
        notes.append(
            f"Partial overlap: {chosen.overlap_fraction:.1%} of the valid source pixels land on valid reference "
            "pixels. 'coverage_within_overlap' counts only grid cells inside that overlap."
        )
    if chosen.accepted and chosen.geometry is not None and chosen.geometry.strip is not None:
        notes.append(
            "The mapping is a smooth along-track model, because one global transformation could not predict "
            "ground left out of its fit. The reported matrix is its best affine approximation, for reference only."
        )

    notes.extend(chosen.notes)
    artifacts: dict[str, str] = {}
    illumination = _illumination(source, reference, metadata, chosen, cfg)
    if out_dir is not None:
        say("rendering artifacts", 0.88)
        artifacts, more_notes = _write_artifacts(Path(out_dir), cfg, source, reference, chosen, footprint)
        notes.extend(more_notes)
        artifacts.update(exports.write_correspondence_tables(Path(out_dir), chosen))
        for o in outcomes:
            if o is not chosen:
                artifacts.update(_write_alternative(Path(out_dir), cfg, source, reference, o, footprint))

    metrics.processing_time_seconds = round(time.perf_counter() - t_start, 3)
    failure_reasons = list(chosen.failures)
    if not chosen.accepted and len(outcomes) > 1:
        failure_reasons = [f"{o.matcher}: {'; '.join(o.failures) or 'no accepted result'}" for o in outcomes]
    result = RegistrationResult(
        scene_id=scene.scene_id if scene else None,
        sensor=scene.sensor if scene else (metadata.sensor if metadata else None),
        reference_sensor=scene.reference_sensor if scene else None,
        status=chosen.status,
        failure_reasons=failure_reasons,
        warnings=chosen.warnings,
        metrics=metrics,
        selected_preprocessing=chosen.representation,
        transformation=transformation,
        spatial_coverage=chosen.coverage,
        metadata_consistency=consistency,
        predicted_footprint=footprint,
        attempts=[a for o in outcomes for a in o.attempts],
        source=_image_summary(source),
        reference=_image_summary(reference),
        metadata=metadata,
        artifacts=artifacts,
        config=cfg.model_dump(),
        notes=notes,
        requested_matcher=cfg.matcher,
        selected_matcher=chosen.matcher if chosen.accepted else None,
        device=device_text(chosen),
        matcher_results=[_summary(o) for o in outcomes],
        selection=selection,
        overlap_estimate=overlap,
        illumination=illumination,
        tie_points=chosen.tie_summary,
        refinement=chosen.refinement_summary,
        refinement_validation=chosen.refinement_validation,
        holdout=chosen.holdout,
        decoys=chosen.decoys,
        appearance=appearance,
        evidence=EvidenceSummary(),
        search_trace=chosen.trace,
    )
    if out_dir is not None:
        say("writing the report", 0.97)
        result.artifacts.update(exports.write_result_documents(Path(out_dir), result))
    say("done", 1.0)
    return result


def device_text(o: BranchOutcome) -> str:
    if o.availability == "UNAVAILABLE":
        return "n/a"
    device = o.result.device
    if device in ("cuda", "cpu") and o.matcher != SIFT:
        try:
            return device_label(device)
        except (ImportError, ModelUnavailable):
            return device
    return device


def _strip_cross_validated(src: np.ndarray, dst: np.ndarray, g: VerifiedGeometry, size: tuple[int, int], seed: int, folds: int = 5) -> float | None:
    """k-fold reprojection RMSE for the strip model, like ``cross_validated_rmse`` for matrices."""
    n = len(src)
    if n < 2 * folds:
        return None
    order = np.random.default_rng(seed).permutation(n)
    errors: list[np.ndarray] = []
    for k in range(folds):
        test = order[k::folds]
        train = np.setdiff1d(order, test, assume_unique=True)
        model = fit_strip(src[train], dst[train], size, g.strip.spacing_px, g.strip.smoothness)
        if model is None:
            return None
        errors.append(model.errors(src[test], dst[test]) ** 2)
    return float(np.sqrt(np.mean(np.concatenate(errors))))


# ------------------------------------------------------------------------------------- artifacts


def _outline(g: VerifiedGeometry, width: int, height: int, samples: int = 41) -> np.ndarray:
    """Footprint of the source on the reference; a polygon that follows a curved mapping."""
    if g.strip is None:
        return apply_transform(g.matrix, _source_corners(width, height))
    t = np.linspace(0.0, 1.0, samples)
    top = np.c_[t * width, np.zeros(samples)]
    right = np.c_[np.full(samples, width), t * height]
    bottom = np.c_[(1.0 - t) * width, np.full(samples, height)]
    left = np.c_[np.zeros(samples), (1.0 - t) * height]
    return g.apply(np.vstack([top, right, bottom, left]))


def warp_geometry(g: VerifiedGeometry, image: np.ndarray, shape: tuple[int, int], outline: np.ndarray, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    if g.strip is None:
        return warp_to_reference(image, g.matrix, shape, interpolation)
    margin = 8
    bbox = (
        int(np.floor(outline[:, 0].min())) - margin,
        int(np.floor(outline[:, 1].min())) - margin,
        int(np.ceil(outline[:, 0].max())) + margin,
        int(np.ceil(outline[:, 1].max())) + margin,
    )
    return warp_with_inverse(image, g.strip.inverse, shape, bbox, interpolation)


def _warp_mask(g: VerifiedGeometry, valid: np.ndarray, shape: tuple[int, int], outline: np.ndarray) -> np.ndarray:
    if g.strip is None:
        return warp_mask_to_reference(valid, g.matrix, shape)
    return warp_geometry(g, valid.astype(np.uint8), shape, outline, cv2.INTER_NEAREST).astype(bool)


def _write_artifacts(
    out: Path,
    cfg: PipelineConfig,
    source: RasterData,
    reference: RasterData,
    o: BranchOutcome,
    footprint: PredictedFootprint | None,
) -> tuple[dict[str, str], list[str]]:
    out.mkdir(parents=True, exist_ok=True)
    viz = cfg.visualization
    files: dict[str, str] = {}
    notes: list[str] = []
    if o.source_pre is None or o.reference_pre is None:
        return files, notes  # the matcher was unavailable: there is nothing to draw
    src_disp, ref_disp = o.source_pre.display, o.reference_pre.display
    src_pts, ref_pts = o.result.source_points, o.result.reference_points
    corners = _source_corners(source.info.width, source.info.height)

    def put(key: str, name: str, image: np.ndarray) -> None:
        render.save_image(out / name, image, viz.jpeg_quality)
        files[key] = name

    put("source", "source.jpg", render.preview(src_disp, viz))
    put("reference", "reference.jpg", render.preview(ref_disp, viz))
    if o.keypoints_source is not None:
        put("keypoints_source", "keypoints_source.jpg", render.draw_keypoints(src_disp, o.keypoints_source, viz))
    if o.keypoints_reference is not None:
        put("keypoints_reference", "keypoints_reference.jpg", render.draw_keypoints(ref_disp, o.keypoints_reference, viz))
    label = "Ratio-filtered candidate matches" if o.matcher == SIFT else f"{o.matcher} accepted candidates"
    put("raw_matches", "raw_matches.jpg", render.draw_matches(src_disp, ref_disp, src_pts, ref_pts, viz, render.RED, label))

    predicted_quad = None
    if footprint is not None:
        predicted_quad = apply_transform(np.vstack([np.asarray(footprint.affine), [0, 0, 1]]), corners)

    # Inliers, footprint, registered image and overlay exist only for an accepted transformation.
    g = o.geometry
    if g is None or not o.accepted:
        put("reference_footprint", "reference_footprint.jpg", render.draw_footprint(ref_disp, None, predicted_quad, viz))
        return files, notes

    idx = g.inlier_idx
    quad = _outline(g, source.info.width, source.info.height)
    put(
        "inlier_matches",
        "inlier_matches.jpg",
        render.draw_matches(src_disp, ref_disp, src_pts[idx], ref_pts[idx], viz, render.GREEN, "Inlier matches"),
    )
    if o.coverage is not None:
        put("spatial_coverage", "spatial_coverage.jpg", render.draw_coverage(src_disp, o.coverage, src_pts[idx], viz))
    put("reference_footprint", "reference_footprint.jpg", render.draw_footprint(ref_disp, quad, predicted_quad, viz))
    if o.selected_idx is not None:
        selected = src_pts[idx][o.selected_idx]
        put(
            "tie_points",
            "tie_points.jpg",
            render.draw_tie_points(src_disp, src_pts[idx], selected, cfg.tie_points.grid_rows, cfg.tie_points.grid_cols, o.overlap_mask, viz),
        )

    shape = reference.data.shape
    reg_mask = _warp_mask(g, source.valid_mask, shape, quad)
    registered = np.where(reg_mask, warp_geometry(g, src_disp, shape, quad), 0).astype(np.uint8)
    put("registered_source", "registered_source.png", render.preview(registered, viz))
    put("overlay", "overlay.png", render.make_overlay(ref_disp, registered, reg_mask, quad, viz))
    put("checkerboard", "checkerboard.jpg", render.make_checkerboard(ref_disp, registered, reg_mask, quad, viz))
    reference_crop, registered_crop = render.aligned_crops(ref_disp, registered, reg_mask, quad, viz)
    put("reference_crop", "reference_crop.jpg", reference_crop)
    put("registered_crop", "registered_crop.png", registered_crop)
    if cfg.write_geotiff:
        notes.extend(
            write_registered_geotiff(
                out / "registered_source.tif",
                source.data,
                source.valid_mask,
                g.matrix,
                reference.info,
                warp=(lambda data: warp_geometry(g, data, shape, quad)) if g.strip is not None else None,
                warp_mask=(lambda valid: _warp_mask(g, valid, shape, quad)) if g.strip is not None else None,
                matcher=o.matcher,
            )
        )
        files["registered_geotiff"] = "registered_source.tif"
    return files, notes


def _write_alternative(
    out: Path,
    cfg: PipelineConfig,
    source: RasterData,
    reference: RasterData,
    o: BranchOutcome,
    footprint: PredictedFootprint | None,
) -> dict[str, str]:
    """A few views of a matcher that was not selected, so that results can be compared side by side."""
    if o.source_pre is None or o.geometry is None or not o.accepted:
        return {}
    prefix = o.matcher.lower()
    viz = cfg.visualization
    src_disp, ref_disp = o.source_pre.display, o.reference_pre.display
    g = o.geometry
    idx = g.inlier_idx
    src_pts, ref_pts = o.result.source_points, o.result.reference_points
    quad = _outline(g, source.info.width, source.info.height)
    shape = reference.data.shape
    reg_mask = _warp_mask(g, source.valid_mask, shape, quad)
    registered = np.where(reg_mask, warp_geometry(g, src_disp, shape, quad), 0).astype(np.uint8)
    files = {
        f"{prefix}_inlier_matches": f"{prefix}__inlier_matches.jpg",
        f"{prefix}_overlay": f"{prefix}__overlay.png",
    }
    render.save_image(out / files[f"{prefix}_inlier_matches"], render.draw_matches(src_disp, ref_disp, src_pts[idx], ref_pts[idx], viz, render.GREEN, f"{o.matcher} inliers"), viz.jpeg_quality)
    render.save_image(out / files[f"{prefix}_overlay"], render.make_overlay(ref_disp, registered, reg_mask, quad, viz), viz.jpeg_quality)
    files.update(exports.write_correspondence_tables(out, o, prefix=f"{prefix}__"))
    return files


def register_scene(
    scene: SceneRecord,
    cfg: PipelineConfig | None = None,
    out_dir: str | Path | None = None,
    progress: ProgressFn | None = None,
) -> RegistrationResult:
    """Load one indexed scene from disk and register it."""
    cfg = cfg or PipelineConfig()
    if progress:
        progress("loading rasters", 0.01)
    fill = cfg.preprocess.min_border_fill_area_px
    source = load_raster(scene.source_path, min_border_fill_area_px=fill)
    reference = load_raster(scene.reference_path, min_border_fill_area_px=fill)
    metadata = parse_scene_xml(scene.xml_path)
    return register(source, reference, cfg, scene=scene, metadata=metadata, out_dir=out_dir, progress=progress)


__all__ = ["LOFTR", "RIFT", "SIFT", "SUPERPOINT_LIGHTGLUE", "BranchOutcome", "grade", "register", "register_scene"]
