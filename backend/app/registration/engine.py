"""Registration engine: preprocessing -> SIFT -> matching -> verified geometry -> metrics.

The engine tries each configured preprocessing representation and, only when a
representation yields no geometry at all, the same representation with the
source mirrored left-right (SIFT is not mirror-invariant and one official
scene is delivered mirrored). The strongest *valid* geometry wins. If nothing
valid is found the result is FAILED with the reasons - never a guess.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
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
from app.geometry.consensus import Consensus, find_consensus
from app.geometry.estimation import apply_transform
from app.geometry.validation import TransformCheck, check_transform
from app.ingestion.indexer import SceneRecord
from app.ingestion.raster import RasterData, compute_stats, load_raster
from app.matching.matcher import MatchSet, match_features
from app.matching.sift import Features, extract_sift
from app.metadata.georef import PredictedFootprint, predict_footprint, reference_gsd_m
from app.metadata.xml_parser import SceneMetadata, parse_scene_xml
from app.preprocessing.pipeline import Preprocessed, preprocess
from app.registration.schemas import (
    AttemptSummary,
    ImageSummary,
    Metrics,
    RegistrationResult,
    Status,
    Transformation,
)
from app.registration.warp import (
    source_overlap_mask,
    warp_mask_to_reference,
    warp_to_reference,
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
    n_src: int,
    n_ref: int,
    n_matches: int,
    inliers: int | None,
    check: TransformCheck | None,
    coverage: float | None,
    rmse: float | None,
) -> tuple[Status, list[str], list[str]]:
    """(status, failure reasons, low-confidence warnings) from measured quantities.

    ``inliers`` is None when no geometric consensus was found at all.
    """
    v = cfg.validation
    fail: list[str] = []
    warn: list[str] = []
    if n_src < v.min_source_keypoints:
        fail.append(f"insufficient source keypoints ({n_src} < {v.min_source_keypoints})")
    if n_ref < v.min_reference_keypoints:
        fail.append(f"insufficient reference keypoints ({n_ref} < {v.min_reference_keypoints})")
    if not fail and n_matches < v.min_matches:
        fail.append(f"insufficient matches ({n_matches} ratio-filtered < {v.min_matches})")
    if fail:
        return "FAILED", fail, warn
    if inliers is None:
        return "FAILED", ["too few inliers: no transformation is supported by enough spatially distinct matches"], warn

    ratio = inliers / n_matches if n_matches else 0.0
    if inliers < v.min_inliers:
        fail.append(f"too few inliers ({inliers} < {v.min_inliers})")
    if ratio < v.min_inlier_ratio:
        fail.append(f"low inlier ratio ({ratio:.2%} < {v.min_inlier_ratio:.2%})")
    if check is not None and not check.valid:
        fail.append("unstable transformation: " + "; ".join(check.reasons))
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
    """Geometric verification, measurement and grading of one set of matches."""
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


def register(
    source: RasterData,
    reference: RasterData,
    cfg: PipelineConfig | None = None,
    *,
    scene: SceneRecord | None = None,
    metadata: SceneMetadata | None = None,
    out_dir: str | Path | None = None,
    progress: ProgressFn | None = None,
) -> RegistrationResult:
    """Register ``source`` onto ``reference``. Writes artifacts to ``out_dir`` if given."""
    cfg = cfg or PipelineConfig()
    t_start = time.perf_counter()
    cv2.setRNGSeed(cfg.random_seed)  # makes OpenCV's RANSAC sampling repeatable
    say: ProgressFn = progress or (lambda message, fraction: None)
    notes = [RMSE_NOTE]

    footprint = predict_footprint(source.info, reference.info, metadata) if metadata is not None else None

    modes = list(dict.fromkeys(cfg.preprocessing_modes))
    attempts: list[_Attempt] = []
    best: _Attempt | None = None
    for i, mode in enumerate(modes):
        lo, step = 0.05 + 0.75 * i / len(modes), 0.75 / len(modes)
        say(f"[{mode}] preprocessing", lo)
        src_pre = preprocess(source.data, source.valid_mask, mode, cfg.preprocess)
        ref_pre = preprocess(reference.data, reference.valid_mask, mode, cfg.preprocess)
        say(f"[{mode}] SIFT on reference", lo + 0.2 * step)
        ref_features = extract_sift(ref_pre.image, ref_pre.detect_mask, cfg.sift)

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
    say("computing metrics", 0.82)
    consensus = best.consensus
    src_pts, ref_pts = best.tie_points()
    src_size = (source.info.width, source.info.height)

    metrics = Metrics(
        source_keypoints=len(best.source_features),
        reference_keypoints=len(best.reference_features),
        raw_matches=best.matches.raw_candidates,
        ratio_filtered_matches=len(best.matches),
        inliers=best.summary.inliers,
        inlier_ratio=best.summary.inlier_ratio,
        registration_status=best.status,
    )
    consistency = None
    transformation = None
    if consensus is not None and best.reprojection is not None:
        idx = consensus.inlier_idx
        metrics.reprojection_rmse_px = best.reprojection.rmse_px
        metrics.median_reprojection_error_px = best.reprojection.median_px
        metrics.max_reprojection_error_px = best.reprojection.max_px
        metrics.cross_validated_rmse_px = cross_validated_rmse(
            src_pts[idx], ref_pts[idx], consensus.model, seed=cfg.random_seed
        )
        metrics.expected_chance_inliers = expected_chance_inliers(
            len(best.matches), cfg.geometry.reproj_threshold_px, int(reference.valid_mask.sum())
        )
        metrics.estimated_overlap_fraction = best.overlap_fraction
        if best.coverage is not None:
            metrics.image_space_spatial_coverage = best.coverage.coverage
            metrics.occupied_cells = best.coverage.occupied_cells
            metrics.valid_cells = best.coverage.valid_cells
            metrics.coverage_within_overlap = best.coverage.coverage_within_overlap
        if footprint is not None:
            consistency = metadata_consistency(
                consensus.matrix, footprint.affine, src_size, reference_gsd_m(reference.info)
            )
        if best.accepted and best.check is not None:
            # A rejected transformation is never returned as a result.
            metrics.transformation_type = consensus.model
            transformation = Transformation(
                model=consensus.model,
                method=consensus.method,
                matrix=consensus.matrix.tolist(),
                mirrored_source=best.mirrored,
                seed_pool=consensus.seed_pool,
                seed_inliers=consensus.seed_inliers,
                properties=best.check.properties,
            )

    if best.accepted and best.mirrored:
        notes.append(
            "The source had to be mirrored left-right to match the reference; the reported matrix "
            "includes that mirror and therefore has a negative determinant."
        )
    if best.accepted and consistency is not None:
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
    if best.accepted and best.overlap_fraction is not None and best.overlap_fraction < 0.999:
        notes.append(
            f"Partial overlap: {best.overlap_fraction:.1%} of the valid source pixels land on valid reference "
            "pixels. 'coverage_within_overlap' counts only grid cells inside that overlap."
        )

    artifacts: dict[str, str] = {}
    if out_dir is not None:
        say("rendering artifacts", 0.88)
        artifacts = _write_artifacts(Path(out_dir), cfg, source, reference, best, footprint)

    metrics.processing_time_seconds = round(time.perf_counter() - t_start, 3)
    say("done", 1.0)
    return RegistrationResult(
        scene_id=scene.scene_id if scene else None,
        sensor=scene.sensor if scene else (metadata.sensor if metadata else None),
        reference_sensor=scene.reference_sensor if scene else None,
        status=best.status,
        failure_reasons=best.failures,
        warnings=best.warnings,
        metrics=metrics,
        selected_preprocessing=best.mode,
        transformation=transformation,
        spatial_coverage=best.coverage,
        metadata_consistency=consistency,
        predicted_footprint=footprint,
        attempts=[a.summary for a in attempts],
        source=_image_summary(source),
        reference=_image_summary(reference),
        metadata=metadata,
        artifacts=artifacts,
        config=cfg.model_dump(),
        notes=notes,
    )


def _write_artifacts(
    out: Path,
    cfg: PipelineConfig,
    source: RasterData,
    reference: RasterData,
    best: _Attempt,
    footprint: PredictedFootprint | None,
) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    viz = cfg.visualization
    src_disp, ref_disp = best.source_pre.display, best.reference_pre.display
    src_pts, ref_pts = best.tie_points()
    corners = _source_corners(source.info.width, source.info.height)
    files: dict[str, str] = {}

    def put(key: str, name: str, image: np.ndarray) -> None:
        render.save_image(out / name, image, viz.jpeg_quality)
        files[key] = name

    put("source", "source.jpg", render.preview(src_disp, viz))
    put("reference", "reference.jpg", render.preview(ref_disp, viz))
    put("keypoints_source", "keypoints_source.jpg", render.draw_keypoints(src_disp, best.source_features.points, viz))
    put(
        "keypoints_reference",
        "keypoints_reference.jpg",
        render.draw_keypoints(ref_disp, best.reference_features.points, viz),
    )
    put(
        "raw_matches",
        "raw_matches.jpg",
        render.draw_matches(src_disp, ref_disp, src_pts, ref_pts, viz, render.RED, "Ratio-filtered candidate matches"),
    )

    predicted_quad = None
    if footprint is not None:
        predicted_quad = apply_transform(np.vstack([np.asarray(footprint.affine), [0, 0, 1]]), corners)

    # Inliers, footprint, registered image and overlay exist only for an accepted transformation.
    consensus = best.consensus
    if consensus is None or not best.accepted:
        put("reference_footprint", "reference_footprint.jpg", render.draw_footprint(ref_disp, None, predicted_quad, viz))
        return files

    idx = consensus.inlier_idx
    quad = apply_transform(consensus.matrix, corners)
    put(
        "inlier_matches",
        "inlier_matches.jpg",
        render.draw_matches(src_disp, ref_disp, src_pts[idx], ref_pts[idx], viz, render.GREEN, "Inlier matches"),
    )
    if best.coverage is not None:
        put("spatial_coverage", "spatial_coverage.jpg", render.draw_coverage(src_disp, best.coverage, src_pts[idx], viz))
    put("reference_footprint", "reference_footprint.jpg", render.draw_footprint(ref_disp, quad, predicted_quad, viz))

    shape = reference.data.shape
    reg_mask = warp_mask_to_reference(source.valid_mask, consensus.matrix, shape)
    registered = np.where(reg_mask, warp_to_reference(src_disp, consensus.matrix, shape), 0).astype(np.uint8)
    put("registered_source", "registered_source.png", render.preview(registered, viz))
    put("overlay", "overlay.png", render.make_overlay(ref_disp, registered, reg_mask, quad, viz))
    if cfg.write_geotiff:
        write_registered_geotiff(
            out / "registered_source.tif", source.data, source.valid_mask, consensus.matrix, reference.info
        )
        files["registered_geotiff"] = "registered_source.tif"
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
