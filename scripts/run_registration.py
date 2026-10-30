"""Register one scene of the official dataset and write all artifacts.

    .venv\\Scripts\\python.exe scripts\\run_registration.py
    .venv\\Scripts\\python.exe scripts\\run_registration.py --scene 2359602
    .venv\\Scripts\\python.exe scripts\\run_registration.py --scene 2436502 --modes clahe --ratio 0.85

``--scene`` accepts a full scene ID or any unique part of it. Without it the
first engineering case (DEFAULT_FIRST_SCENE in app/config.py) is used.
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from app.config import DATA_ROOT, DEFAULT_FIRST_SCENE, PipelineConfig
from app.ingestion.indexer import SceneRecord, discover_scenes
from app.registration.engine import register_scene
from app.registration.schemas import RegistrationResult
from app.registration.store import job_dir, new_job_id, save_result


def find_scene(key: str, data_dir: str = str(DATA_ROOT)) -> SceneRecord:
    scenes = discover_scenes(data_dir).scenes
    hits = [s for s in scenes if key.lower() in s.scene_id.lower()]
    if len(hits) != 1:
        listing = "\n  ".join(s.scene_id for s in scenes)
        raise SystemExit(f"'{key}' matches {len(hits)} scenes. Available:\n  {listing}")
    return hits[0]


def build_config(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig()
    if args.modes:
        cfg.preprocessing_modes = [m.strip() for m in args.modes.split(",")]
    if args.ratio is not None:
        cfg.match.ratio_threshold = args.ratio
    if args.model:
        cfg.geometry.model = args.model
    if args.method:
        cfg.geometry.method = args.method
    if args.threshold is not None:
        cfg.geometry.reproj_threshold_px = args.threshold
    if args.nfeatures is not None:
        cfg.sift.nfeatures = args.nfeatures
    if args.contrast is not None:
        cfg.sift.contrast_threshold = args.contrast
    if args.grid:
        rows, cols = (int(v) for v in args.grid.lower().split("x"))
        cfg.coverage.grid_rows, cfg.coverage.grid_cols = rows, cols
    if args.all_modes:
        cfg.evaluate_all_modes = True
    if args.no_pool:
        cfg.pool_modes = False
    if args.no_geotiff:
        cfg.write_geotiff = False
    return PipelineConfig.model_validate(cfg.model_dump())  # re-validate the overrides


def add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default=str(DATA_ROOT))
    parser.add_argument("--modes", help="Comma list of raw_normalized,clahe,gradient")
    parser.add_argument("--ratio", type=float, help="Lowe ratio threshold")
    parser.add_argument("--model", choices=["auto", "affine", "homography", "similarity"])
    parser.add_argument("--method", choices=["ransac", "magsac"])
    parser.add_argument("--threshold", type=float, help="Reprojection threshold in pixels")
    parser.add_argument("--nfeatures", type=int, help="SIFT keypoints per tile")
    parser.add_argument("--contrast", type=float, help="SIFT contrastThreshold")
    parser.add_argument("--grid", help="Coverage grid as ROWSxCOLS, e.g. 8x8")
    parser.add_argument("--all-modes", action="store_true", help="Evaluate every preprocessing mode")
    parser.add_argument("--no-pool", action="store_true", help="Do not pool candidates across modes")
    parser.add_argument("--no-geotiff", action="store_true")


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def print_result(result: RegistrationResult) -> None:
    m = result.metrics
    print(f"\nSTATUS: {result.status}")
    for reason in result.failure_reasons:
        print(f"  failure : {reason}")
    for warning in result.warnings:
        print(f"  warning : {warning}")
    print(f"  preprocessing used        : {result.selected_preprocessing}")
    print(f"  source / reference kp     : {fmt(m.source_keypoints)} / {fmt(m.reference_keypoints)}")
    print(f"  raw candidate matches     : {fmt(m.raw_matches)}")
    print(f"  ratio-filtered matches    : {fmt(m.ratio_filtered_matches)}")
    print(f"  inliers                   : {fmt(m.inliers)}   (ratio {fmt(m.inlier_ratio, 4)})")
    print(f"  expected chance inliers   : {fmt(m.expected_chance_inliers, 4)}")
    print(f"  transformation            : {fmt(m.transformation_type)}")
    print(f"  Reprojection RMSE (px)    : {fmt(m.reprojection_rmse_px)}   median {fmt(m.median_reprojection_error_px)}   max {fmt(m.max_reprojection_error_px)}")
    print(f"  k-fold CV RMSE (px)       : {fmt(m.cross_validated_rmse_px)}")
    print(f"  Image-space coverage      : {fmt(m.image_space_spatial_coverage)}   ({fmt(m.occupied_cells)}/{fmt(m.valid_cells)} cells)")
    print(f"  coverage within overlap   : {fmt(m.coverage_within_overlap)}   (overlap fraction {fmt(m.estimated_overlap_fraction)})")
    if result.transformation:
        p = result.transformation.properties
        print(f"  mirrored source           : {result.transformation.mirrored_source}")
        print(f"  scale / rotation          : {fmt(p.scale_minor)}..{fmt(p.scale_major)} / {fmt(p.rotation_deg, 2)} deg   inside reference {fmt(p.fraction_inside_reference)}")
        for row in result.transformation.matrix:
            print("      [" + "  ".join(f"{v: .6f}" for v in row) + "]")
    if result.metadata_consistency:
        c = result.metadata_consistency
        print(f"  vs metadata prediction    : footprint centre offset {c.centre_offset_px:.0f} px"
              + (f" ({c.centre_offset_m / 1000:.2f} km)" if c.centre_offset_m is not None else "")
              + f", rotation diff {c.rotation_difference_deg:.2f} deg, scale ratio {c.scale_ratio:.3f},"
              + f" orientation {'agrees' if c.orientation_agrees else 'DIFFERS'}")
    print(f"  processing time           : {fmt(m.processing_time_seconds, 1)} s")
    print("  attempts:")
    for a in result.attempts:
        print(f"    {a.preprocessing_mode:15s} mirrored={str(a.mirrored_source):5s} kp={a.source_keypoints}/{a.reference_keypoints} "
              f"matches={a.ratio_filtered_matches} inliers={a.inliers} -> {a.status} ({a.seconds:.1f}s)")
    for note in result.notes:
        print(f"  note: {note}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default=DEFAULT_FIRST_SCENE, help="Scene ID or unique part of it")
    add_pipeline_arguments(parser)
    args = parser.parse_args()

    scene = find_scene(args.scene, args.data_dir)
    cfg = build_config(args)
    job_id = new_job_id(scene.scene_id)
    out = job_dir(job_id)
    print(f"Scene     : [{scene.sensor}] {scene.scene_id}")
    print(f"Job       : {job_id}")

    result = register_scene(scene, cfg, out, progress=lambda msg, frac: print(f"  [{frac:4.0%}] {msg}"))
    save_result(out, result)
    print_result(result)
    print(f"\nArtifacts : {out}")
    for key, name in result.artifacts.items():
        print(f"  {key:20s} {name}")


if __name__ == "__main__":
    main()
