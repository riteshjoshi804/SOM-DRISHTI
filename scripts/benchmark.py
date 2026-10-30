"""Benchmark every scene of one sensor and write a CSV of the measured results.

    .venv\\Scripts\\python.exe scripts\\benchmark.py --sensor OHRC
    .venv\\Scripts\\python.exe scripts\\benchmark.py --sensor IIRS --all-modes

Each scene is saved as a normal job under results/jobs/, so the web interface
can display it. Values in the CSV come straight from the engine; a scene that
fails is written as FAILED with its reason and empty metric cells.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from run_registration import add_pipeline_arguments, build_config, print_result

from app.config import RESULTS_ROOT
from app.ingestion.indexer import discover_scenes
from app.registration.engine import register_scene
from app.registration.schemas import RegistrationResult
from app.registration.store import job_dir, new_job_id, save_result

# Required columns, in the order of the MVP specification, then extra detail.
COLUMNS = [
    "scene_id", "source_width", "source_height", "reference_width", "reference_height",
    "source_keypoints", "reference_keypoints", "filtered_matches", "inliers", "inlier_ratio",
    "reprojection_rmse_px", "median_error_px", "spatial_coverage", "processing_time_seconds",
    "transformation_type", "status", "failure_reason",
    # extra
    "preprocessing", "mirrored_source", "raw_matches", "cross_validated_rmse_px",
    "coverage_within_overlap", "estimated_overlap_fraction", "expected_chance_inliers",
    "metadata_centre_offset_m", "metadata_rotation_difference_deg", "warnings", "job_id",
]  # fmt: skip

ATTEMPT_COLUMNS = [
    "scene_id", "preprocessing", "mirrored_source", "source_keypoints", "reference_keypoints",
    "ratio_filtered_matches", "inliers", "inlier_ratio", "status", "reasons", "seconds",
]  # fmt: skip


def _num(value, digits: int = 4):
    if value is None:
        return ""
    return round(value, digits) if isinstance(value, float) else value


def row_for(result: RegistrationResult, job_id: str) -> dict:
    m = result.metrics
    c = result.metadata_consistency
    return {
        "scene_id": result.scene_id,
        "source_width": result.source.width,
        "source_height": result.source.height,
        "reference_width": result.reference.width,
        "reference_height": result.reference.height,
        "source_keypoints": m.source_keypoints,
        "reference_keypoints": m.reference_keypoints,
        "filtered_matches": m.ratio_filtered_matches,
        "inliers": m.inliers,
        "inlier_ratio": _num(m.inlier_ratio),
        "reprojection_rmse_px": _num(m.reprojection_rmse_px, 3),
        "median_error_px": _num(m.median_reprojection_error_px, 3),
        "spatial_coverage": _num(m.image_space_spatial_coverage, 3),
        "processing_time_seconds": _num(m.processing_time_seconds, 1),
        "transformation_type": m.transformation_type or "",
        "status": result.status,
        "failure_reason": "; ".join(result.failure_reasons),
        "preprocessing": result.selected_preprocessing,
        "mirrored_source": result.transformation.mirrored_source if result.transformation else "",
        "raw_matches": m.raw_matches,
        "cross_validated_rmse_px": _num(m.cross_validated_rmse_px, 3),
        "coverage_within_overlap": _num(m.coverage_within_overlap, 3),
        "estimated_overlap_fraction": _num(m.estimated_overlap_fraction, 3),
        "expected_chance_inliers": _num(m.expected_chance_inliers, 5),
        "metadata_centre_offset_m": _num(c.centre_offset_m, 0) if c and result.status != "FAILED" else "",
        "metadata_rotation_difference_deg": _num(c.rotation_difference_deg, 2) if c and result.status != "FAILED" else "",
        "warnings": "; ".join(result.warnings),
        "job_id": job_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sensor", required=True, choices=["OHRC", "IIRS", "TMC-2"])
    parser.add_argument("--out", help="CSV path (default results/<sensor>_benchmark.csv)")
    add_pipeline_arguments(parser)
    args = parser.parse_args()

    scenes = discover_scenes(args.data_dir).by_sensor(args.sensor)
    if not scenes:
        raise SystemExit(f"No {args.sensor} scenes found under {args.data_dir}")
    cfg = build_config(args)
    stem = args.sensor.lower().replace("-", "")
    out = Path(args.out) if args.out else RESULTS_ROOT / f"{stem}_benchmark.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    rows, attempt_rows, summary = [], [], []
    for scene in scenes:
        print("\n" + "=" * 100 + f"\n[{scene.sensor}] {scene.scene_id}")
        job_id = new_job_id(scene.scene_id)
        result = register_scene(scene, cfg, job_dir(job_id))
        save_result(job_dir(job_id), result)
        print_result(result)
        rows.append(row_for(result, job_id))
        for a in result.attempts:
            attempt_rows.append(
                {
                    "scene_id": scene.scene_id,
                    "preprocessing": a.preprocessing_mode,
                    "mirrored_source": a.mirrored_source,
                    "source_keypoints": a.source_keypoints,
                    "reference_keypoints": a.reference_keypoints,
                    "ratio_filtered_matches": a.ratio_filtered_matches,
                    "inliers": a.inliers,
                    "inlier_ratio": _num(a.inlier_ratio),
                    "status": a.status,
                    "reasons": "; ".join(a.reasons),
                    "seconds": a.seconds,
                }
            )
        summary.append({"scene_id": scene.scene_id, "job_id": job_id, "status": result.status})

    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    attempts_out = out.with_name(out.stem + "_attempts.csv")
    with attempts_out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ATTEMPT_COLUMNS)
        writer.writeheader()
        writer.writerows(attempt_rows)
    out.with_suffix(".json").write_text(
        json.dumps({"sensor": args.sensor, "config": cfg.model_dump(), "scenes": summary}, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 100)
    print(f"{args.sensor} benchmark written to {out}")
    print(f"Per-representation attempts written to {attempts_out}")
    for r in rows:
        print(f"  {r['scene_id']}: {r['status']:14s} inliers={r['inliers']} rmse={r['reprojection_rmse_px'] or 'N/A'} "
              f"coverage={r['spatial_coverage'] or 'N/A'} {r['failure_reason']}")


if __name__ == "__main__":
    main()
