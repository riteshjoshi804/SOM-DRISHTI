"""Synthetic validation against exactly known transformations. NOT lunar data.

    .venv\\Scripts\\python.exe scripts\\synthetic_validation.py
    .venv\\Scripts\\python.exe scripts\\synthetic_validation.py --matchers SIFT,LOFTR --cases baseline,sun_azimuth_180

Writes results/synthetic_validation.csv and .json. Every row is labelled as synthetic. For
each tie point the truth is known exactly, so this is also the direct test of whether sub-pixel
refinement reduces error: the same points are measured before and after it.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from _bootstrap import keep_awake

from app import config
from app.config import PipelineConfig
from app.evaluation.synthetic import SYNTHETIC_LABEL, SyntheticOutcome, build_cases, truth_errors
from app.geometry.estimation import apply_transform
from app.geometry.strip_model import StripModel
from app.matching.registry import matcher_statuses
from app.registration.engine import register


def mapping(result):
    t = result.transformation
    if t is None:
        return None
    if t.strip_model is not None:
        return StripModel.from_dict(t.strip_model).apply
    matrix = np.asarray(t.matrix, dtype=np.float64)
    return lambda points: apply_transform(matrix, points)


def tie_errors(out_dir: Path, truth: np.ndarray) -> dict:
    """Error of each refined tie point against the known truth, before and after refinement."""
    path = out_dir / "tie_points_refined.csv"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    refined = [r for r in rows if r["refined"] == "1"]
    if not refined:
        return {"refined_points": 0, "tie_points": len(rows)}
    src = np.array([[float(r["source_x"]), float(r["source_y"])] for r in refined])
    initial = np.array([[float(r["reference_x_initial"]), float(r["reference_y_initial"])] for r in refined])
    after = np.array([[float(r["reference_x_refined"]), float(r["reference_y_refined"])] for r in refined])
    target = apply_transform(truth, src)
    e0, e1 = np.linalg.norm(initial - target, axis=1), np.linalg.norm(after - target, axis=1)
    return {
        "tie_points": len(rows),
        "refined_points": len(refined),
        "tie_error_initial_median_px": float(np.median(e0)),
        "tie_error_refined_median_px": float(np.median(e1)),
        "tie_error_initial_rmse_px": float(np.sqrt(np.mean(e0**2))),
        "tie_error_refined_rmse_px": float(np.sqrt(np.mean(e1**2))),
    }


def main() -> int:
    keep_awake()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matchers", default="SIFT,SUPERPOINT_LIGHTGLUE,LOFTR", help="Comma list")
    parser.add_argument("--cases", help="Comma list of case names (default: all)")
    args = parser.parse_args()

    usable = {s.name for s in matcher_statuses() if s.available or s.needs_download}
    matchers = [m for m in args.matchers.split(",") if m]
    cases = build_cases()
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in cases if c.name in wanted]

    root = config.RESULTS_ROOT
    work = root / "synthetic"
    rows: list[dict] = []
    for case in cases:
        for matcher in matchers:
            if matcher not in usable:
                rows.append(SyntheticOutcome(case=case.name, group=case.group, matcher=matcher, status="UNAVAILABLE").model_dump())
                continue
            cfg = PipelineConfig(matcher=matcher, write_geotiff=False)
            cfg.guided.decoy_controls = 1
            cfg.guided.anchor_max_pairs = 400
            cfg.preprocessing_modes = ["clahe"]
            out = work / f"{case.name}__{matcher}"
            t0 = time.perf_counter()
            print(f"{case.name:18s} {matcher:22s}", end=" ", flush=True)
            result = register(case.source, case.reference, cfg, out_dir=out, overlap=case.prior)
            seconds = time.perf_counter() - t0
            model = mapping(result)
            truth = truth_errors(model, case.truth, case.source.valid_mask, (case.reference.info.width, case.reference.info.height)) if model else {}
            ties = tie_errors(out, case.truth) if model else {}
            row = SyntheticOutcome(
                case=case.name,
                group=case.group,
                matcher=matcher,
                status=result.status,
                model=result.metrics.transformation_type,
                inliers=result.metrics.inliers,
                candidates=result.metrics.accepted_count,
                truth_rmse_px=truth.get("rmse"),
                truth_median_px=truth.get("median"),
                truth_max_px=truth.get("max"),
                seconds=round(seconds, 1),
                failure="; ".join(result.failure_reasons) or None,
                **ties,
            ).model_dump()
            rows.append(row)
            print(
                f"{row['status']:14s} inliers={row['inliers']} truth rmse={row['truth_rmse_px']} "
                f"tie {row['tie_error_initial_median_px']} -> {row['tie_error_refined_median_px']} ({seconds:.0f} s)",
                flush=True,
            )
            _save(root, rows)
    _save(root, rows)
    return 0


def _save(root: Path, rows: list[dict]) -> None:
    columns = list(rows[0].keys())
    with (root / "synthetic_validation.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (root / "synthetic_validation.json").write_text(
        json.dumps({"label": SYNTHETIC_LABEL, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rows": rows}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    sys.exit(main())
