"""Run every available matcher on every official scene under one protocol.

    .venv\\Scripts\\python.exe scripts\\benchmark_matchers.py
    .venv\\Scripts\\python.exe scripts\\benchmark_matchers.py --scene 2470502 --include-rift

Writes results/matcher_benchmark.csv and .json, and results/benchmark_charts/*.svg; each scene is
also a normal job under results/jobs/, with its images, tables and report. Results are saved after
every scene, so an interrupted run keeps what it measured. Matchers that cannot run are listed as
UNAVAILABLE. A run stopped by an exception is listed as FAILED with the exception as its reason and no
figures, and the benchmark goes on to the next scene. Every number comes from the run.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import keep_awake

from app import config
from app.config import PipelineConfig
from app.ingestion.indexer import discover_scenes
from app.matching.registry import matcher_statuses
from app.registration.engine import register_scene
from app.registration.exports import BENCHMARK_COLUMNS, benchmark_rows
from app.registration.store import job_dir, new_job_id, save_result

EXTRA_COLUMNS = ["selected_by_auto", "job_id"]
# One colour per matcher, whichever matchers a run contains (dark surface; validated as adjacent pairs).
COLOURS = {"SIFT": "#3987e5", "SUPERPOINT_LIGHTGLUE": "#d95926", "LOFTR": "#199e70", "RIFT": "#c98500"}
LABELS = {"SIFT": "SIFT / RootSIFT", "SUPERPOINT_LIGHTGLUE": "SuperPoint + LightGlue", "LOFTR": "LoFTR", "RIFT": "RIFT (experimental)"}


def environment() -> dict:
    import cv2
    import numpy as np

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
    }
    try:
        import kornia
        import torch
        import transformers

        info.update(torch=torch.__version__, kornia=kornia.__version__, transformers=transformers.__version__)
        info["device"] = "cuda" if torch.cuda.is_available() else f"cpu ({torch.get_num_threads()} threads)"
    except ImportError:
        info["device"] = "cpu (learned matchers not installed)"
    return info


def scene_config(sensor: str, matcher: str) -> PipelineConfig:
    cfg = PipelineConfig(matcher=matcher, sift_decoy_controls=1)
    if sensor == "IIRS":
        # The SIFT settings of the stored baseline, so that its figures stay comparable.
        cfg.preprocessing_modes = ["raw_normalized", "clahe", "gradient"]
        cfg.evaluate_all_modes = True
    return cfg


def chart(path: Path, title: str, subtitle: str, rows: list[dict], field: str, formatter) -> None:
    """Grouped horizontal bars as a stand-alone SVG, with the same marks as the web dashboard."""
    scenes = list(dict.fromkeys(r["scene_id"] for r in rows))
    matchers = list(dict.fromkeys(r["matcher"] for r in rows))
    lookup = {(r["scene_id"], r["matcher"]): r for r in rows}
    bar, gap, group_gap, label_w, right, top, bottom = 10, 2, 14, 150, 80, 66, 30
    group = len(matchers) * (bar + gap) - gap
    width, height = 860, top + len(scenes) * (group + group_gap) - group_gap + bottom
    values = [r[field] for r in rows if isinstance(r.get(field), (int, float)) and r["registration_status"] not in ("FAILED", "UNAVAILABLE")]
    if field == "processing_seconds":
        values = [r[field] for r in rows if isinstance(r.get(field), (int, float))]
    top_value = max(values) if values else 1.0
    plot = width - label_w - right
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="system-ui, Segoe UI, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#1a1a19"/>',
        f'<text x="16" y="24" fill="#f0efec" font-size="15" font-weight="600">{title}</text>',
        f'<text x="16" y="42" fill="#898781" font-size="12">{subtitle}</text>',
    ]
    x = 16
    for m in matchers:
        parts.append(f'<rect x="{x}" y="50" width="10" height="10" rx="2" fill="{COLOURS.get(m, "#898781")}"/>')
        parts.append(f'<text x="{x + 15}" y="59" fill="#c3c2b7" font-size="11">{LABELS.get(m, m)}</text>')
        x += 30 + 7 * len(LABELS.get(m, m))
    parts.append(f'<line x1="{label_w}" x2="{label_w}" y1="{top - 4}" y2="{height - bottom}" stroke="#383835"/>')
    for i, scene in enumerate(scenes):
        y0 = top + i * (group + group_gap)
        short = scene.split("CHO")[-1][:7] if "CHO" in scene else scene[:14]
        parts.append(f'<text x="{label_w - 8}" y="{y0 + group / 2 + 4}" fill="#c3c2b7" font-size="12" text-anchor="end">{short}</text>')
        for j, m in enumerate(matchers):
            y = y0 + j * (bar + gap)
            r = lookup.get((scene, m))
            status = r["registration_status"] if r else "not run"
            value = r.get(field) if r else None
            accepted = field == "processing_seconds" or status not in ("FAILED", "UNAVAILABLE")
            if isinstance(value, (int, float)) and accepted:
                w = max(value / top_value * plot, 2.0) if value > 0 else 0.0
                radius = min(4.0, bar / 2.0, w)
                parts.append(
                    f'<path d="M{label_w},{y} H{label_w + w - radius} Q{label_w + w},{y} {label_w + w},{y + radius} '
                    f'V{y + bar - radius} Q{label_w + w},{y + bar} {label_w + w - radius},{y + bar} H{label_w} Z" fill="{COLOURS.get(m, "#898781")}"/>'
                )
                parts.append(f'<text x="{label_w + w + 5}" y="{y + bar - 1}" fill="#c3c2b7" font-size="10">{formatter(value)}</text>')
            else:
                word = {"FAILED": "failed", "UNAVAILABLE": "unavailable"}.get(status, "not measured")
                parts.append(f'<text x="{label_w + 6}" y="{y + bar - 1}" fill="#898781" font-size="10">{word}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_outputs(rows: list[dict], cfg_used: dict, started: str, statuses: list[dict]) -> None:
    root = config.RESULTS_ROOT
    columns = BENCHMARK_COLUMNS + EXTRA_COLUMNS
    with (root / "matcher_benchmark.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "label": "Official SIH26166 sample, one protocol for every matcher. Errors are measured on automatic matches, not ground truth.",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "started_at": started,
        "environment": environment(),
        "matchers": statuses,
        "rows": rows,
        "config": cfg_used,
    }
    (root / "matcher_benchmark.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    charts = root / "benchmark_charts"
    charts.mkdir(exist_ok=True)
    chart(charts / "inlier_ratio.svg", "Inlier ratio", "Geometrically verified share of accepted matches", rows, "inlier_ratio", lambda v: f"{v * 100:.1f}%")
    chart(charts / "heldout_error.svg", "Held-out error", "Median, px; automatic matches left out of the fit, not ground truth", rows, "heldout_median_px", lambda v: f"{v:.2f} px")
    chart(charts / "spatial_coverage.svg", "Spatial coverage", "Share of valid 8 x 8 grid cells over the overlap that hold an inlier", rows, "overlap_spatial_coverage", lambda v: f"{v * 100:.0f}%")
    chart(charts / "processing_time.svg", "Processing time", "Seconds per scene, decoy controls included", rows, "processing_seconds", lambda v: f"{v:.0f} s")


def error_row(scene, matcher: str, exc: Exception, seconds: float) -> dict:
    """A run stopped by an exception measured nothing, so all its figures stay empty."""
    row = dict.fromkeys(BENCHMARK_COLUMNS + EXTRA_COLUMNS)
    row.update(
        scene_id=scene.scene_id, sensor=scene.sensor, matcher=matcher, registration_status="FAILED",
        processing_seconds=round(seconds, 1), selected_by_auto=False,
        reasons=f"the run stopped with an exception, nothing was measured: {type(exc).__name__}: {exc}",
    )  # fmt: skip
    return row


def run_scene(scene, matcher: str, job_id: str, progress):
    """The result and rows of one run. An exception fails that run only, so the benchmark can go on."""
    t0 = time.perf_counter()
    try:
        result = register_scene(scene, scene_config(scene.sensor, matcher), job_dir(job_id), progress)
        save_result(job_dir(job_id), result)
    except Exception as exc:
        traceback.print_exc()
        return None, [error_row(scene, matcher, exc, time.perf_counter() - t0)]
    rows = benchmark_rows(result)
    for row in rows:
        row["selected_by_auto"] = row["matcher"] == result.selected_matcher
        row["job_id"] = job_id
    return result, rows


def main() -> int:
    keep_awake()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", help="Only scenes whose ID contains this")
    parser.add_argument("--sensor", choices=["OHRC", "IIRS"], help="Only this sensor")
    parser.add_argument("--include-rift", action="store_true", help="Also run the experimental RIFT matcher")
    parser.add_argument("--rift-only", action="store_true", help="Run only RIFT and add its rows to the existing CSV/JSON")
    parser.add_argument("--keep", action="store_true", help="Add to the existing CSV/JSON instead of replacing it")
    args = parser.parse_args()

    scenes = discover_scenes(config.DATA_ROOT).scenes
    if args.scene:
        scenes = [s for s in scenes if args.scene in s.scene_id]
    if args.sensor:
        scenes = [s for s in scenes if s.sensor == args.sensor]
    if not scenes:
        raise SystemExit("No matching scenes")

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    statuses = [s.model_dump() for s in matcher_statuses()]
    rows: list[dict] = []
    existing = config.RESULTS_ROOT / "matcher_benchmark.json"
    if (args.keep or args.rift_only) and existing.is_file():
        rows = json.loads(existing.read_text(encoding="utf-8"))["rows"]
    cfg_used = PipelineConfig(matcher="AUTO", sift_decoy_controls=1).model_dump()

    for scene in scenes:
        t0 = time.perf_counter()
        print("\n" + "=" * 100 + f"\n[{scene.sensor}] {scene.scene_id}", flush=True)
        job_id = new_job_id(scene.scene_id)

        def progress(message: str, fraction: float) -> None:
            print(f"  [{fraction:4.0%}] {message}", flush=True)

        result, new_rows = (None, []) if args.rift_only else run_scene(scene, "AUTO", job_id, progress)
        if args.include_rift or args.rift_only:
            _, rift_rows = run_scene(scene, "RIFT", new_job_id(f"{scene.scene_id}_RIFT"), progress)
            for row in rift_rows:
                row["selected_by_auto"] = False
            new_rows += rift_rows
        # New rows replace older ones of the scene and matcher. An AUTO row exists only for a run that stopped.
        replaced = {(r["scene_id"], r["matcher"]) for r in new_rows}
        if not args.rift_only:
            replaced.add((scene.scene_id, "AUTO"))
        rows = [r for r in rows if (r["scene_id"], r["matcher"]) not in replaced] + new_rows
        write_outputs(rows, cfg_used, started, statuses)
        outcome = "RIFT only" if args.rift_only else f"{result.status}, selected {result.selected_matcher}" if result else "stopped with an exception"
        print(f"  -> {outcome}, {time.perf_counter() - t0:.0f} s", flush=True)
        for row in rows:
            if row["scene_id"] == scene.scene_id:
                print(
                    f"     {row['matcher']:22s} {row['registration_status']:14s} inliers={row['geometric_inliers']} ratio={row['inlier_ratio']} "
                    f"heldout_median={row['heldout_median_px']} coverage={row['overlap_spatial_coverage']} decoy={row['decoy_max_inliers']} t={row['processing_seconds']}s {row['reasons'][:110]}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
