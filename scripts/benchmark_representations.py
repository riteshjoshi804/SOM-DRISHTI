"""Which image representation works best, per scene and matcher?

    .venv\\Scripts\\python.exe scripts\\benchmark_representations.py
    .venv\\Scripts\\python.exe scripts\\benchmark_representations.py --scene 2470502 --matchers LOFTR

Run scripts\\benchmark_matchers.py first: it provides the locations used here.

Protocol, the same for every representation:

* **SIFT**: the normal SIFT branch on that one representation, without pooling.
* **Learned matchers, where the benchmark found the scene's location**: one tile pass around
  that location, then the shared verification. This compares representations fairly and
  cheaply, but it starts from a known location, so its counts are not full-run results.
* **Learned matchers on scenes with no known location**: the anchor search on each
  representation, with a smaller budget. This asks whether any representation finds one.

Values are verified inliers. Nothing is selected or tuned from this output automatically.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone

import numpy as np
from _bootstrap import keep_awake

from app import config
from app.config import PipelineConfig
from app.geometry.estimation import apply_transform
from app.geometry.strip_model import StripModel
from app.geometry.verification import verify_candidates
from app.ingestion.indexer import discover_scenes
from app.ingestion.raster import load_raster
from app.matching.guided import ImageView, build_hypotheses, match_tiles, plan_tile_centres, sweep_anchor, working_scale
from app.matching.registry import create_pair_matcher
from app.metadata.overlap import estimate_overlap
from app.metadata.xml_parser import parse_scene_xml
from app.preprocessing.pipeline import preprocess
from app.registration import store
from app.registration.engine import register

MODES = ["raw_normalized", "clahe", "gradient", "structural", "phase_congruency", "shadow_aware"]
ANCHOR_BUDGET = 150


def known_locations() -> dict[str, dict]:
    """Accepted transformations of the latest matcher benchmark, per scene and matcher."""
    path = config.RESULTS_ROOT / "matcher_benchmark.json"
    if not path.is_file():
        return {}
    out: dict[str, dict] = {}
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    for row in rows:
        if row["registration_status"] in ("FAILED", "UNAVAILABLE") or not row.get("job_id"):
            continue
        result = store.load_result(row["job_id"])
        if result and result.get("transformation"):
            # The selected matcher's model is the best estimate of where the scene lies.
            if row.get("selected_by_auto") or row["scene_id"] not in out:
                out[row["scene_id"]] = result["transformation"]
    return out


def predictor(transformation: dict):
    if transformation.get("strip_model"):
        strip = StripModel.from_dict(transformation["strip_model"])
        return strip.apply, lambda c: strip.jacobian(c[None])[0], np.asarray(transformation["matrix"], float)
    matrix = np.asarray(transformation["matrix"], float)
    return (lambda p: apply_transform(matrix, p)), (lambda c: matrix[:2, :2]), matrix


def sift_row(source, reference, metadata, modes) -> dict:
    values, seconds = {}, {}
    for mode in modes:
        cfg = PipelineConfig(matcher="SIFT", preprocessing_modes=[mode], pool_modes=False, write_geotiff=False)
        cfg.refinement.enabled = False
        cfg.holdout.enabled = False
        t0 = time.perf_counter()
        result = register(source, reference, cfg, metadata=metadata)
        values[mode] = result.metrics.inliers
        seconds[mode] = round(time.perf_counter() - t0, 1)
        print(f"    SIFT {mode:18s} inliers {result.metrics.inliers}  {result.status}  {seconds[mode]} s", flush=True)
    return {"values": values, "seconds": seconds, "protocol": "full SIFT branch"}


def seeded_row(name, source, reference, transformation, modes) -> dict:
    predict, local, matrix = predictor(transformation)
    cfg = PipelineConfig()
    matcher = create_pair_matcher(name, cfg)
    w, h = source.info.width, source.info.height
    short = min(w, h) * float(np.mean(np.linalg.svd(matrix[:2, :2], compute_uv=False)))
    scale = working_scale(cfg.guided.fine_scale, short, 200.0)
    fov = cfg.guided.fine_tile_px / scale
    centres = plan_tile_centres((w, h), matrix[:2, :2], fov, fov * cfg.guided.fine_step_fraction)
    values, seconds, details = {}, {}, {}
    for mode in modes:
        t0 = time.perf_counter()
        src = preprocess(source.data, source.valid_mask, mode, cfg.preprocess)
        ref = preprocess(reference.data, reference.valid_mask, mode, cfg.preprocess)
        tile_pass = match_tiles(
            matcher, ImageView(src.image, src.detect_mask), ImageView(ref.image, ref.detect_mask), predict, local, centres, cfg.guided,
            scale=scale, margin_px=float(cfg.guided.fine_margin_px), rng=np.random.default_rng(cfg.random_seed),
        )  # fmt: skip
        geometry, _ = verify_candidates(
            tile_pass.points0, tile_pass.points1, 1.0 - tile_pass.confidence, (w, h), (reference.info.width, reference.info.height),
            cfg.geometry, cfg.validation, cfg.guided, mirrored=bool(transformation.get("mirrored_source")), start=matrix,
        )  # fmt: skip
        values[mode] = len(geometry.inlier_idx) if geometry is not None else 0
        seconds[mode] = round(time.perf_counter() - t0, 1)
        details[mode] = {"accepted": len(tile_pass.points0), "model": geometry.kind if geometry is not None else None, "valid": bool(geometry.valid) if geometry is not None else False}
        print(f"    {name} {mode:18s} inliers {values[mode]}  accepted {len(tile_pass.points0)}  {seconds[mode]} s", flush=True)
    return {"values": values, "seconds": seconds, "details": details, "protocol": "one tile pass at the known location"}


def anchor_row(name, source, reference, metadata, modes) -> dict:
    cfg = PipelineConfig()
    cfg.guided.anchor_max_pairs = ANCHOR_BUDGET
    matcher = create_pair_matcher(name, cfg)
    overlap = estimate_overlap(source.info, reference.info, metadata, reference.valid_mask, cfg.guided, cfg.validation)
    size = (source.info.width, source.info.height)
    hypotheses = build_hypotheses(overlap, size, (reference.info.width, reference.info.height), cfg.guided)
    radius = overlap.search_radius_px if overlap.mode == "metadata" and overlap.search_radius_px else float(max(reference.info.width, reference.info.height))
    values, seconds, details = {}, {}, {}
    for mode in modes:
        t0 = time.perf_counter()
        src = preprocess(source.data, source.valid_mask, mode, cfg.preprocess)
        ref = preprocess(reference.data, reference.valid_mask, mode, cfg.preprocess)
        anchor, log, _ = sweep_anchor(
            matcher, ImageView(src.image, src.detect_mask), ImageView(ref.image, ref.detect_mask), hypotheses, radius, cfg.guided, lambda m, f: None
        )
        values[mode] = anchor.votes if anchor is not None else 0
        seconds[mode] = round(time.perf_counter() - t0, 1)
        details[mode] = {"anchor_found": anchor is not None, "last_log": log[-1] if log else None}
        print(f"    {name} {mode:18s} anchor {'found, ' + str(anchor.votes) + ' votes' if anchor else 'not found'}  {seconds[mode]} s", flush=True)
    return {"values": values, "seconds": seconds, "details": details, "protocol": f"anchor search, budget {ANCHOR_BUDGET} image pairs; value = votes of the anchor, 0 = none found"}


def main() -> int:
    keep_awake()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", help="Only scenes whose ID contains this")
    parser.add_argument("--matchers", default="SIFT,LOFTR", help="Comma list (default SIFT,LOFTR)")
    parser.add_argument("--modes", default=",".join(MODES))
    args = parser.parse_args()
    modes = args.modes.split(",")
    matchers = args.matchers.split(",")

    scenes = discover_scenes(config.DATA_ROOT).scenes
    if args.scene:
        scenes = [s for s in scenes if args.scene in s.scene_id]
    locations = known_locations()
    table: list[dict] = []
    out = config.RESULTS_ROOT / "representation_benchmark.json"
    for scene in scenes:
        print(f"\n[{scene.sensor}] {scene.scene_id}", flush=True)
        source, reference = load_raster(scene.source_path), load_raster(scene.reference_path)
        metadata = parse_scene_xml(scene.xml_path)
        for name in matchers:
            if name == "SIFT":
                row = sift_row(source, reference, metadata, modes)
            elif scene.scene_id in locations:
                row = seeded_row(name, source, reference, locations[scene.scene_id], modes)
            else:
                row = anchor_row(name, source, reference, metadata, modes)
            table.append({"scene_id": scene.scene_id, "sensor": scene.sensor, "matcher": name, **row})
            out.write_text(
                json.dumps(
                    {
                        "label": "Verified inliers per representation on the official scenes",
                        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "modes": modes,
                        "table": table,
                        "notes": [
                            "SIFT: the full SIFT branch on that representation, without pooling.",
                            "LoFTR on a scene whose location is known: one tile pass at that location, so counts are comparable with each other, not with full runs.",
                            f"LoFTR on a scene with no known location: anchor search with a budget of {ANCHOR_BUDGET} image pairs; the value is the anchor's votes, 0 means none was found.",
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
