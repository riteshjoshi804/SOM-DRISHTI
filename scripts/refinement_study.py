"""Which sub-pixel estimator actually reduces error, and is any of them biased by the Sun?

    .venv\\Scripts\\python.exe scripts\\refinement_study.py

On rendered terrain the true position of every tie point is known exactly, so each estimator is
measured directly: error before and after refinement, on the same points. A changed Sun moves
shading and shadows, which can pull a correlation peak away from the true position, so a
method can look consistent while being systematically wrong. Only known truth can show that.

Writes results/refinement_study.json. SYNTHETIC: rendered terrain, not lunar data.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone

import _bootstrap  # noqa: F401
import numpy as np

from app import config
from app.config import PipelineConfig, RefinementConfig
from app.evaluation.synthetic import SYNTHETIC_LABEL, build_cases
from app.geometry.estimation import apply_transform
from app.matching.registry import create_pair_matcher
from app.preprocessing.pipeline import preprocess
from app.refinement.subpixel import refine_correspondences

CASES = ["baseline", "noise_0.05", "blur_1.5", "sun_azimuth_90", "sun_azimuth_180", "sun_low_5", "sun_low_opposite"]
STRATEGIES = ["ncc", "lucas_kanade", "ecc", "phase"]
REPRESENTATIONS = ["clahe", "gradient", "raw_normalized"]


def main() -> int:
    cases = [c for c in build_cases() if c.name in CASES]
    cfg = PipelineConfig()
    matcher = create_pair_matcher("LOFTR", cfg)
    rows = []
    for case in cases:
        src_pre = preprocess(case.source.data, case.source.valid_mask, "clahe", cfg.preprocess)
        ref_pre = preprocess(case.reference.data, case.reference.valid_mask, "clahe", cfg.preprocess)
        w, h = case.source.info.width, case.source.info.height
        rw, rh = case.reference.info.width, case.reference.info.height
        truth = case.truth

        # Starting points from a real matcher, kept where they lie near the truth (the tests below are
        # about refining verified correspondences, and here the truth does the verifying).
        rng = np.random.default_rng(0)
        import cv2

        src_all, ref_all = [], []
        matrix = truth[:2, :2]
        for cy in np.arange(160, h, 240):
            centre = np.array([w / 2.0, float(cy)])
            forward = np.c_[matrix, np.array([240.0, 240.0]) - matrix @ centre]
            tile = cv2.warpAffine(src_pre.image, forward, (480, 480), flags=cv2.INTER_LINEAR)
            tile_mask = cv2.warpAffine(src_pre.detect_mask, forward, (480, 480), flags=cv2.INTER_NEAREST)
            target = apply_transform(truth, centre[None])[0] + rng.uniform(-16, 16, 2)
            x0, y0 = int(np.clip(target[0] - 280, 0, rw - 560)), int(np.clip(target[1] - 280, 0, rh - 560))
            window = ref_pre.image[y0 : y0 + 560, x0 : x0 + 560]
            pair = matcher.match_pair(tile, window, tile_mask, ref_pre.detect_mask[y0 : y0 + 560, x0 : x0 + 560])
            if len(pair):
                inverse = np.linalg.inv(np.vstack([forward, [0.0, 0.0, 1.0]]))
                src_all.append(apply_transform(inverse, pair.points0))
                ref_all.append(pair.points1 + [x0, y0])
        src = np.concatenate(src_all)
        ref = np.concatenate(ref_all)
        inside = (src[:, 0] >= 0) & (src[:, 1] >= 0) & (src[:, 0] < w) & (src[:, 1] < h)
        src, ref = src[inside], ref[inside]
        initial_error = np.linalg.norm(ref - apply_transform(truth, src), axis=1)
        keep = initial_error < 3.0
        src, ref, initial_error = src[keep], ref[keep], initial_error[keep]
        print(f"\n{case.name}: {len(src)} starting points, initial error median {np.median(initial_error):.3f}, rmse {np.sqrt(np.mean(initial_error**2)):.3f} px", flush=True)
        if len(src) < 10:
            rows.append({"case": case.name, "group": case.group, "starting_points": int(len(src)), "note": "too few starting points"})
            continue

        for representation in REPRESENTATIONS:
            a = preprocess(case.source.data, case.source.valid_mask, representation, cfg.preprocess)
            b = preprocess(case.reference.data, case.reference.valid_mask, representation, cfg.preprocess)
            mask_a, mask_b = case.source.valid_mask.astype(np.uint8), case.reference.valid_mask.astype(np.uint8)
            for strategy in STRATEGIES:
                t0 = time.perf_counter()
                out = refine_correspondences(
                    a.image, mask_a, b.image, mask_b, src, ref,
                    lambda points: np.repeat(matrix[None], len(points), axis=0),
                    RefinementConfig(strategy=strategy, representation=representation), representation,
                )  # fmt: skip
                flags = out.refined
                row = {
                    "case": case.name,
                    "group": case.group,
                    "representation": representation,
                    "strategy": strategy,
                    "starting_points": int(len(src)),
                    "refined": int(flags.sum()),
                    "rejected_by_reason": out.summary.rejected_by_reason,
                    "seconds": round(time.perf_counter() - t0, 2),
                }
                if flags.sum() >= 5:
                    target = apply_transform(truth, src[flags])
                    before = np.linalg.norm(ref[flags] - target, axis=1)
                    after = np.linalg.norm(out.reference_refined[flags] - target, axis=1)
                    signed = out.reference_refined[flags] - target
                    row.update(
                        initial_median_px=float(np.median(before)),
                        refined_median_px=float(np.median(after)),
                        initial_rmse_px=float(np.sqrt(np.mean(before**2))),
                        refined_rmse_px=float(np.sqrt(np.mean(after**2))),
                        refined_p90_px=float(np.percentile(after, 90)),
                        refined_bias_px=np.round(np.median(signed, axis=0), 3).tolist(),
                        share_under_half_px=float(np.mean(after < 0.5)),
                        improved=bool(np.median(after) < np.median(before)),
                    )
                    print(
                        f"  {representation:15s} {strategy:13s} refined {row['refined']:4d}/{len(src):4d}  median {row['initial_median_px']:.3f} -> {row['refined_median_px']:.3f}  "
                        f"rmse {row['initial_rmse_px']:.3f} -> {row['refined_rmse_px']:.3f}  under 0.5 px {row['share_under_half_px']:.0%}  bias {row['refined_bias_px']}",
                        flush=True,
                    )
                else:
                    print(f"  {representation:15s} {strategy:13s} refined {row['refined']} of {len(src)}: too few to measure {out.summary.rejected_by_reason}", flush=True)
                rows.append(row)
    (config.RESULTS_ROOT / "refinement_study.json").write_text(
        json.dumps({"label": SYNTHETIC_LABEL, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rows": rows}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
