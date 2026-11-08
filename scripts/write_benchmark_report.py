"""Generate docs/benchmark_matchers.md from the result files, so that no figure is copied by hand.

    .venv\\Scripts\\python.exe scripts\\write_benchmark_report.py

Reads results/matcher_benchmark.json, representation_benchmark.json, synthetic_validation.json and
refinement_study.json. Sections whose file does not exist yet are left out.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import _bootstrap  # noqa: F401

from app import config
from app.config import PROJECT_ROOT

LABELS = {"SIFT": "SIFT", "SUPERPOINT_LIGHTGLUE": "SP+LG", "LOFTR": "LoFTR", "RIFT": "RIFT"}


def read(name: str):
    path = config.RESULTS_ROOT / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def n(value, digits: int = 2) -> str:
    if value is None or value == "":
        return "–"
    return f"{value:.{digits}f}" if isinstance(value, float) else f"{value:,}".replace(",", " ")


def p(value, digits: int = 1) -> str:
    return "–" if value is None else f"{value * 100:.{digits}f} %"


def short(scene_id: str) -> str:
    return scene_id.split("CHO")[-1][:7] if "CHO" in scene_id else scene_id


def table(header: list[str], rows: list[list[str]], align: str | None = None) -> list[str]:
    marks = align or ("-" * len(header))
    rule = "|" + "|".join("---:" if a == "r" else "---" for a in marks) + "|"
    return ["| " + " | ".join(header) + " |", rule, *("| " + " | ".join(str(c) for c in row) + " |" for row in rows)]


def matcher_section(data: dict) -> list[str]:
    env = data.get("environment", {})
    rows = data["rows"]
    out = [
        "## Matchers on the official scenes",
        "",
        f"Generated {data.get('generated_at')} by `scripts/benchmark_matchers.py`: Python {env.get('python')}, torch {env.get('torch')}, "
        f"Kornia {env.get('kornia')}, transformers {env.get('transformers')}, OpenCV {env.get('opencv')}, on {env.get('device')}. "
        "The same protocol applies to every matcher. **Errors below are measured on automatic matches, not against ground truth.**",
        "",
    ]
    out += table(
        ["Scene", "Sensor", "Matcher", "Status", "Model", "Accepted", "Inliers", "Ratio", "RMSE px", "Held-out RMSE / median / p90 px", "Unseen blocks px", "Coverage (overlap)", "Strip support", "Decoy max", "Time s"],
        [
            [
                short(r["scene_id"]) + (" *" if r.get("selected_by_auto") else ""),
                r["sensor"],
                LABELS.get(r["matcher"], r["matcher"]),
                r["registration_status"],
                r["transformation_model"] or "–",
                n(r["accepted_correspondences"]),
                n(r["geometric_inliers"]),
                p(r["inlier_ratio"]),
                n(r["reprojection_rmse_px"]),
                f"{n(r['heldout_rmse_px'])} / {n(r['heldout_median_px'])} / {n(r['heldout_p90_px'])}",
                n(r["block_heldout_rmse_px"]),
                p(r["overlap_spatial_coverage"]),
                p(r["along_track_support"], 0),
                n(r["decoy_max_inliers"]),
                n(r["processing_seconds"], 0),
            ]
            for r in rows
        ],
        "lllllrrrrrrrrrr",
    )
    out += ["", "`*` = the matcher AUTO selected. `–` = not measured or not applicable. Strip support = share of the 20 stretches along the strip holding at least 3 inliers; under 50 % a result cannot be SUCCESS.", ""]
    notes = read("matcher_benchmark_notes.json")
    if notes:
        out += ["### Notes on this run", "", *[f"* {note}" for note in notes["notes"]], ""]

    failures = [r for r in rows if r["registration_status"] in ("FAILED", "UNAVAILABLE")]
    if failures:
        out += ["### Why branches failed", ""]
        out += table(["Scene", "Matcher", "Reason"], [[short(r["scene_id"]), LABELS.get(r["matcher"], r["matcher"]), r["reasons"] or "–"] for r in failures])
        out.append("")

    warned = [r for r in rows if r["registration_status"] == "LOW_CONFIDENCE"]
    if warned:
        out += ["### Why results are LOW_CONFIDENCE", ""]
        out += table(["Scene", "Matcher", "Reason"], [[short(r["scene_id"]), LABELS.get(r["matcher"], r["matcher"]), r["reasons"] or "–"] for r in warned])
        out.append("")

    refined = [r for r in rows if r.get("refined_points")]
    if refined:
        out += [
            "### Tie points and refinement",
            "",
            "Refinement figures here are movements and agreement, not accuracy. Selected tie points come only from verified inliers.",
            "",
        ]
        out += table(
            ["Scene", "Matcher", "Selected tie points", "Distribution score", "Refined points", "Median shift px", "Improved held-out agreement"],
            [
                [
                    short(r["scene_id"]),
                    LABELS.get(r["matcher"], r["matcher"]),
                    n(r["selected_tie_points"]),
                    n(r["distribution_score"]),
                    n(r["refined_points"]),
                    n(r["median_refinement_px"]),
                    {True: "yes", False: "no", None: "–"}.get(r.get("refinement_improved_heldout")),
                ]
                for r in refined
            ],
            "llrrrrl",
        )
        out.append("")
    return out


def representation_section(data: dict) -> list[str]:
    modes = data["modes"]
    out = ["## Image representations", "", f"Generated {data.get('generated_at')} by `scripts/benchmark_representations.py`. {data.get('label')}.", ""]
    out += table(
        ["Scene", "Matcher", *modes],
        [[short(row["scene_id"]), LABELS.get(row["matcher"], row["matcher"]), *[n(row["values"].get(m)) for m in modes]] for row in data["table"]],
        "ll" + "r" * len(modes),
    )
    out += ["", *[f"* {note}" for note in data.get("notes", [])], ""]
    return out


def synthetic_section(data: dict) -> list[str]:
    out = [
        "## Synthetic ground truth",
        "",
        f"**SYNTHETIC: {data['label']}.** Generated {data.get('generated_at')} by `scripts/synthetic_validation.py`. "
        "These figures show that the machinery recovers known geometry. They do not predict accuracy on real scenes.",
        "",
    ]
    out += table(
        ["Case", "Matcher", "Status", "Inliers", "Error vs truth, RMSE / max px", "Tie-point median px, initial → refined", "Refined points"],
        [
            [
                r["case"],
                LABELS.get(r["matcher"], r["matcher"]),
                r["status"],
                n(r["inliers"]),
                f"{n(r['truth_rmse_px'])} / {n(r['truth_max_px'])}",
                f"{n(r['tie_error_initial_median_px'], 3)} → {n(r['tie_error_refined_median_px'], 3)}",
                n(r["refined_points"]),
            ]
            for r in data["rows"]
        ],
        "lllrrrr",
    )
    out.append("")
    return out


def refinement_section(data: dict) -> list[str]:
    rows = [r for r in data["rows"] if r.get("strategy")]
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["case"]].append(r)
    out = [
        "## Which sub-pixel estimator?",
        "",
        f"**SYNTHETIC: {data['label']}.** Generated {data.get('generated_at')} by `scripts/refinement_study.py`. Median error against the known "
        "truth, in pixels, before → after refinement, on the same points. `–` means fewer than 5 points passed the checks.",
        "",
    ]
    methods = list(dict.fromkeys((r["representation"], r["strategy"]) for r in rows))
    body = []
    for case, members in groups.items():
        lookup = {(r["representation"], r["strategy"]): r for r in members}
        cells = [case, n(members[0].get("starting_points"))]
        for method in methods:
            r = lookup.get(method)
            cells.append(f"{r['initial_median_px']:.2f} → {r['refined_median_px']:.2f}" if r and "refined_median_px" in r else "–")
        body.append(cells)
    out += table(["Case", "Points", *[f"{a} {b}" for a, b in methods]], body, "lr" + "r" * len(methods))
    out.append("")
    return out


def main() -> int:
    lines = [
        "# Benchmark results: matcher comparison",
        "",
        "This file is generated by `scripts/write_benchmark_report.py`; do not edit it by hand. The classical SIFT baseline and how to read it are in",
        "[benchmark_results.md](benchmark_results.md), and the reasoning behind the design in [engineering_findings.md](engineering_findings.md).",
        "",
    ]
    for name, section in (
        ("matcher_benchmark.json", matcher_section),
        ("representation_benchmark.json", representation_section),
        ("synthetic_validation.json", synthetic_section),
        ("refinement_study.json", refinement_section),
    ):
        data = read(name)
        if data:
            lines += section(data)
    path = PROJECT_ROOT / "docs" / "benchmark_matchers.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
