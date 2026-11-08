"""Compare a fresh SIFT benchmark with the stored baseline CSVs, value by value.

    .venv\\Scripts\\python.exe scripts\\benchmark.py --sensor OHRC --out results\\regression\\ohrc_benchmark.csv --no-geotiff
    .venv\\Scripts\\python.exe scripts\\benchmark_iirs.py --out results\\regression\\iirs_benchmark.csv --no-geotiff
    .venv\\Scripts\\python.exe scripts\\check_sift_regression.py

The classical pipeline is the project's fixed point of comparison, so everything that
describes its geometry must stay identical. Processing time and job IDs naturally differ.
Exits with status 1 if anything else changed.
"""
from __future__ import annotations

import csv
import sys

import _bootstrap  # noqa: F401

from app.config import RESULTS_ROOT

IGNORED = {"processing_time_seconds", "job_id"}


def read(path) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["scene_id"]: row for row in csv.DictReader(fh)}


def main() -> int:
    differences = 0
    compared = 0
    for name in ("ohrc_benchmark.csv", "iirs_benchmark.csv"):
        baseline = read(RESULTS_ROOT / name)
        fresh_path = RESULTS_ROOT / "regression" / name
        if not fresh_path.is_file():
            print(f"MISSING {fresh_path}")
            differences += 1
            continue
        fresh = read(fresh_path)
        for scene, old in baseline.items():
            new = fresh.get(scene)
            if new is None:
                print(f"MISSING scene {scene} in {fresh_path}")
                differences += 1
                continue
            changed = [k for k in old if k not in IGNORED and old[k] != new.get(k)]
            compared += 1
            status = "identical" if not changed else "CHANGED"
            print(f"{scene}: {status} ({old['status']}, {old['inliers']} inliers)")
            for key in changed:
                print(f"    {key}: {old[key]!r} -> {new.get(key)!r}")
            differences += len(changed)
    print(f"\n{compared} scenes compared, {differences} differences")
    return 1 if differences else 0


if __name__ == "__main__":
    sys.exit(main())
