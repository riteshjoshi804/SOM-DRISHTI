"""Attempt all official IIRS / LRO-WAC scenes -> results/iirs_benchmark.csv

    .venv\\Scripts\\python.exe scripts\\benchmark_iirs.py

Every representation (raw_normalized, clahe, gradient) is evaluated so that
results/iirs_benchmark_attempts.csv shows which one gives the strongest valid
geometry - including when none of them does.
"""
import sys

import _bootstrap  # noqa: F401
import benchmark

if __name__ == "__main__":
    sys.argv[1:1] = ["--sensor", "IIRS", "--all-modes", "--modes", "raw_normalized,clahe,gradient"]
    benchmark.main()
