"""Benchmark all official OHRC / LRO-NAC scenes -> results/ohrc_benchmark.csv

    .venv\\Scripts\\python.exe scripts\\benchmark_ohrc.py

Accepts the same pipeline options as run_registration.py (e.g. --ratio 0.8).
"""
import sys

import _bootstrap  # noqa: F401
import benchmark

if __name__ == "__main__":
    sys.argv[1:1] = ["--sensor", "OHRC"]
    benchmark.main()
