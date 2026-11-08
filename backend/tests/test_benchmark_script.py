"""scripts/benchmark_matchers.py must record a run that stopped, and go on to the next scene."""
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import benchmark_matchers as bm  # noqa: E402


def scene(key: str) -> SimpleNamespace:
    return SimpleNamespace(scene_id=f"OHRXXD18CHO{key}", sensor="OHRC")


def measured_row(scene_id: str, matcher: str = "SIFT", **fields) -> dict:
    row = dict.fromkeys(bm.BENCHMARK_COLUMNS + bm.EXTRA_COLUMNS)
    row.update(scene_id=scene_id, sensor="OHRC", matcher=matcher, registration_status="SUCCESS", geometric_inliers=100, reasons="")
    row.update(fields)
    return row


@pytest.fixture
def benchmark(monkeypatch, tmp_path):
    """The script with its scenes, engine and output folder replaced; ``register_scene`` is set by each test."""
    monkeypatch.setattr(bm.config, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(bm, "keep_awake", lambda: None)
    monkeypatch.setattr(bm, "matcher_statuses", lambda: [])
    monkeypatch.setattr(bm, "job_dir", lambda job_id: tmp_path / job_id)
    monkeypatch.setattr(bm, "save_result", lambda out, result: None)
    monkeypatch.setattr(bm, "benchmark_rows", lambda result: [measured_row(result.scene_id)])
    monkeypatch.setattr(bm, "discover_scenes", lambda root: SimpleNamespace(scenes=[scene("1111111"), scene("2222222")]))
    monkeypatch.setattr(sys, "argv", ["benchmark_matchers.py"])
    return tmp_path


def rows_of(folder: Path) -> list[dict]:
    return json.loads((folder / "matcher_benchmark.json").read_text(encoding="utf-8"))["rows"]


def finished(scene_record, *args, **kwargs) -> SimpleNamespace:
    return SimpleNamespace(scene_id=scene_record.scene_id, status="SUCCESS", selected_matcher="SIFT")


def test_a_run_that_raises_is_recorded_and_the_next_scene_still_runs(benchmark, monkeypatch):
    def engine(scene_record, *args, **kwargs):
        if "1111111" in scene_record.scene_id:
            raise MemoryError("out of memory in the test")
        return finished(scene_record)

    monkeypatch.setattr(bm, "register_scene", engine)
    assert bm.main() == 0

    stopped, measured = rows_of(benchmark)
    assert stopped["scene_id"].endswith("1111111") and stopped["registration_status"] == "FAILED"
    assert "MemoryError" in stopped["reasons"] and "out of memory in the test" in stopped["reasons"]
    assert measured["scene_id"].endswith("2222222") and measured["geometric_inliers"] == 100


def test_a_run_that_raises_reports_no_figures(benchmark, monkeypatch):
    def engine(scene_record, *args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(bm, "register_scene", engine)
    bm.main()

    figures = set(bm.BENCHMARK_COLUMNS) - {"scene_id", "sensor", "matcher", "registration_status", "processing_seconds", "reasons"}
    for row in rows_of(benchmark):
        assert all(row[name] is None for name in figures), row
        assert isinstance(row["processing_seconds"], float)
    with (benchmark / "matcher_benchmark.csv").open(encoding="utf-8", newline="") as fh:
        table = list(csv.DictReader(fh))
    assert len(table) == 2 and all(row["geometric_inliers"] == "" for row in table)
    assert "failed" in (benchmark / "benchmark_charts" / "inlier_ratio.svg").read_text(encoding="utf-8")


def test_a_later_run_replaces_the_stopped_row(benchmark, monkeypatch):
    def engine(scene_record, *args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(bm, "register_scene", engine)
    bm.main()
    assert {r["matcher"] for r in rows_of(benchmark)} == {"AUTO"}

    monkeypatch.setattr(bm, "register_scene", finished)
    monkeypatch.setattr(sys, "argv", ["benchmark_matchers.py", "--keep", "--scene", "1111111"])
    bm.main()
    rows = {(r["scene_id"][-7:], r["matcher"]): r["registration_status"] for r in rows_of(benchmark)}
    assert rows == {("1111111", "SIFT"): "SUCCESS", ("2222222", "AUTO"): "FAILED"}


def test_rift_only_adds_its_rows_and_keeps_the_others(benchmark, monkeypatch):
    monkeypatch.setattr(bm, "register_scene", finished)
    bm.main()

    asked = []

    def rift(scene_record, cfg, *args, **kwargs):
        asked.append(cfg.matcher)
        return SimpleNamespace(scene_id=scene_record.scene_id, status="FAILED", selected_matcher="RIFT")

    monkeypatch.setattr(bm, "register_scene", rift)
    monkeypatch.setattr(bm, "benchmark_rows", lambda result: [measured_row(result.scene_id, "RIFT", registration_status="FAILED")])
    monkeypatch.setattr(sys, "argv", ["benchmark_matchers.py", "--rift-only"])
    bm.main()

    assert asked == ["RIFT", "RIFT"]
    rows = {(r["scene_id"][-7:], r["matcher"]): r for r in rows_of(benchmark)}
    assert set(rows) == {("1111111", "SIFT"), ("2222222", "SIFT"), ("1111111", "RIFT"), ("2222222", "RIFT")}
    assert rows[("1111111", "SIFT")]["registration_status"] == "SUCCESS"
    assert rows[("1111111", "RIFT")]["selected_by_auto"] is False
