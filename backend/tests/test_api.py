"""HTTP API against a small temporary dataset."""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import config
from app.main import app


@pytest.fixture
def client(synthetic_dataset: Path, tmp_path: Path, monkeypatch) -> TestClient:
    monkeypatch.setattr(config, "DATA_ROOT", synthetic_dataset)
    monkeypatch.setattr(config, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(config, "JOBS_ROOT", tmp_path / "results" / "jobs")
    return TestClient(app)


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["sift_available"] is True
    assert body["scene_count"] == 2


def test_scenes_and_detail(client: TestClient) -> None:
    scenes = client.get("/api/scenes").json()
    assert [s["scene_id"] for s in scenes] == ["OHRXXD18TEST_V2_1", "OHRXXD18TEST_V2_1_01"]
    assert scenes[0]["native_resolution_m"] == 0.25 and scenes[0]["sun_elevation_deg"] == 3.25
    detail = client.get("/api/scenes/OHRXXD18TEST_V2_1").json()
    assert detail["metadata"]["corners"]["top_left"]["latitude"] == -85.1
    assert detail["source_info"]["width"] == 140
    assert client.get("/api/scenes/NOPE").status_code == 404
    assert client.get("/api/scenes/OHRXXD18TEST_V2_1/preview/source").headers["content-type"] == "image/jpeg"
    dataset = client.get("/api/dataset").json()
    assert dataset["tmc2_present"] is False and dataset["sensor_counts"] == {"OHRC": 2}


def test_register_poll_metrics_and_artifacts(client: TestClient) -> None:
    started = client.post("/api/register/OHRXXD18TEST_V2_1", json={"config": {"preprocessing_modes": ["clahe"]}})
    assert started.status_code == 202
    job_id = started.json()["job_id"]
    deadline = time.time() + 120
    while True:
        body = client.get(f"/api/results/{job_id}").json()
        if body["job"]["state"] in ("done", "error") or time.time() > deadline:
            break
        time.sleep(0.5)
    assert body["job"]["state"] == "done", body["job"]
    result = body["result"]
    assert result["status"] == "SUCCESS"
    metrics = client.get(f"/api/results/{job_id}/metrics").json()
    assert metrics["registration_status"] == "SUCCESS"
    assert metrics["metrics"]["inliers"] == result["metrics"]["inliers"] > 0
    assert client.get(f"/api/results/{job_id}/artifacts/overlay.png").status_code == 200
    assert client.get(f"/api/results/{job_id}/artifacts/..%2Fresult.json").status_code == 404
    assert any(j["job_id"] == job_id for j in client.get("/api/results").json())


def test_rejects_bad_ids_and_config(client: TestClient) -> None:
    assert client.get("/api/results/..").status_code == 404
    assert client.get("/api/results/does-not-exist").status_code == 404
    bad = client.post("/api/register/OHRXXD18TEST_V2_1", json={"config": {"match": {"ratio_threshold": 5}}})
    assert bad.status_code == 422
    assert client.post("/api/register/NOPE").status_code == 404
