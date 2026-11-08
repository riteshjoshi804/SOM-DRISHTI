"""Grading, exports, reports, control points and the API, across matcher branches."""
from __future__ import annotations

import csv
import io
import json
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient

from app import config
from app.config import PipelineConfig
from app.evaluation.synthetic import SYNTHETIC_LABEL, build_cases, crater_terrain, render, truth_errors
from app.geometry.estimation import apply_transform
from app.ingestion.raster import RasterInfo, read_raster_info
from app.main import app
from app.matching import guided as guided_module
from app.matching.base import LOFTR, SIFT
from app.matching.models import ModelUnavailable, weights_present
from app.registration import control, engine, exports, store
from app.registration.engine import BranchOutcome, _apply_decoy_rule, grade, register
from app.registration.schemas import DecoySummary, RegistrationResult
from app.registration.warp import smooth_maps, source_overlap_mask_smooth, warp_with_inverse, write_registered_geotiff

needs_loftr = pytest.mark.skipif(not weights_present("loftr"), reason="LoFTR weights not downloaded")


# ---------------------------------------------------------------------------- grading
def test_learned_matchers_face_stricter_limits() -> None:
    cfg = PipelineConfig()
    assert grade(cfg, 100, 100, 200, 20, None, 0.5, 1.0)[0] != "FAILED"  # the classical limit allows it
    status, fail, _ = grade(cfg, None, None, 200, 20, None, 0.5, 1.0, learned=True)
    assert status == "FAILED" and any("too few inliers" in f for f in fail)


def test_detector_free_matchers_have_no_keypoint_requirement() -> None:
    assert grade(PipelineConfig(), None, None, 500, 300, None, 0.6, 1.0, learned=True)[0] == "SUCCESS"


def test_invalid_geometry_fails_with_its_reason() -> None:
    status, fail, _ = grade(PipelineConfig(), None, None, 500, 300, None, 0.6, 1.0, learned=True, invalid_reasons=["implausible local scale"])
    assert status == "FAILED" and "implausible local scale" in fail[0]


def _outcome(inliers: int, status: str = "SUCCESS") -> BranchOutcome:
    class Geometry:
        inlier_idx = np.arange(inliers)

    from app.matching.base import failed_result

    return BranchOutcome(
        matcher=LOFTR, result=failed_result(LOFTR, "x"), geometry=Geometry(), status=status, failures=[], warnings=[],
        representation="clahe", source_pre=None, reference_pre=None, attempts=[], seconds=1.0,
    )  # fmt: skip


def test_result_must_stand_above_the_null_control() -> None:
    cfg = PipelineConfig()
    weak = _outcome(40)
    weak.decoys = [DecoySummary(kind="constant", tiles_used=5, inliers=20)]
    _apply_decoy_rule(weak, cfg)
    assert weak.status == "FAILED" and "null control" in weak.failures[0]

    strong = _outcome(400)
    strong.decoys = [DecoySummary(kind="constant", tiles_used=5, inliers=20)]
    _apply_decoy_rule(strong, cfg)
    assert strong.status == "SUCCESS"


def test_poor_prediction_of_unseen_ground_is_not_a_clean_success() -> None:
    """Regression: a result marked SUCCESS was visibly off by about 20 px away from its tie points."""
    from app.evaluation.validation import HoldoutResult

    cfg = PipelineConfig()
    outcome = _outcome(600)
    outcome.holdout = HoldoutResult(status="ok", block_rmse_px=9.0)
    engine._apply_unseen_ground_rule(outcome, cfg)
    assert outcome.status == "LOW_CONFIDENCE" and "left out of the fit" in outcome.warnings[0]
    fine = _outcome(600)
    fine.holdout = HoldoutResult(status="ok", block_rmse_px=2.0)
    engine._apply_unseen_ground_rule(fine, cfg)
    assert fine.status == "SUCCESS" and not fine.warnings


def _support_outcome(positions: list[float], axis: int = 1) -> BranchOutcome:
    outcome = _outcome(len(positions))
    outcome.geometry.strip = None
    points = np.full((len(positions), 2), 10.0)
    points[:, axis] = positions
    outcome.result.source_points = points
    return outcome


STRIP = SimpleNamespace(info=SimpleNamespace(width=100, height=2000))  # each twentieth is 100 px long
NINE_STRETCHES = [100.0 * stretch + 50.0 for stretch in range(9) for _ in range(3)]


def test_a_stray_tie_point_is_not_support_along_the_strip() -> None:
    """Regression: one point in a stretch decided SUCCESS against LOW_CONFIDENCE, and with it AUTO's choice."""
    cfg = PipelineConfig()
    short = _support_outcome(NINE_STRETCHES)
    engine._apply_support_rule(short, STRIP, cfg)
    assert short.status == "LOW_CONFIDENCE" and "45%" in short.warnings[0]

    stray = _support_outcome(NINE_STRETCHES + [950.0])
    engine._apply_support_rule(stray, STRIP, cfg)
    assert stray.status == "LOW_CONFIDENCE" and "45%" in stray.warnings[0]

    enough = _support_outcome(NINE_STRETCHES + [950.0] * cfg.validation.along_track_support_min_points)
    engine._apply_support_rule(enough, STRIP, cfg)
    assert enough.status == "SUCCESS" and not enough.warnings


def test_support_is_measured_along_the_long_side_for_a_model_without_a_strip() -> None:
    cfg = PipelineConfig()
    wide = SimpleNamespace(info=SimpleNamespace(width=2000, height=100))
    outcome = _support_outcome(NINE_STRETCHES, axis=0)
    assert engine._along_track_support(outcome, wide, cfg) == pytest.approx(0.45)
    # The same points on an upright strip all share one row, so they support a single stretch.
    assert engine._along_track_support(outcome, STRIP, cfg) == pytest.approx(0.05)


def test_the_support_rule_never_touches_a_failed_result() -> None:
    outcome = _support_outcome(NINE_STRETCHES)
    outcome.status = "FAILED"
    engine._apply_support_rule(outcome, STRIP, PipelineConfig())
    assert outcome.status == "FAILED" and not outcome.warnings


def test_sift_faces_the_support_rule_where_results_are_compared_but_not_as_the_baseline(synthetic_case, monkeypatch) -> None:
    def refuse(name, cfg):
        raise ModelUnavailable("weights are missing")

    monkeypatch.setattr(engine, "create_pair_matcher", refuse)
    monkeypatch.setattr(engine, "_along_track_support", lambda o, source, cfg: 0.1 if o.geometry is not None else None)
    monkeypatch.setattr(engine, "_apply_unseen_ground_rule", lambda o, cfg: None)

    baseline = register(synthetic_case.source, synthetic_case.reference, PipelineConfig(matcher=SIFT, preprocessing_modes=["clahe"]))
    assert not any("support only" in w for w in baseline.warnings)

    compared = register(synthetic_case.source, synthetic_case.reference, PipelineConfig(matcher="AUTO", preprocessing_modes=["clahe"]))
    assert compared.selected_matcher == SIFT and compared.status == "LOW_CONFIDENCE"
    assert any("support only 10%" in w for w in compared.warnings)


def test_unmeasured_null_is_never_a_silent_pass() -> None:
    cfg = PipelineConfig()
    outcome = _outcome(400)
    outcome.decoys = [DecoySummary(kind="scattered", tiles_used=0)]
    _apply_decoy_rule(outcome, cfg)
    assert outcome.status == "LOW_CONFIDENCE" and "null level" in outcome.warnings[0]


# ---------------------------------------------------------------------------- warping and outputs
def test_smooth_maps_equal_the_exact_mapping_for_a_linear_one() -> None:
    forward = lambda p: p * np.array([1.5, 0.5]) + np.array([3.0, -2.0])
    map_x, map_y = smooth_maps(forward, 40, 30, x0=10, y0=5, step=8)
    assert map_x.shape == (30, 40)
    assert map_x[0, 0] == pytest.approx(10 * 1.5 + 3.0) and map_y[29, 39] == pytest.approx(34 * 0.5 - 2.0)
    assert map_x[17, 23] == pytest.approx(33 * 1.5 + 3.0, abs=1e-3)


def test_warp_with_inverse_recovers_a_translation() -> None:
    image = np.zeros((100, 100), np.uint8)
    image[40, 60] = 200
    out = warp_with_inverse(image, lambda p: p - np.array([10.0, 5.0]), (120, 120), (0, 0, 120, 120))
    assert out[45, 70] == 200 and out.shape == (120, 120)


def test_overlap_mask_for_a_smooth_model() -> None:
    source = np.ones((50, 50), bool)
    reference = np.zeros((100, 100), bool)
    reference[:, :60] = True
    mask = source_overlap_mask_smooth(source, reference, lambda p: p + np.array([40.0, 0.0]))
    assert mask[:, :15].all() and not mask[:, 25:].any()


def _info(tmp_path: Path, crs: str | None, consistent: bool | None) -> RasterInfo:
    return RasterInfo(
        path=str(tmp_path / "r.tif"), width=64, height=64, band_count=1, dtype="uint8", nodata=0, crs=crs,
        crs_is_geographic=False, transform=[200.0, 0.0, 0.0, 0.0, -200.0, 0.0], has_geotransform=True,
        pixel_size=[200.0, 200.0], bounds=None, gcps=[], gcp_crs=None, crs_label_consistent=consistent, tags={},
    )  # fmt: skip


LUNAR = 'GEOGCS["Moon",DATUM["D_Moon",SPHEROID["Moon",1737400,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'


def test_geotiff_never_copies_a_crs_label_that_contradicts_its_grid(tmp_path) -> None:
    data = np.full((64, 64), 7, np.uint8)
    matrix = np.eye(3)
    notes = write_registered_geotiff(tmp_path / "bad.tif", data, np.ones((64, 64), bool), matrix, _info(tmp_path, LUNAR, False))
    with rasterio.open(tmp_path / "bad.tif") as ds:
        assert ds.crs is None or not ds.crs.is_valid
        assert ds.transform.a == 200.0
    assert any("no CRS" in n for n in notes)
    assert write_registered_geotiff(tmp_path / "ok.tif", data, np.ones((64, 64), bool), matrix, _info(tmp_path, LUNAR, True)) == []
    with rasterio.open(tmp_path / "ok.tif") as ds:
        assert ds.crs is not None


# ---------------------------------------------------------------------------- synthetic ground truth
def test_rendering_moves_shadows_with_the_sun() -> None:
    terrain = crater_terrain((300, 300), 1)
    east, west = render(terrain, 0.0, 15.0), render(terrain, 180.0, 15.0)
    assert east.dtype == np.float32 and 0.0 <= east.min() and east.max() <= 1.0
    assert (east < 0.01).mean() > 0.005  # there are real shadows
    assert np.corrcoef(east.ravel(), west.ravel())[0, 1] < 0.6  # and they really change


def test_synthetic_cases_cover_what_the_brief_lists() -> None:
    cases = build_cases(source_size=(120, 240), reference_size=(420, 420))
    groups = {c.group for c in cases}
    assert {"rotation", "scale", "translation", "perspective", "noise", "blur", "illumination", "partial_overlap", "no_prior"} <= groups
    assert all(SYNTHETIC_LABEL in c.source.info.tags["SYNTHETIC"] for c in cases)
    partial = next(c for c in cases if c.group == "partial_overlap")
    assert 0.2 < 1.0 - float(partial.source.valid_mask.mean()) < 0.7
    assert next(c for c in cases if c.group == "no_prior").prior is None


def test_truth_errors_measure_only_inside_the_reference() -> None:
    truth = np.eye(3)
    valid = np.ones((100, 100), bool)
    report = truth_errors(lambda p: p + np.array([3.0, 4.0]), truth, valid, (100, 100))
    assert report["rmse"] == pytest.approx(5.0) and report["max"] == pytest.approx(5.0)
    assert truth_errors(lambda p: p, truth, valid, (0, 0))["rmse"] is None


# ---------------------------------------------------------------------------- the engine end to end
@pytest.fixture(scope="module")
def synthetic_case():
    cases = build_cases(source_size=(260, 620), reference_size=(1100, 1100))
    return next(c for c in cases if c.name == "baseline")


def test_sift_run_writes_every_deliverable(synthetic_case, tmp_path) -> None:
    cfg = PipelineConfig(preprocessing_modes=["clahe"])
    result = register(synthetic_case.source, synthetic_case.reference, cfg, out_dir=tmp_path)
    assert result.status != "FAILED", result.failure_reasons
    assert result.selected_matcher == SIFT and result.requested_matcher == SIFT
    for name in (
        "correspondences_raw.csv", "correspondences_inliers.csv", "tie_points_selected.csv", "tie_points_refined.csv",
        "transformation.json", "metadata_summary.json", "metrics.json", "benchmark_row.csv", "report.md", "report.html",
        "registered_source.tif", "registered_source.png", "reference.jpg", "overlay.png", "checkerboard.jpg",
        "tie_points.jpg", "reference_crop.jpg", "registered_crop.png",
    ):  # fmt: skip
        assert (tmp_path / name).is_file(), name
        assert name in result.artifacts.values() or name == "reference.jpg"

    truth = truth_errors(lambda p: apply_transform(np.array(result.transformation.matrix), p), synthetic_case.truth, synthetic_case.source.valid_mask, (1100, 1100))
    assert truth["rmse"] < 2.0

    # Each table holds what its name says, in original pixels.
    with (tmp_path / "tie_points_refined.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == result.metrics.inliers
    assert {"reference_x_initial", "reference_x_refined", "displacement_x", "refined", "outcome"} <= set(rows[0])
    refined = [r for r in rows if r["refined"] == "1"]
    assert refined and any(float(r["displacement_x"]) != 0.0 for r in refined)
    with (tmp_path / "tie_points_selected.csv").open(encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == result.tie_points.selected <= result.metrics.inliers


def test_the_report_keeps_the_kinds_of_evidence_apart(synthetic_case, tmp_path) -> None:
    result = register(synthetic_case.source, synthetic_case.reference, PipelineConfig(preprocessing_modes=["clahe"]), out_dir=tmp_path)
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    for heading in ("A. Geometric consistency", "B. Held-out validation", "C. Independent control-point accuracy"):
        assert heading in text
    assert "no independent control points have been supplied" in text and "not accuracy" in text
    assert result.evidence.has_independent_ground_truth is False
    assert result.holdout is not None and "not independent ground truth" in result.holdout.label
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>") and "<script" not in html


def test_failed_run_produces_a_report_and_no_registered_image(tmp_path) -> None:
    cases = build_cases(source_size=(200, 400), reference_size=(800, 800))
    other = build_cases(seed=4242, source_size=(200, 400), reference_size=(800, 800))
    result = register(other[0].source, cases[0].reference, PipelineConfig(preprocessing_modes=["clahe"]), out_dir=tmp_path)
    assert result.status == "FAILED" and result.transformation is None
    assert (tmp_path / "report.md").is_file() and not (tmp_path / "registered_source.tif").exists()
    assert "registered_source" not in result.artifacts


def test_a_requested_matcher_that_cannot_run_is_reported_not_replaced(synthetic_case, monkeypatch) -> None:
    def refuse(name, cfg):
        raise ModelUnavailable("weights are missing")

    monkeypatch.setattr(engine, "create_pair_matcher", refuse)
    result = register(synthetic_case.source, synthetic_case.reference, PipelineConfig(matcher=LOFTR))
    assert result.status == "FAILED" and result.selected_matcher is None
    assert "unavailable" in result.failure_reasons[0] and result.metrics.matcher == LOFTR
    assert result.matcher_results[0].availability == "UNAVAILABLE"
    assert result.transformation is None


def test_auto_falls_back_when_a_matcher_is_unavailable(synthetic_case, monkeypatch) -> None:
    def refuse(name, cfg):
        raise ModelUnavailable("weights are missing")

    monkeypatch.setattr(engine, "create_pair_matcher", refuse)
    cfg = PipelineConfig(matcher="AUTO", preprocessing_modes=["clahe"])
    result = register(synthetic_case.source, synthetic_case.reference, cfg)
    assert result.selected_matcher == SIFT and result.selection.selected == SIFT
    availability = {m.matcher: m.availability for m in result.matcher_results}
    assert availability[SIFT] == "OK" and "UNAVAILABLE" in availability.values()


def test_benchmark_rows_mark_unavailable_matchers(synthetic_case, monkeypatch) -> None:
    def refuse(name, cfg):
        raise ModelUnavailable("weights are missing")

    monkeypatch.setattr(engine, "create_pair_matcher", refuse)
    result = register(synthetic_case.source, synthetic_case.reference, PipelineConfig(matcher="AUTO", preprocessing_modes=["clahe"]))
    rows = {r["matcher"]: r for r in exports.benchmark_rows(result)}
    assert rows[SIFT]["registration_status"] != "UNAVAILABLE"
    assert any(r["registration_status"] == "UNAVAILABLE" for r in rows.values())
    assert set(rows[SIFT]) == set(exports.BENCHMARK_COLUMNS)


@needs_loftr
def test_learned_branch_end_to_end(tmp_path) -> None:
    cases = build_cases(source_size=(300, 800), reference_size=(1200, 1200))
    case = next(c for c in cases if c.name == "baseline")
    cfg = PipelineConfig(matcher=LOFTR)
    cfg.guided.fine_tile_px = 320
    cfg.guided.anchor_tile_px = 256
    cfg.guided.anchor_scale = 1.0
    cfg.guided.decoy_controls = 1
    cfg.guided.use_strip_model = False
    cfg.validation.learned_min_inliers = 20
    result = register(case.source, case.reference, cfg, out_dir=tmp_path, overlap=case.prior)
    assert result.selected_matcher == LOFTR, (result.failure_reasons, result.search_trace)
    assert result.status != "FAILED" and result.decoys and result.metrics.decoy_max_inliers is not None
    truth = truth_errors(lambda p: apply_transform(np.array(result.transformation.matrix), p), case.truth, case.source.valid_mask, (1200, 1200))
    assert truth["rmse"] < 3.0, truth
    assert result.metrics.source_keypoints is None  # detector-free
    assert result.metrics.heldout_median_px is not None and result.tie_points.selected > 0
    assert (tmp_path / "tie_points_refined.csv").is_file() and (tmp_path / "overlay.png").is_file()
    assert not (tmp_path / "keypoints_source.jpg").exists()


def _unrelated_pair() -> tuple:
    """A source and a reference of different terrain, with a prior that claims they overlap."""
    size = {"source_size": (300, 800), "reference_size": (1200, 1200)}
    case = next(c for c in build_cases(**size) if c.name == "baseline")
    other = next(c for c in build_cases(seed=99, **size) if c.name == "baseline")
    cfg = PipelineConfig(matcher=LOFTR)
    cfg.guided.fine_tile_px = 320
    cfg.guided.anchor_tile_px = 256
    cfg.guided.anchor_scale = 1.0
    cfg.guided.decoy_controls = 1
    return case, other, cfg


@needs_loftr
def test_unrelated_imagery_is_rejected_by_the_full_pipeline() -> None:
    case, other, cfg = _unrelated_pair()
    result = register(case.source, other.reference, cfg, overlap=case.prior)
    assert result.status == "FAILED", (result.metrics.inliers, result.decoys)
    assert result.selected_matcher is None and result.transformation is None
    assert result.failure_reasons


@needs_loftr
def test_a_fooled_anchor_sweep_does_not_become_a_registration(monkeypatch) -> None:
    """The later gates must hold on their own, for the day the sweep accepts a look-alike."""

    def fooled(matcher, source, reference, hypotheses, radius, guided, say):
        centre = (np.array(source.size, dtype=np.float64) - 1.0) / 2.0
        return guided_module.Anchor(hypotheses[0], np.zeros(2), centre, 20, 2, 1, 1, 0.0, []), [], guided.anchor_scale

    monkeypatch.setattr(guided_module, "sweep_anchor", fooled)
    case, other, cfg = _unrelated_pair()
    result = register(case.source, other.reference, cfg, overlap=case.prior)
    assert result.search_trace["anchor"]["votes"] == 20  # the search really went on to the tile passes
    assert result.search_trace["passes"]
    assert result.status == "FAILED", (result.metrics.inliers, result.decoys)
    assert result.selected_matcher is None and result.transformation is None


# ---------------------------------------------------------------------------- API
@pytest.fixture
def client(synthetic_dataset: Path, tmp_path: Path, monkeypatch) -> TestClient:
    monkeypatch.setattr(config, "DATA_ROOT", synthetic_dataset)
    monkeypatch.setattr(config, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(config, "JOBS_ROOT", tmp_path / "results" / "jobs")
    return TestClient(app)


def _finished_job(client: TestClient) -> str:
    started = client.post("/api/register/OHRXXD18TEST_V2_1", json={"config": {"preprocessing_modes": ["clahe"]}})
    assert started.status_code == 202
    job_id = started.json()["job_id"]
    deadline = time.time() + 120
    while time.time() < deadline:
        body = client.get(f"/api/results/{job_id}").json()
        if body["job"]["state"] in ("done", "error"):
            break
        time.sleep(0.3)
    assert body["job"]["state"] == "done", body["job"]
    return job_id


def test_scene_detail_reports_the_estimated_overlap(client: TestClient) -> None:
    overlap = client.get("/api/scenes/OHRXXD18TEST_V2_1").json()["overlap_estimate"]
    assert overlap["mode"] in ("metadata", "full_image") and overlap["reason"]
    assert overlap["coordinate_system"]["body"] == "Moon"
    assert overlap["search_bbox_px"] is not None


def test_matcher_status_endpoint(client: TestClient) -> None:
    statuses = {s["name"]: s for s in client.get("/api/matchers").json()}
    assert statuses[SIFT]["available"] is True and statuses["RIFT"]["experimental"] is True
    assert "filtering" in statuses[LOFTR] and "weights" in statuses[LOFTR]


def test_requesting_an_unavailable_matcher_is_refused(client: TestClient, monkeypatch) -> None:
    from app.api import routes
    from app.matching.base import MatcherStatus

    monkeypatch.setattr(
        routes,
        "matcher_statuses",
        lambda cfg=None: [MatcherStatus(name=LOFTR, label="LoFTR", available=False, reason="no weights", filtering="x", implementation="y")],
    )
    response = client.post("/api/register/OHRXXD18TEST_V2_1", json={"config": {"matcher": LOFTR}})
    assert response.status_code == 409 and "no weights" in response.json()["detail"]


def test_bad_matcher_name_is_a_validation_error(client: TestClient) -> None:
    assert client.post("/api/register/OHRXXD18TEST_V2_1", json={"config": {"matcher": "NOPE"}}).status_code == 422


def test_benchmarks_are_empty_until_a_script_has_run(client: TestClient) -> None:
    assert client.get("/api/benchmarks").json() == {"matchers": None, "notes": None, "representations": None, "synthetic": None}


def test_downloads_and_new_artifact_types(client: TestClient) -> None:
    job_id = _finished_job(client)
    result = client.get(f"/api/results/{job_id}").json()["result"]
    assert result["selected_matcher"] == SIFT and result["holdout"] and result["tie_points"]
    for name in ("report.html", "report.md", "metrics.json", "tie_points_refined.csv"):
        assert client.get(f"/api/results/{job_id}/artifacts/{name}").status_code == 200, name
    assert client.get(f"/api/results/{job_id}/artifacts/..%2F..%2Fsecret.csv").status_code == 404
    archive = client.get(f"/api/results/{job_id}/download")
    assert archive.status_code == 200 and archive.headers["content-type"] == "application/zip"
    names = zipfile.ZipFile(io.BytesIO(archive.content)).namelist()
    assert {"result.json", "report.html", "tie_points_refined.csv", "transformation.json"} <= set(names)


def test_control_points_measure_without_changing_the_transformation(client: TestClient) -> None:
    job_id = _finished_job(client)
    before = client.get(f"/api/results/{job_id}").json()["result"]
    matrix = np.array(before["transformation"]["matrix"])
    src = np.array([[20.0, 30.0], [100.0, 150.0], [60.0, 90.0], [30.0, 170.0], [120.0, 40.0], [80.0, 120.0]])
    ref = apply_transform(matrix, src) + np.array([3.0, 4.0])
    points = [{"source_x": s[0], "source_y": s[1], "reference_x": r[0], "reference_y": r[1]} for s, r in zip(src, ref)]
    response = client.post(f"/api/results/{job_id}/control-points", json={"provenance": "picked by a test", "points": points})
    assert response.status_code == 200, response.text
    assert response.json()["rmse_px"] == pytest.approx(5.0) and response.json()["count"] == 6

    after = client.get(f"/api/results/{job_id}").json()["result"]
    assert after["transformation"]["matrix"] == before["transformation"]["matrix"]
    assert after["metrics"]["inliers"] == before["metrics"]["inliers"]
    assert after["metrics"]["control_point_rmse_px"] == pytest.approx(5.0)
    assert after["evidence"]["has_independent_ground_truth"] is True
    report = client.get(f"/api/results/{job_id}/artifacts/report.md").text
    assert "picked by a test" in report and "no independent control points" not in report


def test_chips_show_original_pixels_at_a_known_magnification(client: TestClient) -> None:
    import cv2

    job_id = _finished_job(client)
    url = f"/api/results/{job_id}/chip"
    plain = client.get(url, params={"which": "reference", "x": 200, "y": 150, "size": 64, "zoom": 2})
    assert plain.status_code == 200 and plain.headers["content-type"] == "image/png"
    chip = cv2.imdecode(np.frombuffer(plain.content, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert chip.shape == (64, 64) and chip.std() > 1.0
    # Far outside the raster there is nothing to show, and that is not an error.
    empty = client.get(url, params={"which": "reference", "x": 90000, "y": 90000, "size": 64})
    assert empty.status_code == 200
    assert cv2.imdecode(np.frombuffer(empty.content, np.uint8), cv2.IMREAD_GRAYSCALE).max() == 0
    assert client.get(url, params={"which": "source", "x": 50, "y": 50, "rectify": "true"}).status_code == 200
    assert client.get(url, params={"which": "nope", "x": 1, "y": 1}).status_code == 422
    assert client.get("/api/results/does-not-exist/chip", params={"which": "source", "x": 1, "y": 1}).status_code == 404


def test_chip_mapping_matches_what_the_picker_assumes(tmp_path) -> None:
    """Pixel (u, v) of a chip must show original pixel c + (u - (size - 1) / 2) / zoom."""
    import cv2

    from conftest import write_tif

    data = np.zeros((200, 200), np.uint8)
    data[60:140, 60:140] = 40
    data[100, 120] = 255  # one bright original pixel
    path = write_tif(tmp_path / "bright.tif", data)
    size, zoom, cx, cy = 65, 4, 118.0, 97.0
    png = control.render_chip(path, cx, cy, size, np.eye(2) * zoom)
    chip = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
    v, u = np.unravel_index(int(chip.argmax()), chip.shape)
    assert cx + (u - (size - 1) / 2) / zoom == pytest.approx(120.0, abs=0.3)
    assert cy + (v - (size - 1) / 2) / zoom == pytest.approx(100.0, abs=0.3)


def test_control_points_need_provenance_and_valid_input(client: TestClient) -> None:
    job_id = _finished_job(client)
    csv_text = "source_x,source_y,reference_x,reference_y\n1,2,3,4\n"
    assert client.post(f"/api/results/{job_id}/control-points", json={"provenance": "  ", "csv": csv_text}).status_code == 400
    assert client.post(f"/api/results/{job_id}/control-points", json={"provenance": "x", "csv": "a,b\n1,2\n"}).status_code == 400
    assert client.post(f"/api/results/{job_id}/control-points", json={"provenance": "csv file", "csv": csv_text}).status_code == 200
    assert client.post("/api/results/does-not-exist/control-points", json={"provenance": "x", "csv": csv_text}).status_code == 404


def test_results_from_earlier_versions_still_load() -> None:
    old = {
        "scene_id": "S", "sensor": "OHRC", "reference_sensor": "LRO-NAC", "status": "FAILED", "failure_reasons": ["x"], "warnings": [],
        "metrics": {"registration_status": "FAILED"}, "selected_preprocessing": None, "transformation": None, "spatial_coverage": None,
        "metadata_consistency": None, "predicted_footprint": None, "attempts": [],
        "source": {"path": "s", "width": 1, "height": 1, "dtype": "uint8", "nodata": None, "has_georeferencing": False, "stats": _stats()},
        "reference": {"path": "r", "width": 1, "height": 1, "dtype": "uint8", "nodata": None, "has_georeferencing": False, "stats": _stats()},
        "metadata": None, "artifacts": {}, "config": {}, "notes": [],
    }  # fmt: skip
    result = RegistrationResult.model_validate(old)
    assert result.selected_matcher is None and result.matcher_results == [] and control.mapping_of(result) is None
    assert json.loads(result.model_dump_json())["holdout"] is None


def _stats() -> dict:
    return {
        "valid_pixels": 1, "valid_fraction": 1.0, "nan_count": 0, "inf_count": 0, "zero_count": 0, "negative_count": 0, "minimum": 1.0,
        "maximum": 1.0, "mean": 1.0, "std": 0.0, "percentiles": {}, "histogram_counts": [], "histogram_edges": [],
    }  # fmt: skip


def test_read_raster_info_stays_the_entry_point(tmp_path) -> None:
    assert store.is_valid_job_id("a") and read_raster_info is not None
