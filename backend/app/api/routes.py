"""HTTP API. Every number returned comes from the dataset or from a registration run."""
from __future__ import annotations

import json
import re
import shutil
import threading
from pathlib import Path

import cv2
from fastapi import APIRouter, Body, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ValidationError

from app import config
from app.api.jobs import JobManager, JobState, merge_config
from app.ingestion.indexer import SceneRecord, discover_scenes
from app.ingestion.raster import RasterInfo, RasterStats, compute_stats, load_raster, read_raster_info
from app.metadata.georef import PredictedFootprint, predict_footprint
from app.metadata.xml_parser import SceneMetadata, parse_scene_xml
from app.preprocessing.pipeline import percentile_normalize
from app.registration import store
from app.registration.engine import register, register_scene
from app.visualization.render import fit_scale, save_image

router = APIRouter()
jobs = JobManager()

ARTIFACT_NAME = re.compile(r"^[A-Za-z0-9_.-]+\.(jpg|jpeg|png|tif|tiff|json)$")
_detail_cache: dict[str, dict] = {}
_detail_lock = threading.Lock()


class SceneSummary(BaseModel):
    scene_id: str
    sensor: str
    reference_sensor: str | None
    source_file: str
    reference_file: str
    xml_file: str
    source_size: list[int]
    reference_size: list[int]
    source_dtype: str
    reference_dtype: str
    native_resolution_m: float | None
    sun_azimuth_deg: float | None
    sun_elevation_deg: float | None
    date_of_pass: str | None
    area: str | None


class SceneDetail(BaseModel):
    scene: SceneRecord
    summary: SceneSummary
    metadata: SceneMetadata
    source_info: RasterInfo
    reference_info: RasterInfo
    source_stats: RasterStats
    reference_stats: RasterStats
    predicted_footprint: PredictedFootprint | None


def _index():
    return discover_scenes(config.DATA_ROOT)


def _scene_or_404(scene_id: str) -> SceneRecord:
    scene = _index().get(scene_id)
    if scene is None:
        raise HTTPException(status_code=404, detail=f"Unknown scene '{scene_id}'")
    return scene


def _summary(scene: SceneRecord, meta: SceneMetadata, src: RasterInfo, ref: RasterInfo) -> SceneSummary:
    return SceneSummary(
        scene_id=scene.scene_id,
        sensor=scene.sensor,
        reference_sensor=scene.reference_sensor,
        source_file=Path(scene.source_path).name,
        reference_file=Path(scene.reference_path).name,
        xml_file=Path(scene.xml_path).name,
        source_size=[src.width, src.height],
        reference_size=[ref.width, ref.height],
        source_dtype=src.dtype,
        reference_dtype=ref.dtype,
        native_resolution_m=meta.resolution_m,
        sun_azimuth_deg=meta.sun_azimuth_deg,
        sun_elevation_deg=meta.sun_elevation_deg,
        date_of_pass=meta.date_of_pass,
        area=meta.area,
    )


@router.get("/api/dataset")
def dataset() -> dict:
    index = _index()
    return {
        "data_root": index.data_root,
        "exists": index.exists,
        "scene_count": len(index.scenes),
        "sensor_counts": index.sensor_counts,
        "tmc2_present": bool(index.sensor_counts.get("TMC-2")),
        "incomplete": [i.model_dump() for i in index.incomplete],
        "unpaired_rasters": [Path(p).name for p in index.unpaired_rasters],
    }


@router.get("/api/scenes", response_model=list[SceneSummary])
def list_scenes() -> list[SceneSummary]:
    out = []
    for scene in _index().scenes:
        meta = parse_scene_xml(scene.xml_path)
        out.append(_summary(scene, meta, read_raster_info(scene.source_path), read_raster_info(scene.reference_path)))
    return out


@router.get("/api/scenes/{scene_id}", response_model=SceneDetail)
def scene_detail(scene_id: str) -> SceneDetail:
    scene = _scene_or_404(scene_id)
    key = scene.scene_id + "|" + "|".join(str(Path(p).stat().st_mtime) for p in (scene.source_path, scene.reference_path, scene.xml_path))
    with _detail_lock:
        cached = _detail_cache.get(key)
    if cached is None:
        meta = parse_scene_xml(scene.xml_path)
        src, ref = load_raster(scene.source_path), load_raster(scene.reference_path)
        cached = SceneDetail(
            scene=scene,
            summary=_summary(scene, meta, src.info, ref.info),
            metadata=meta,
            source_info=src.info,
            reference_info=ref.info,
            source_stats=compute_stats(src.data, src.valid_mask),
            reference_stats=compute_stats(ref.data, ref.valid_mask),
            predicted_footprint=predict_footprint(src.info, ref.info, meta),
        ).model_dump()
        with _detail_lock:
            _detail_cache[key] = cached
    return SceneDetail.model_validate(cached)


@router.get("/api/scenes/{scene_id}/preview/{which}")
def scene_preview(scene_id: str, which: str) -> FileResponse:
    """Percentile-stretched JPEG preview of the source or reference raster (read-only on the data)."""
    if which not in ("source", "reference"):
        raise HTTPException(status_code=404, detail="Preview must be 'source' or 'reference'")
    scene = _scene_or_404(scene_id)
    cache = config.RESULTS_ROOT / "cache" / "previews" / f"{scene.scene_id}_{which}.jpg"
    raster_path = Path(scene.source_path if which == "source" else scene.reference_path)
    if not cache.is_file() or cache.stat().st_mtime < raster_path.stat().st_mtime:
        raster = load_raster(raster_path)
        image, _, _ = percentile_normalize(raster.data, raster.valid_mask, 1.0, 99.0)
        scale = fit_scale(image.shape[1], image.shape[0], 1600)
        if scale < 1:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        save_image(cache, image, 85)
    return FileResponse(cache, media_type="image/jpeg")


@router.get("/api/config/default")
def default_config() -> dict:
    return config.PipelineConfig().model_dump()


@router.post("/api/register/upload", response_model=JobState, status_code=202)
def register_upload(
    source: UploadFile = File(..., description="Source image (GeoTIFF/TIFF)"),
    reference: UploadFile = File(..., description="Reference image (GeoTIFF/TIFF)"),
    xml: UploadFile | None = File(None, description="Optional Chandrayaan-2 product XML"),
) -> JobState:
    """Register two uploaded rasters. Uploads are stored under results/uploads/, never in data/."""
    for upload, kinds in ((source, (".tif", ".tiff")), (reference, (".tif", ".tiff"))):
        if Path(upload.filename or "").suffix.lower() not in kinds:
            raise HTTPException(status_code=400, detail=f"'{upload.filename}' must be a .tif/.tiff file")
    if xml is not None and Path(xml.filename or "").suffix.lower() != ".xml":
        raise HTTPException(status_code=400, detail=f"'{xml.filename}' must be an .xml file")

    upload_id = store.new_job_id("upload")
    folder = config.RESULTS_ROOT / "uploads" / upload_id
    folder.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for role, upload in (("source", source), ("reference", reference), ("xml", xml)):
        if upload is None:
            continue
        target = folder / f"{role}{Path(upload.filename or '').suffix.lower()}"
        with target.open("wb") as fh:
            shutil.copyfileobj(upload.file, fh)
        paths[role] = target
    try:
        src, ref = load_raster(paths["source"]), load_raster(paths["reference"])
        meta = parse_scene_xml(paths["xml"]) if "xml" in paths else None
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read the uploaded files: {exc}") from exc
    cfg = config.PipelineConfig()

    def run(out: Path, progress):
        return register(src, ref, cfg, metadata=meta, out_dir=out, progress=progress)

    return jobs.submit(f"upload-{Path(source.filename or 'source').stem}", None, run)


@router.post("/api/register/{scene_id}", response_model=JobState, status_code=202)
def register_official_scene(scene_id: str, overrides: dict | None = Body(default=None)) -> JobState:
    """Start registering an official scene. The body may hold partial config overrides."""
    scene = _scene_or_404(scene_id)
    try:
        cfg = merge_config((overrides or {}).get("config", overrides))
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=json.loads(exc.json())) from exc

    def run(out: Path, progress):
        return register_scene(scene, cfg, out, progress)

    return jobs.submit(scene.scene_id, scene.scene_id, run)


@router.get("/api/results", response_model=list[JobState])
def list_results() -> list[JobState]:
    return jobs.list()


def _job_or_404(job_id: str) -> JobState:
    state = jobs.get(job_id) if store.is_valid_job_id(job_id) else None
    if state is None:
        raise HTTPException(status_code=404, detail=f"Unknown job '{job_id}'")
    return state


@router.get("/api/results/{job_id}")
def get_result(job_id: str) -> dict:
    state = _job_or_404(job_id)
    body: dict = {"job": state.model_dump(), "result": None}
    if state.state == "done":
        body["result"] = store.load_result(job_id)
    return body


@router.get("/api/results/{job_id}/metrics")
def get_metrics(job_id: str) -> dict:
    state = _job_or_404(job_id)
    if state.state == "error":
        return {"job_id": job_id, "state": state.state, "registration_status": "FAILED", "failure_reasons": [state.error]}
    if state.state != "done":
        return {"job_id": job_id, "state": state.state, "metrics": None}
    result = store.load_result(job_id) or {}
    return {
        "job_id": job_id,
        "state": state.state,
        "registration_status": result.get("status"),
        "failure_reasons": result.get("failure_reasons", []),
        "warnings": result.get("warnings", []),
        "metrics": result.get("metrics"),
    }


@router.get("/api/results/{job_id}/artifacts/{name}")
def get_artifact(job_id: str, name: str) -> FileResponse:
    _job_or_404(job_id)
    if not ARTIFACT_NAME.match(name):
        raise HTTPException(status_code=404, detail="Unknown artifact")
    path = store.job_dir(job_id) / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Unknown artifact")
    return FileResponse(path)
