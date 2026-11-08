"""Independent control points for a finished registration.

The points are applied to the stored transformation and nothing is refitted, so they
cannot influence what they measure. They are saved with the job, and its report is rewritten
to include the result under its own heading.
"""
from __future__ import annotations

import io
import warnings
import zipfile
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window

from app.evaluation.validation import ControlPoint, ControlPointEvaluation, evaluate_control_points
from app.geometry.estimation import apply_transform
from app.geometry.strip_model import StripModel
from app.registration import exports, store
from app.registration.schemas import RegistrationResult

CONTROL_FILE = "control_points.json"


def mapping_of(result: RegistrationResult):
    """The stored source -> reference mapping, or None if the run accepted no transformation."""
    transformation = result.transformation
    if transformation is None:
        return None
    if transformation.strip_model is not None:
        return StripModel.from_dict(transformation.strip_model).apply
    matrix = np.asarray(transformation.matrix, dtype=np.float64)
    return lambda points: apply_transform(matrix, points)


def add_control_points(job_id: str, points: list[ControlPoint], provenance: str) -> ControlPointEvaluation:
    """Evaluate a job against independent points. Raises ValueError when that is not possible."""
    data = store.load_result(job_id)
    if data is None:
        raise ValueError("this job has no stored result")
    result = RegistrationResult.model_validate(data)
    mapping = mapping_of(result)
    if mapping is None:
        raise ValueError("this registration has no accepted transformation to evaluate")
    provenance = provenance.strip()
    if not provenance:
        raise ValueError("say where the control points come from; provenance is what makes them independent")
    evaluation = evaluate_control_points(
        points,
        mapping,
        provenance,
        (result.source.width, result.source.height),
        (result.reference.width, result.reference.height),
    )
    directory = store.job_dir(job_id)
    (directory / CONTROL_FILE).write_text(evaluation.model_dump_json(indent=2), encoding="utf-8")
    result.control_points = evaluation
    result.metrics.control_point_rmse_px = evaluation.rmse_px
    if result.evidence is not None:
        result.evidence.has_independent_ground_truth = True
    result.artifacts["control_points"] = CONTROL_FILE
    result.artifacts.update(exports.write_result_documents(directory, result))
    store.save_result(directory, result)
    return evaluation


def render_chip(path: Path, x: float, y: float, size: int, linear: np.ndarray) -> bytes:
    """PNG of the raster around (x, y), through ``linear`` (original pixel offsets -> chip pixel offsets).

    Only the part of the raster under the chip is read and stretched, so that terrain inside a
    dim region of a big image is still visible. Outside the raster the chip is black.
    """
    linear = np.asarray(linear, dtype=np.float64)
    inverse = np.linalg.inv(linear)
    half = (size - 1) / 2.0
    corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]]) @ inverse.T
    reach = float(np.abs(corners).max()) + 3.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as ds:
            x0, y0 = int(np.clip(np.floor(x - reach), 0, ds.width)), int(np.clip(np.floor(y - reach), 0, ds.height))
            x1, y1 = int(np.clip(np.ceil(x + reach) + 1, 0, ds.width)), int(np.clip(np.ceil(y + reach) + 1, 0, ds.height))
            if x1 <= x0 or y1 <= y0:
                data = np.zeros((1, 1), np.float32)
            else:
                data = ds.read(1, window=Window(x0, y0, x1 - x0, y1 - y0)).astype(np.float32)
    finite = data[np.isfinite(data)]
    low, high = (np.percentile(finite, (1.0, 99.0)) if finite.size else (0.0, 1.0))
    stretched = np.clip((np.nan_to_num(data, nan=low) - low) * (255.0 / max(float(high - low), 1e-6)), 0, 255).astype(np.float32)
    offset = np.array([half, half]) - linear @ np.array([x - x0, y - y0], dtype=np.float64)
    chip = cv2.warpAffine(stretched, np.c_[linear, offset], (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ok, encoded = cv2.imencode(".png", np.clip(chip, 0, 255).astype(np.uint8))
    if not ok:
        raise OSError("Could not encode the chip")
    return encoded.tobytes()


def zip_job(directory: Path) -> io.BytesIO:
    """All deliverables of a job as one archive."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(directory.iterdir()):
            if path.is_file():
                archive.write(path, path.name)
    buffer.seek(0)
    return buffer
