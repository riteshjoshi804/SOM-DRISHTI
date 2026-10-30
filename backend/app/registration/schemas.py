"""JSON-serialisable result of one registration run."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.evaluation.metrics import MetadataConsistency, SpatialCoverage
from app.geometry.validation import TransformProperties
from app.ingestion.raster import RasterStats
from app.metadata.georef import PredictedFootprint
from app.metadata.xml_parser import SceneMetadata

Status = Literal["SUCCESS", "LOW_CONFIDENCE", "FAILED"]


class ImageSummary(BaseModel):
    path: str | None
    width: int
    height: int
    dtype: str
    nodata: float | None
    has_georeferencing: bool
    stats: RasterStats


class AttemptSummary(BaseModel):
    preprocessing_mode: str
    mirrored_source: bool
    source_keypoints: int
    reference_keypoints: int
    raw_matches: int
    ratio_filtered_matches: int
    inliers: int
    inlier_ratio: float | None
    transformation_type: str | None
    status: Status
    reasons: list[str]
    log: list[str]
    seconds: float


class Metrics(BaseModel):
    """Every value is measured on this run; ``None`` means 'not available'."""

    source_keypoints: int | None = None
    reference_keypoints: int | None = None
    raw_matches: int | None = None
    ratio_filtered_matches: int | None = None
    inliers: int | None = None
    inlier_ratio: float | None = None
    reprojection_rmse_px: float | None = None
    median_reprojection_error_px: float | None = None
    max_reprojection_error_px: float | None = None
    cross_validated_rmse_px: float | None = None
    image_space_spatial_coverage: float | None = None
    occupied_cells: int | None = None
    valid_cells: int | None = None
    coverage_within_overlap: float | None = None
    estimated_overlap_fraction: float | None = None
    expected_chance_inliers: float | None = None
    processing_time_seconds: float | None = None
    transformation_type: str | None = None
    registration_status: Status = "FAILED"


class Transformation(BaseModel):
    model: str
    method: str
    matrix: list[list[float]]  # 3x3, source pixel -> reference pixel
    mirrored_source: bool
    seed_pool: int
    seed_inliers: int
    properties: TransformProperties


class RegistrationResult(BaseModel):
    scene_id: str | None
    sensor: str | None
    reference_sensor: str | None
    status: Status
    failure_reasons: list[str]
    warnings: list[str]
    metrics: Metrics
    selected_preprocessing: str | None
    transformation: Transformation | None
    spatial_coverage: SpatialCoverage | None
    metadata_consistency: MetadataConsistency | None
    predicted_footprint: PredictedFootprint | None
    attempts: list[AttemptSummary]
    source: ImageSummary
    reference: ImageSummary
    metadata: SceneMetadata | None
    artifacts: dict[str, str]
    config: dict
    notes: list[str]
