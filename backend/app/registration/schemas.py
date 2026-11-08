"""JSON-serialisable result of one registration run.

Everything added for the advanced matchers is optional, so results stored by earlier
versions still load and display.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.evaluation.metrics import MetadataConsistency, SpatialCoverage
from app.evaluation.validation import ControlPointEvaluation, HoldoutResult
from app.geometry.strip_model import StripCheck
from app.geometry.tiepoints import TiePointSummary
from app.geometry.validation import TransformProperties
from app.ingestion.raster import RasterStats
from app.matching.appearance import AppearanceSummary
from app.matching.selection import SelectionSummary
from app.metadata.georef import PredictedFootprint
from app.metadata.overlap import OverlapEstimate
from app.metadata.xml_parser import SceneMetadata
from app.preprocessing.illumination import IlluminationSummary
from app.refinement.subpixel import RefinementSummary

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
    source_keypoints: int | None
    reference_keypoints: int | None
    raw_matches: int
    ratio_filtered_matches: int
    inliers: int
    inlier_ratio: float | None
    transformation_type: str | None
    status: Status
    reasons: list[str]
    log: list[str]
    seconds: float
    matcher: str | None = None


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
    # Added with the advanced matchers.
    matcher: str | None = None
    device: str | None = None
    candidate_count: int | None = None  # proposed by the matcher, before its own filtering
    accepted_count: int | None = None  # after matcher-specific filtering
    global_model_inliers: int | None = None  # support of the best single global transformation
    selected_tie_points: int | None = None
    refined_tie_points: int | None = None
    median_refinement_px: float | None = None
    refined_reprojection_rmse_px: float | None = None
    refined_reprojection_median_px: float | None = None
    heldout_transfer_rmse_px: float | None = None  # automatic matches left out of the fit; not ground truth
    heldout_median_px: float | None = None
    heldout_p90_px: float | None = None
    block_heldout_rmse_px: float | None = None
    control_point_rmse_px: float | None = None  # independent points, only if a user supplied them
    spatial_distribution_score: float | None = None
    decoy_max_inliers: int | None = None
    appearance_median_zncc: float | None = None
    supported_along_track_fraction: float | None = None


class Transformation(BaseModel):
    model: str
    method: str
    matrix: list[list[float]]  # 3x3, source pixel -> reference pixel; best affine approximation for a strip model
    mirrored_source: bool
    seed_pool: int
    seed_inliers: int
    properties: TransformProperties
    strip_model: dict | None = None  # present when the mapping is the smooth along-track model
    strip_check: StripCheck | None = None
    refit_on_refined_points: bool = False
    direction: str = "source pixel -> reference pixel; pixel centres at integer coordinates"


class DecoySummary(BaseModel):
    """One run of the matcher on ground known to be wrong."""

    kind: str
    shift_px: list[float] | None = None
    tiles_used: int = 0
    candidates: int = 0
    candidates_before_appearance: int = 0
    inliers: int = 0
    valid: bool = False
    model: str | None = None
    seconds: float = 0.0


class MatcherSummary(BaseModel):
    """What one matcher branch achieved, whether or not it was selected."""

    matcher: str
    availability: Literal["OK", "FAILED", "UNAVAILABLE"] = "OK"
    registration_status: Status = "FAILED"
    reasons: list[str] = []
    warnings: list[str] = []
    device: str | None = None
    representation: str | None = None
    mirrored_source: bool | None = None
    source_keypoints: int | None = None
    reference_keypoints: int | None = None
    candidate_count: int | None = None
    accepted_count: int | None = None
    inliers: int | None = None
    inlier_ratio: float | None = None
    reprojection_rmse_px: float | None = None
    median_reprojection_error_px: float | None = None
    image_space_spatial_coverage: float | None = None
    coverage_within_overlap: float | None = None
    estimated_overlap_fraction: float | None = None
    along_track_support: float | None = None
    transformation_type: str | None = None
    holdout: HoldoutResult | None = None
    tie_points: TiePointSummary | None = None
    refinement: RefinementSummary | None = None
    refinement_validation: dict | None = None
    decoys: list[DecoySummary] = []
    appearance: AppearanceSummary | None = None
    processing_time_seconds: float | None = None
    artifacts: dict[str, str] = {}


class EvidenceSummary(BaseModel):
    """The three kinds of error evidence, never to be read as one another."""

    geometric_consistency: str = (
        "Residuals of the points used for the fit. Shows internal consistency only; it cannot reveal a systematic error."
    )
    heldout_validation: str = (
        "Automatically matched points left out of the fit. Real out-of-sample evidence, but they come from the same "
        "matcher, so they are not independent ground truth."
    )
    independent_accuracy: str = (
        "Independently identified control points, used for evaluation only. None ships with the official sample, "
        "so this stays empty unless a user supplies them."
    )
    has_independent_ground_truth: bool = False


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
    # Added with the advanced matchers.
    requested_matcher: str | None = None
    selected_matcher: str | None = None
    device: str | None = None
    matcher_results: list[MatcherSummary] = []
    selection: SelectionSummary | None = None
    overlap_estimate: OverlapEstimate | None = None
    illumination: IlluminationSummary | None = None
    tie_points: TiePointSummary | None = None
    refinement: RefinementSummary | None = None
    refinement_validation: dict | None = None
    holdout: HoldoutResult | None = None
    control_points: ControlPointEvaluation | None = None
    decoys: list[DecoySummary] = []
    appearance: AppearanceSummary | None = None
    evidence: EvidenceSummary | None = None
    search_trace: dict | None = None
