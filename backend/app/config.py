"""Paths and tunable pipeline configuration.

Every threshold used by the registration engine lives here so that nothing is
hard-coded inside the algorithms and the API can accept per-run overrides.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader (KEY=VALUE lines); real environment variables win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(PROJECT_ROOT / ".env")


def _path_from_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if not raw:
        return default
    p = Path(raw)
    return p if p.is_absolute() else PROJECT_ROOT / p


DATA_ROOT = _path_from_env("SOM_DRISHTI_DATA_DIR", PROJECT_ROOT / "data" / "official_sih_2026")
RESULTS_ROOT = _path_from_env("SOM_DRISHTI_RESULTS_DIR", PROJECT_ROOT / "results")
JOBS_ROOT = RESULTS_ROOT / "jobs"

# Scene used by scripts/run_registration.py when --scene is omitted. The a-priori first
# engineering case was 2436502 (highest Sun elevation, 5.1 deg); classical SIFT finds no
# geometry on it at all (see docs/engineering_findings.md), so the default demonstration
# case is 2359602, the scene the MVP registers with the most support.
DEFAULT_FIRST_SCENE = os.environ.get(
    "SOM_DRISHTI_FIRST_SCENE", "OHRXXD18CHO2359602NNNN24342131250969_V1_0_03"
)

PreprocessMode = Literal["raw_normalized", "clahe", "gradient"]
TransformModel = Literal["similarity", "affine", "homography"]


class PreprocessConfig(BaseModel):
    percentile_low: float = Field(1.0, ge=0, lt=50)
    percentile_high: float = Field(99.0, gt=50, le=100)
    clahe_clip_limit: float = Field(3.0, gt=0)
    clahe_tile_px: int = Field(64, ge=8, description="Approximate CLAHE tile size in pixels")
    gaussian_sigma: float = Field(0.0, ge=0, description="0 disables noise reduction")
    gradient_sigma: float = Field(1.0, ge=0, description="Smoothing applied before the gradient")
    mask_erosion_px: int = Field(8, ge=0, description="Keep features this far from invalid pixels")
    min_border_fill_area_px: int = Field(64, ge=1)


class SiftConfig(BaseModel):
    nfeatures: int = Field(3000, ge=0, description="Max keypoints per tile (0 = unlimited)")
    contrast_threshold: float = Field(0.02, gt=0)
    edge_threshold: float = Field(10.0, gt=0)
    sigma: float = Field(1.6, gt=0)
    n_octave_layers: int = Field(3, ge=1)
    tile_size: int = Field(768, ge=128, description="Tile edge for tiled extraction")
    tile_overlap: int = Field(64, ge=0)
    root_sift: bool = True
    max_workers: int = Field(0, ge=0, description="0 = automatic")


class MatchConfig(BaseModel):
    matcher: Literal["auto", "bf", "flann"] = "auto"
    ratio_threshold: float = Field(0.90, gt=0, le=1)
    bf_max_pairs: float = Field(6e9, description="'auto' switches to FLANN above this many pairs")
    flann_trees: int = 8
    flann_checks: int = 128
    unique_reference: bool = Field(True, description="Keep only the best match per reference keypoint")


class GeometryConfig(BaseModel):
    model: TransformModel | Literal["auto"] = "auto"
    method: Literal["ransac", "magsac"] = "magsac"
    reproj_threshold_px: float = Field(3.0, gt=0)
    max_iters: int = Field(50000, ge=100)
    confidence: float = Field(0.999, gt=0, lt=1)
    # Seed search: RANSAC on the best-ratio matches first, widening the pool
    # only if no consensus is found (progressive sampling, as in PROSAC).
    seed_pool_sizes: list[int] = Field(default_factory=lambda: [60, 150, 400, 1000])
    min_seed_inliers: int = Field(5, ge=4, description="Spatially distinct inliers a seed needs")
    seed_min_separation_px: float = Field(16.0, gt=0)
    # Expansion: ratio-filtered matches within these radii of the current model
    # are re-fitted, tightening towards ``reproj_threshold_px``.
    expansion_radii_px: list[float] = Field(default_factory=lambda: [40.0, 15.0])
    tie_point_min_separation_px: float = Field(2.0, gt=0)
    homography_min_inliers: int = Field(30, ge=4)
    homography_min_gain: float = Field(1.2, ge=1.0, description="Inlier gain needed to prefer it in 'auto'")
    allow_mirrored_source: bool = Field(
        True, description="If no geometry is found, retry with the source flipped left-right"
    )


class ValidationConfig(BaseModel):
    # Hard limits -> FAILED
    min_source_keypoints: int = 50
    min_reference_keypoints: int = 50
    min_matches: int = 12
    min_inliers: int = 12
    # With pooled, ratio-filtered candidates only a few percent can be right on
    # these pairs; the decisive evidence is inliers vs. expected_chance_inliers.
    min_inlier_ratio: float = 0.005
    min_scale: float = 0.2
    max_scale: float = 5.0
    max_anisotropy: float = 2.0
    max_shear_deg: float = 30.0
    max_perspective_ratio: float = Field(
        1.5, description="Max ratio of the homogeneous weight across the source corners"
    )
    max_condition_number: float = 1e4
    min_projected_overlap: float = Field(0.02, description="Min fraction of source landing in the reference")
    min_spatial_coverage: float = 0.05
    # Soft limits -> LOW_CONFIDENCE
    good_inliers: int = 30
    good_inlier_ratio: float = 0.05
    good_spatial_coverage: float = 0.25
    good_rmse_px: float = 2.5
    metadata_offset_warn_m: float = Field(
        2000.0, description="Add a note when the result is further than this from the XML/GCP prediction"
    )


class CoverageConfig(BaseModel):
    grid_rows: int = Field(8, ge=1)
    grid_cols: int = Field(8, ge=1)
    min_valid_fraction: float = Field(0.10, ge=0, le=1, description="Cell counts as valid above this")


class VisualizationConfig(BaseModel):
    max_side_px: int = Field(2200, ge=256)
    max_match_lines: int = Field(150, ge=1)
    jpeg_quality: int = Field(88, ge=1, le=100)


class PipelineConfig(BaseModel):
    preprocessing_modes: list[PreprocessMode] = Field(
        default_factory=lambda: ["clahe", "raw_normalized", "gradient"],
        description="Representations to try; the strongest valid geometry wins",
    )
    evaluate_all_modes: bool = Field(
        False, description="Try every representation even after one reaches SUCCESS (benchmarking)"
    )
    pool_modes: bool = Field(
        True, description="If no single representation reaches SUCCESS, verify their candidates jointly"
    )
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    sift: SiftConfig = Field(default_factory=SiftConfig)
    match: MatchConfig = Field(default_factory=MatchConfig)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    coverage: CoverageConfig = Field(default_factory=CoverageConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    random_seed: int = 20260166
    write_geotiff: bool = True
