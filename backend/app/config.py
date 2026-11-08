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
# Pretrained weights for the learned matchers; kept out of Git (see .gitignore).
MODELS_ROOT = _path_from_env("SOM_DRISHTI_MODELS_DIR", PROJECT_ROOT / "models")

# Scene used by scripts/run_registration.py when --scene is omitted. The a-priori first
# engineering case was 2436502 (highest Sun elevation, 5.1 deg); classical SIFT finds no
# geometry on it at all (see docs/engineering_findings.md), so the default demonstration
# case is 2359602, the scene the MVP registers with the most support.
DEFAULT_FIRST_SCENE = os.environ.get(
    "SOM_DRISHTI_FIRST_SCENE", "OHRXXD18CHO2359602NNNN24342131250969_V1_0_03"
)

PreprocessMode = Literal["raw_normalized", "clahe", "gradient", "structural", "phase_congruency", "shadow_aware"]
TransformModel = Literal["similarity", "affine", "homography"]
MatcherChoice = Literal["AUTO", "SIFT", "SUPERPOINT_LIGHTGLUE", "LOFTR", "RIFT"]


class PreprocessConfig(BaseModel):
    percentile_low: float = Field(1.0, ge=0, lt=50)
    percentile_high: float = Field(99.0, gt=50, le=100)
    clahe_clip_limit: float = Field(3.0, gt=0)
    clahe_tile_px: int = Field(64, ge=8, description="Approximate CLAHE tile size in pixels")
    gaussian_sigma: float = Field(0.0, ge=0, description="0 disables noise reduction")
    gradient_sigma: float = Field(1.0, ge=0, description="Smoothing applied before the gradient")
    mask_erosion_px: int = Field(8, ge=0, description="Keep features this far from invalid pixels")
    min_border_fill_area_px: int = Field(64, ge=1)
    structural_window_px: int = Field(31, ge=3, description="Local-mean window of the structural representation")
    phase_scales: int = Field(4, ge=2)
    phase_orientations: int = Field(6, ge=2)
    shadow_threshold: int = Field(10, ge=0, le=255, description="Grey level (after the stretch) at or below which terrain counts as shadow")
    shadow_min_area_px: int = Field(100, ge=1)
    shadow_feather_px: float = Field(4.0, gt=0)
    shadow_margin_px: int = Field(6, ge=0, description="Features this close to a deep shadow are not used by shadow_aware")


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
    use_search_region: bool = Field(
        False, description="Only look in the metadata search region of the reference (off for the classical baseline)"
    )


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
    # Learned matchers return something for any input, so they face stricter limits and a measured
    # null control. Provisional values from decoy runs on one official scene; they must be recalibrated
    # across all scenes (docs/DEVELOPMENT_PROGRESS.md).
    learned_min_inliers: int = Field(40, description="Hard limit for learned matchers")
    learned_min_inlier_ratio: float = Field(0.02, description="Hard limit for learned matchers")
    # Metadata knows orientation and scale far better than position (within 1.7 deg on the scenes
    # where SIFT found geometry), so a guided result far from them is not plausible.
    max_prior_rotation_deg: float = Field(10.0, gt=0, description="Hard limit against the metadata orientation")
    max_prior_scale_deviation: float = Field(0.25, gt=0, description="Hard limit against the metadata scale")
    max_matcher_disagreement_px: float = Field(
        10.0, gt=0, description="Median distance of another matcher's tie points before AUTO lowers the status"
    )
    # Soft limits -> LOW_CONFIDENCE
    good_inliers: int = 30
    good_inlier_ratio: float = 0.05
    good_spatial_coverage: float = 0.25
    good_rmse_px: float = 2.5
    good_unseen_block_rmse_px: float = Field(
        5.0, description="Soft limit on predicting ground left out of the fit; not applied to the SIFT-only baseline"
    )
    min_along_track_support: float = Field(
        0.5, description="Soft limit on the share of the strip length with tie points; not applied to the SIFT-only baseline"
    )
    along_track_support_min_points: int = Field(
        3, ge=1, description="Tie points a twentieth of the strip needs to count as supported; fewer cannot constrain it locally"
    )
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


class SuperPointConfig(BaseModel):
    keypoint_threshold: float = Field(0.005, gt=0, description="Detection score threshold")
    nms_radius: int = Field(4, ge=1)
    max_keypoints: int = Field(2048, ge=0, description="Per matched image; 0 = unlimited")
    tile_px: int = Field(1024, ge=128, description="Large images are extracted in tiles of this size")
    tile_overlap_px: int = Field(32, ge=0)


class LightGlueConfig(BaseModel):
    filter_threshold: float = Field(0.2, ge=0, le=1, description="Minimum LightGlue match confidence")
    depth_confidence: float = Field(-1.0, description="Early stopping; -1 disables it (accuracy over speed)")
    width_confidence: float = Field(-1.0, description="Point pruning; -1 disables it (accuracy over speed)")


class LoftrConfig(BaseModel):
    confidence_threshold: float = Field(0.2, ge=0, le=1, description="Minimum LoFTR coarse match confidence")


class GuidedSearchConfig(BaseModel):
    """Search used by the learned matchers, which are not invariant to rotation or large scale change.

    All pixel values are in reference pixels unless stated otherwise.
    """

    representation: PreprocessMode = Field("clahe", description="Image representation the matcher sees")
    search_radius_m: float = Field(9000.0, gt=0, description="Assumed error of the metadata position, in metres")
    search_radius_footprint_fraction: float = Field(
        0.1, ge=0, description="...or this fraction of the footprint's long side, whichever is larger"
    )
    # Location search by gradient correlation, before any matcher runs: see matching/localisation.py.
    correlation_search: bool = Field(True, description="Try the matcher-independent location search first")
    correlation_tile_px: int = Field(128, ge=32, description="Tile size at the working scale")
    correlation_step_fraction: float = Field(0.5, gt=0, le=1, description="Tile spacing as a fraction of a tile's extent")
    correlation_max_tiles: int = Field(20, ge=1, description="Tiles per orientation")
    correlation_agreement_px: float = Field(120.0, gt=0, description="Tiles agree if their offsets are this close; a strip drifts by tens of pixels")
    correlation_angles_deg: list[float] = Field(default_factory=lambda: [-2.0, 0.0, 2.0], description="Tried around the metadata orientation")
    correlation_min_overlap: float = Field(0.35, gt=0, le=1, description="Share of a tile's valid pixels that must land on valid reference pixels")
    correlation_min_valid_fraction: float = Field(0.1, ge=0, le=1, description="Share of a tile that must hold valid source pixels")
    correlation_min_z: float = Field(4.5, gt=0, description="Robust z-score of a peak within its correlation map")
    correlation_min_tiles: int = Field(2, ge=2, description="Tiles sharing no source pixels that must agree on one offset")
    # True locations reached 8.4 to 11.9; pairs with no shared ground and wrong orientations (28 scans) at most 7.1.
    correlation_min_mean_z: float = Field(8.0, gt=0, description="Mean z of the (up to three) strongest agreeing tiles")
    # Anchor sweep: find where one source tile lands, from the metadata prior alone.
    anchor_scale: float = Field(0.5, gt=0, le=4, description="Working scale of the sweep (0.5 = half resolution)")
    anchor_tile_px: int = Field(320, ge=64, description="Tile size at the working scale")
    anchor_step_fraction: float = Field(0.5, gt=0, le=1, description="Sweep step as a fraction of the window")
    # Eight tiles, because a strip can lie mostly off its reference and texture does not tell which tiles
    # have ground: with four tiles, chosen by texture per stretch, scene 2436502 never tried the tile that
    # finds its location (source row 1583) or the one that confirms it (row 321). A sweep costs about 100
    # image pairs per tile and orientation there, so all tiles fit in the budget. This only lengthens the
    # search; every acceptance limit stays as it was.
    anchor_max_tiles: int = Field(8, ge=1, description="Source tiles to sweep before giving up")
    anchor_max_pairs: int = Field(1700, ge=1, description="Budget of image pairs for the whole sweep")
    anchor_vote_bin_px: float = Field(4.0, gt=0, description="Translation-vote bin at the working scale")
    anchor_min_peak: int = Field(6, ge=2, description="Votes for one translation, after the appearance check, that make a window a candidate")
    anchor_agreement_px: float = Field(90.0, gt=0, description="Tiles agree if their implied offsets are this close")
    anchor_confirm_tiles: int = Field(3, ge=1, description="Other tiles tried to confirm a candidate")
    anchor_min_confirmations: int = Field(1, ge=1, description="Independent tiles that must confirm it")
    anchor_confirm_min_ground: float = Field(0.5, ge=0, le=1, description="Share of a confirming tile that must land on the reference")
    # Appearance check, independent of the matcher: see matching/appearance.py.
    appearance_filter: bool = True
    appearance_min_zncc: float = Field(
        0.25, ge=-1, le=1, description="Gradient patch similarity a match must reach; best separation measured at 0.20-0.25"
    )
    appearance_patch_radius_px: int = Field(15, ge=3, description="Patch radius during tile matching")
    anchor_patch_radius_px: int = Field(12, ge=3, description="Patch radius during the anchor sweep")
    # Tile matching at full resolution around the current estimate.
    fine_scale: float = Field(1.0, gt=0, le=4, description="Working scale of the tile passes")
    fine_tile_px: int = Field(640, ge=64, description="Tile size at the working scale")
    fine_step_fraction: float = Field(0.75, gt=0, le=1)
    fine_margin_px: int = Field(96, ge=0, description="Extra reference margin around each predicted tile")
    fine_jitter_px: int = Field(
        32, ge=0, description="Random window offset, so window-relative positional bias cannot look like agreement"
    )
    min_tile_ground_fraction: float = Field(
        0.1, ge=0, le=1, description="Tiles predicted to have less valid reference ground than this are skipped"
    )
    fine_passes: int = Field(3, ge=1, description="Passes; each re-centres the windows on the improved model")
    tile_vote_bin_px: float = Field(8.0, gt=0, description="Translation-vote bin of the per-tile check")
    tile_min_votes: int = Field(4, ge=2, description="Votes that start the along-track track, after the appearance check")
    mask_erosion_px: int = Field(8, ge=0)
    min_tile_valid_fraction: float = Field(0.05, ge=0, le=1)
    max_orientation_hypotheses: int = Field(16, ge=1, description="When no metadata fixes the orientation")
    # Strip model: smooth along-track deviation from an affine (pushbroom scan lines).
    use_strip_model: bool = True
    strip_min_aspect: float = Field(2.0, ge=1.0, description="Only images at least this elongated are strips")
    strip_knot_spacing_px: float = Field(450.0, gt=0, description="Along-track spacing of the spline knots")
    strip_smoothness_candidates: list[float] = Field(default_factory=lambda: [1.0, 10.0, 1e2, 1e3, 1e4, 1e6])
    strip_radii_px: list[float] = Field(default_factory=lambda: [40.0, 20.0, 10.0, 5.0])
    strip_min_gain: float = Field(1.25, ge=1.0, description="Inlier gain over the global model needed to adopt it")
    strip_min_tiles: int = Field(3, ge=2, description="Tiles that must support the strip model")
    # Null control: repeat the verification on deliberately wrong ground.
    decoy_controls: int = Field(2, ge=0, description="Wrong-ground repetitions per accepted result; 0 disables")
    decoy_factor: float = Field(3.0, ge=1.0, description="Inliers must exceed the null result by this factor")


class RiftConfig(BaseModel):
    """Experimental phase-congruency matcher."""

    scales: int = Field(4, ge=2)
    orientations: int = Field(6, ge=2)
    max_keypoints: int = Field(2000, ge=8)
    nms_radius_px: int = Field(3, ge=1)
    min_phase_congruency: float = Field(0.05, ge=0, le=1)
    patch_radius_px: int = Field(48, ge=8)
    grid_cells: int = Field(6, ge=2)
    min_similarity: float = Field(0.7, ge=0, le=1, description="Minimum descriptor cosine similarity")


class TiePointConfig(BaseModel):
    """Spatially balanced tie-point selection from verified inliers."""

    grid_rows: int = Field(8, ge=1)
    grid_cols: int = Field(8, ge=1)
    max_per_cell: int = Field(6, ge=1)
    min_tie_points: int = Field(12, ge=1, description="Below this the result is flagged; points are never invented")
    max_tie_points: int = Field(400, ge=1)
    min_separation_px: float = Field(8.0, ge=0, description="Duplicate suppression radius in source pixels")
    anms_robust: float = Field(0.9, gt=0, le=1, description="Suppression only by clearly stronger points")


class RefinementConfig(BaseModel):
    """Sub-pixel refinement of verified correspondences."""

    enabled: bool = True
    strategy: Literal["ncc", "lucas_kanade", "ecc", "phase", "none"] = "ncc"
    # Chosen from scripts/refinement_study.py: on plain intensity a changed Sun pulled points several
    # pixels off while their correlation stayed high; on gradient magnitude it did not.
    representation: PreprocessMode = Field("gradient", description="Image representation compared by the refinement")
    patch_radius_px: int = Field(15, ge=4)
    search_radius_px: int = Field(4, ge=1, description="How far a point may move, in reference pixels")
    min_score: float = Field(0.35, ge=-1, le=1, description="Minimum correlation / ECC score at the optimum")
    min_peak_ratio: float = Field(1.05, ge=1.0, description="Best vs. second-best separated peak (ambiguity)")
    min_valid_fraction: float = Field(1.0, ge=0, le=1, description="Share of the patch that must hold valid pixels")
    min_texture_std: float = Field(4.0, ge=0, description="Patches flatter than this (grey levels) are not refined")
    refit_transformation: bool = Field(
        True, description="Refit the model on refined points when held-out error confirms the improvement"
    )


class HoldoutConfig(BaseModel):
    """Spatially stratified held-out validation of correspondences."""

    enabled: bool = True
    fraction: float = Field(0.3, gt=0, lt=1, description="Share of the candidates kept out of the fit")
    grid_rows: int = Field(8, ge=1)
    grid_cols: int = Field(8, ge=1)
    repeats: int = Field(5, ge=1)
    gate_px: float = Field(10.0, gt=0, description="Held-out candidates further than this count as non-corresponding")
    min_train_inliers: int = Field(6, ge=3)
    min_points: int = Field(5, ge=1, description="Held-out points needed to report statistics")


class PipelineConfig(BaseModel):
    matcher: MatcherChoice = Field(
        "SIFT", description="SIFT (classical baseline), SUPERPOINT_LIGHTGLUE, LOFTR, RIFT, or AUTO to compare them"
    )
    sift_decoy_controls: int = Field(
        0, ge=0, description="Run SIFT on the reference with the true ground removed (the benchmark enables it)"
    )
    device: str = Field("auto", description="auto, cpu or cuda; only the learned matchers use it")
    allow_model_download: bool = Field(True, description="Fetch missing weights from their official source")
    superpoint: SuperPointConfig = Field(default_factory=SuperPointConfig)
    lightglue: LightGlueConfig = Field(default_factory=LightGlueConfig)
    loftr: LoftrConfig = Field(default_factory=LoftrConfig)
    rift: RiftConfig = Field(default_factory=RiftConfig)
    guided: GuidedSearchConfig = Field(default_factory=GuidedSearchConfig)
    tie_points: TiePointConfig = Field(default_factory=TiePointConfig)
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)
    holdout: HoldoutConfig = Field(default_factory=HoldoutConfig)
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
