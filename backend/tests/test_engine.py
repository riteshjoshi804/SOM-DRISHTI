"""End-to-end registration on synthetic data with a known answer."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import rasterio
from conftest import lunar_like, write_tif

from app.config import PipelineConfig
from app.geometry.estimation import apply_transform
from app.ingestion.raster import load_raster
from app.registration.engine import register

# Source pixel -> reference pixel: 15 deg rotation, 3% scale, offset into a larger reference.
TRUE = np.array(
    [
        [1.03 * np.cos(np.radians(15)), -1.03 * np.sin(np.radians(15)), 420.0],
        [1.03 * np.sin(np.radians(15)), 1.03 * np.cos(np.radians(15)), 180.0],
        [0.0, 0.0, 1.0],
    ]
)
SRC_W, SRC_H = 260, 560


def _config() -> PipelineConfig:
    cfg = PipelineConfig(preprocessing_modes=["clahe"])
    cfg.sift.tile_size = 512
    cfg.sift.nfeatures = 1500
    return cfg


@pytest.fixture(scope="module")
def rasters(tmp_path_factory):
    d = tmp_path_factory.mktemp("engine")
    ref = (lunar_like((900, 900), 21) * 250 + 1).astype(np.uint8)
    src = cv2.warpAffine(ref, TRUE[:2], (SRC_W, SRC_H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    other = (lunar_like((SRC_H, SRC_W), 99) * 250 + 1).astype(np.uint8)
    return {
        "reference": load_raster(write_tif(d / "ref.tif", ref, nodata=0, georef=True)),
        "source": load_raster(write_tif(d / "src.tif", src)),
        "mirrored": load_raster(write_tif(d / "mir.tif", np.ascontiguousarray(src[:, ::-1]))),
        "unrelated": load_raster(write_tif(d / "other.tif", other)),
        "blank": load_raster(write_tif(d / "blank.tif", np.full((SRC_H, SRC_W), 90, np.uint8))),
    }


def _corners() -> np.ndarray:
    return np.array([[0, 0], [SRC_W, 0], [SRC_W, SRC_H], [0, SRC_H]], float)


def test_recovers_known_transform_and_writes_artifacts(rasters, tmp_path: Path) -> None:
    result = register(rasters["source"], rasters["reference"], _config(), out_dir=tmp_path)
    assert result.status == "SUCCESS", (result.failure_reasons, result.warnings)
    m = result.metrics
    assert m.inliers >= 30 and m.reprojection_rmse_px < 1.5
    assert m.image_space_spatial_coverage >= 0.5
    assert m.expected_chance_inliers < 0.1
    matrix = np.array(result.transformation.matrix)
    np.testing.assert_allclose(apply_transform(matrix, _corners()), apply_transform(TRUE, _corners()), atol=2.0)
    assert not result.transformation.mirrored_source

    for name in ("registered_source.png", "overlay.png", "inlier_matches.jpg", "raw_matches.jpg", "spatial_coverage.jpg", "registered_source.tif"):
        assert (tmp_path / name).is_file(), name
    with rasterio.open(tmp_path / "registered_source.tif") as ds:
        assert ds.crs is not None and ds.width == 900 and ds.dtypes[0] == "uint8"


def test_detects_mirrored_source(rasters) -> None:
    result = register(rasters["mirrored"], rasters["reference"], _config())
    assert result.status == "SUCCESS"
    assert result.transformation.mirrored_source
    matrix = np.array(result.transformation.matrix)
    assert np.linalg.det(matrix[:2, :2]) < 0
    flip = np.array([[-1.0, 0, SRC_W - 1.0], [0, 1, 0], [0, 0, 1]])
    np.testing.assert_allclose(apply_transform(matrix @ flip, _corners()), apply_transform(TRUE, _corners()), atol=2.0)


def test_unrelated_source_fails_safely(rasters, tmp_path: Path) -> None:
    result = register(rasters["unrelated"], rasters["reference"], _config(), out_dir=tmp_path)
    assert result.status == "FAILED"
    assert result.failure_reasons
    assert result.transformation is None
    assert result.metrics.transformation_type is None
    assert "registered_source" not in result.artifacts and not (tmp_path / "overlay.png").exists()


def test_featureless_source_reports_insufficient_keypoints(rasters) -> None:
    result = register(rasters["blank"], rasters["reference"], _config())
    assert result.status == "FAILED"
    assert any("insufficient source keypoints" in r for r in result.failure_reasons)
