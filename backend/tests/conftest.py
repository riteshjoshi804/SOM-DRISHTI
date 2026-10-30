"""Shared fixtures: small synthetic rasters so unit tests run in seconds."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pytest
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import from_origin

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app import config  # noqa: E402

OFFICIAL = config.DATA_ROOT
requires_official = pytest.mark.skipif(
    not (OFFICIAL.is_dir() and any(OFFICIAL.rglob("*.xml"))), reason="official mentor dataset not present"
)


def lunar_like(size: tuple[int, int], seed: int = 0) -> np.ndarray:
    """Deterministic crater-field texture (float32, 0..1) - rich in SIFT features."""
    rng = np.random.default_rng(seed)
    h, w = size
    img = np.zeros((h, w), np.float32)
    for sigma, weight in ((24, 0.5), (8, 0.3), (2, 0.2)):
        img += weight * cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), sigma)
    for _ in range(h * w // 2500):  # craters: dark bowl with a bright rim
        cx, cy, r = rng.integers(0, w), rng.integers(0, h), int(rng.integers(4, 22))
        cv2.circle(img, (int(cx), int(cy)), r, float(img.min()), -1, cv2.LINE_AA)
        cv2.circle(img, (int(cx), int(cy)), r, float(img.max()), 2, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 1.0)
    return (img - img.min()) / (img.max() - img.min())


def write_tif(path: Path, data: np.ndarray, nodata=None, georef: bool = False) -> Path:
    profile = dict(driver="GTiff", height=data.shape[0], width=data.shape[1], count=1, dtype=str(data.dtype))
    if nodata is not None:
        profile["nodata"] = nodata
    if georef:
        profile["crs"] = "+proj=stere +lat_0=-90 +lon_0=0 +R=1737400 +units=m +no_defs"
        profile["transform"] = from_origin(1000.0, 2000.0, 5.0, 5.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)  # plain TIFFs are intended here
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data, 1)
    return path


XML_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<product>
    <job_id>{scene_id}</job_id>
    <pan>
        <corners>
            <topleft_latitude>-85.1</topleft_latitude>
            <topleft_longitude>30.0</topleft_longitude>
            <topright_latitude>-85.2</topright_latitude>
            <topright_longitude>29.0</topright_longitude>
            <bottomleft_latitude>-84.5</bottomleft_latitude>
            <bottomleft_longitude>27.0</bottomleft_longitude>
            <bottomright_latitude>-84.6</bottomright_latitude>
            <bottomright_longitude>26.0</bottomright_longitude>
            <topleft_latitude>-85.1</topleft_latitude>
        </corners>
        <statistics>
            <minimum>0</minimum>
            <mean>12.5</mean>
            <mean>99.0</mean>
        </statistics>
    </pan>
    <Resolution_in_meter>0.25</Resolution_in_meter>
    <Sun_azimuth_in_degree>45.5</Sun_azimuth_in_degree>
    <Sun_elevation_in_degree>3.25</Sun_elevation_in_degree>
    <projection>Polar stereographic</projection>
</product>
"""


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """Two OHRC-like scenes (one with a prefix-colliding ID), one orphan XML."""
    root = tmp_path / "data"
    (root / "ohrc").mkdir(parents=True)
    ref = (lunar_like((400, 400), 1) * 250 + 1).astype(np.uint8)
    src = ref[100:300, 120:260].copy()
    for sid in ("OHRXXD18TEST_V2_1", "OHRXXD18TEST_V2_1_01"):
        (root / "ohrc" / f"{sid}.xml").write_text(XML_TEMPLATE.format(scene_id=sid), encoding="utf-8")
        write_tif(root / "ohrc" / f"{sid}_source_correctly_aligned_at_5m.tif", src)
        write_tif(root / "ohrc" / f"{sid}_reference_at_5m.tif", ref, nodata=0, georef=True)
    (root / "ohrc" / "OHRXXD18ORPHAN_V1.xml").write_text(XML_TEMPLATE.format(scene_id="ORPHAN"), encoding="utf-8")
    return root
