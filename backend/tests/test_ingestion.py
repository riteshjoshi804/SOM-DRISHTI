"""Dataset discovery, XML parsing and raster loading."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from conftest import OFFICIAL, requires_official, write_tif

from app.ingestion.indexer import discover_scenes, sensor_from_scene_id
from app.ingestion.raster import compute_valid_mask, load_raster
from app.metadata.xml_parser import parse_scene_xml


def test_discovery_pairs_by_scene_id(synthetic_dataset: Path) -> None:
    index = discover_scenes(synthetic_dataset)
    ids = {s.scene_id for s in index.scenes}
    assert ids == {"OHRXXD18TEST_V2_1", "OHRXXD18TEST_V2_1_01"}
    short = index.get("OHRXXD18TEST_V2_1")
    # The shorter ID must not claim the rasters of the longer, prefix-sharing ID.
    assert Path(short.source_path).name == "OHRXXD18TEST_V2_1_source_correctly_aligned_at_5m.tif"
    assert Path(short.reference_path).name == "OHRXXD18TEST_V2_1_reference_at_5m.tif"
    assert index.sensor_counts == {"OHRC": 2}


def test_discovery_reports_incomplete_scenes(synthetic_dataset: Path) -> None:
    index = discover_scenes(synthetic_dataset)
    assert [i.scene_id for i in index.incomplete] == ["OHRXXD18ORPHAN_V1"]
    assert index.incomplete[0].missing == ["source", "reference"]


def test_missing_data_root(tmp_path: Path) -> None:
    index = discover_scenes(tmp_path / "nope")
    assert not index.exists and index.scenes == []


def test_sensor_from_scene_id() -> None:
    assert sensor_from_scene_id("OHRXXD18CHO2359602") == "OHRC"
    assert sensor_from_scene_id("IIRXXD18CHO2686502") == "IIRS"
    assert sensor_from_scene_id("TMCXXD18") == "TMC-2"
    assert sensor_from_scene_id("ABC") == "UNKNOWN"


def test_xml_parser_values_duplicates_and_missing(synthetic_dataset: Path) -> None:
    meta = parse_scene_xml(synthetic_dataset / "ohrc" / "OHRXXD18TEST_V2_1.xml")
    assert meta.scene_id == "OHRXXD18TEST_V2_1"
    assert meta.sensor == "OHRC"
    assert meta.resolution_m == 0.25
    assert meta.sun_azimuth_deg == 45.5 and meta.sun_elevation_deg == 3.25
    assert meta.corners["top_left"].latitude == -85.1
    assert meta.corners["bottom_right"].longitude == 26.0
    # Absent fields are None - never a default.
    assert meta.corners_refined is None and meta.corners_en is None
    assert meta.roll_deg is None and meta.spacecraft_altitude_km is None
    # Duplicates: identical repeats are counted, disagreeing repeats reported.
    assert meta.duplicated_tag_count == 2
    assert meta.duplicate_conflicts == ["pan/statistics/mean"]
    assert meta.statistics["mean"] == 12.5  # first occurrence wins


def test_valid_mask_nodata_nan_inf_and_border_fill() -> None:
    data = np.full((50, 60), 7.0, np.float32)
    data[0:5, :] = 0  # zero fill touching the border -> invalid
    data[20:25, 20:25] = 0  # interior zeros (shadow) -> valid
    data[30, 30] = np.nan
    data[31, 31] = np.inf
    data[40, 40] = -9999
    mask = compute_valid_mask(data, nodata=-9999, min_border_fill_area_px=10)
    assert not mask[0:5].any()
    assert mask[20:25, 20:25].all()
    assert not mask[30, 30] and not mask[31, 31] and not mask[40, 40]
    assert mask[10, 10]


def test_load_raster_keeps_original_values(tmp_path: Path) -> None:
    data = np.linspace(-5, 1500, 40 * 30, dtype=np.float32).reshape(40, 30)
    path = write_tif(tmp_path / "f.tif", data)
    raster = load_raster(path)
    assert raster.data.dtype == np.float32
    np.testing.assert_array_equal(raster.data, data)
    assert raster.info.width == 30 and raster.info.height == 40
    assert raster.info.crs is None and not raster.info.has_geotransform

    u16 = write_tif(tmp_path / "u.tif", (np.arange(600, dtype=np.uint16).reshape(20, 30) * 50), georef=True)
    info = load_raster(u16).info
    assert info.dtype == "uint16" and info.has_geotransform and info.pixel_size == [5.0, 5.0]


@requires_official
def test_official_dataset_has_six_scenes() -> None:
    index = discover_scenes(OFFICIAL)
    assert len(index.scenes) == 6
    assert index.sensor_counts == {"OHRC": 4, "IIRS": 2}
    assert "TMC-2" not in index.sensor_counts
    assert index.incomplete == [] and index.unpaired_rasters == []


@requires_official
def test_official_xml_fields() -> None:
    index = discover_scenes(OFFICIAL)
    by_id = {s.scene_id: parse_scene_xml(s.xml_path) for s in index.scenes}
    ohrc = by_id["OHRXXD18CHO2436502NNNN25039175231280_V2_1_01"]
    assert ohrc.resolution_m == 0.27
    assert ohrc.sun_azimuth_deg == 19.910131 and ohrc.sun_elevation_deg == 5.103276
    assert ohrc.corners_en["top_left"].easting == 73762.324746
    assert ohrc.corners_en["top_left"].northing == 121889.026404
    iirs = by_id["IIRXXD18CHO2686502NNNN25244140531312_V2_1"]
    assert iirs.resolution_m == 93.74 and iirs.sensor == "IIRS"
    assert iirs.corners_en is None  # not present in that XML
    assert {round(m.resolution_m, 2) for m in by_id.values()} == {0.23, 0.25, 0.26, 0.27, 93.74, 83.14}


@requires_official
def test_official_iirs_is_float32() -> None:
    index = discover_scenes(OFFICIAL)
    for scene in index.by_sensor("IIRS"):
        raster = load_raster(scene.source_path)
        assert raster.data.dtype == np.float32
        assert np.isfinite(raster.data[raster.valid_mask]).all()
