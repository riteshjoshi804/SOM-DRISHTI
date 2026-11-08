"""New image representations, phase congruency, illumination analysis, lunar CRS checks and overlap."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from conftest import XML_TEMPLATE, lunar_like, write_tif

from app.config import GuidedSearchConfig, PreprocessConfig, ValidationConfig
from app.ingestion.raster import load_raster, read_raster_info
from app.metadata.overlap import check_coordinate_system, estimate_overlap
from app.metadata.xml_parser import SceneMetadata, parse_scene_xml
from app.preprocessing.illumination import analyse_illumination, axis_difference_in_reference_frame, shading_axis
from app.preprocessing.phase_congruency import log_gabor_responses, phase_congruency, phase_congruency_from_responses
from app.preprocessing.pipeline import preprocess, shadow_mask

MOON_POLAR = (
    'PROJCS["PolarStereographic Moon",GEOGCS["GCS_Moon",DATUM["D_Moon",SPHEROID["Moon",1737400,0]],'
    'PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433]],PROJECTION["Polar_Stereographic"],'
    'PARAMETER["latitude_of_origin",-90],PARAMETER["central_meridian",0],PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],UNIT["metre",1]]'
)


@pytest.mark.parametrize("mode", ["structural", "phase_congruency", "shadow_aware"])
def test_new_modes_produce_masked_uint8_images(mode: str) -> None:
    data = (lunar_like((300, 200), 5) * 900).astype(np.float32)
    valid = np.ones(data.shape, bool)
    valid[:, :10] = False
    pre = preprocess(data, valid, mode)
    assert pre.image.dtype == np.uint8 and pre.image.shape == data.shape and pre.mode == mode
    assert (pre.image[~valid] == 0).all()
    assert pre.detect_mask[:, :10].max() == 0
    assert pre.image.std() > 1.0


def test_input_data_are_never_modified() -> None:
    data = (lunar_like((120, 120), 6) * 900).astype(np.float32)
    before = data.copy()
    for mode in ("structural", "phase_congruency", "shadow_aware"):
        preprocess(data, np.ones(data.shape, bool), mode)
    np.testing.assert_array_equal(data, before)


def test_structural_map_equalises_dim_and_bright_texture() -> None:
    rng = np.random.default_rng(0)
    texture = cv2.GaussianBlur(rng.standard_normal((200, 400)).astype(np.float32), (0, 0), 2.0)
    data = np.zeros((200, 400), np.float32)
    data[:, :200] = 40 + texture[:, :200] * 15  # dim
    data[:, 200:] = 200 + texture[:, 200:] * 150  # bright, ten times the contrast
    pre = preprocess(data, np.ones(data.shape, bool), "structural", PreprocessConfig(clahe_tile_px=400))
    left, right = pre.image[:, 20:180].mean(), pre.image[:, 220:380].mean()
    assert 0.5 < left / right < 2.0


def test_phase_congruency_marks_features_regardless_of_contrast() -> None:
    weak, strong = np.full((128, 128), 100.0, np.float32), np.full((128, 128), 100.0, np.float32)
    weak[:, 64:] = 110.0
    strong[:, 64:] = 200.0
    pc_weak, pc_strong = phase_congruency(weak), phase_congruency(strong)
    # The step edge stands out at either contrast, and by a similar amount.
    assert pc_weak[64, 60:68].max() > 0.5 and pc_strong[64, 60:68].max() > 0.5
    assert pc_weak[64, 60:68].max() == pytest.approx(pc_strong[64, 60:68].max(), abs=0.15)
    # Away from the edge there is nothing.
    assert pc_strong[64, 10:40].max() < 0.2


def test_phase_congruency_has_a_bounded_range_and_streaming_agrees() -> None:
    image = (lunar_like((128, 160), 7) * 255).astype(np.float32)
    streamed = phase_congruency(image)
    assert streamed.dtype == np.float32 and streamed.min() >= 0.0 and streamed.max() <= 1.0
    np.testing.assert_allclose(streamed, phase_congruency_from_responses(log_gabor_responses(image)), atol=1e-4)


def test_shadow_mask_finds_real_shadows_and_ignores_specks() -> None:
    display = np.full((200, 200), 120, np.uint8)
    display[50:120, 50:120] = 2  # a real shadow
    display[160, 160] = 2  # a dark speck
    mask = shadow_mask(display, np.ones(display.shape, bool), 10, 100)
    assert mask[80, 80] and not mask[160, 160] and not mask[10, 10]


def test_shadow_aware_keeps_features_away_from_shadows() -> None:
    data = (lunar_like((300, 300), 8) * 200 + 50).astype(np.float32)
    data[100:180, 100:180] = 0.5
    pre = preprocess(data, np.ones(data.shape, bool), "shadow_aware")
    assert pre.shadow_mask is not None and pre.shadow_mask[140, 140]
    assert pre.detect_mask[140, 140] == 0
    assert pre.detect_mask[20, 20] == 255
    # The edge of the shadow must not be the strongest thing in the image.
    gx = cv2.Sobel(pre.image.astype(np.float32), cv2.CV_32F, 1, 0)
    assert np.abs(gx[140, 90:110]).max() <= np.abs(gx[20:80, 20:280]).max() * 2.0


def test_shading_axis_of_a_directional_pattern() -> None:
    xx = np.arange(200, dtype=np.float32)
    image = np.tile((127 + 100 * np.sin(xx * 0.3)).astype(np.uint8), (200, 1))
    axis, strength = shading_axis(image, np.ones(image.shape, bool))
    assert min(axis, 180.0 - axis) < 3.0 and strength > 0.8


def test_illumination_summary_labels_reference_values_as_estimates() -> None:
    image = (lunar_like((300, 300), 9) * 250).astype(np.uint8)
    valid = np.ones(image.shape, bool)
    summary = analyse_illumination(image, valid, image, valid, sun_azimuth_deg=45.0, sun_elevation_deg=3.0, solar_incidence_deg=87.0)
    assert summary.regime == "grazing" and summary.shadow_length_factor == pytest.approx(19.08, abs=0.05)
    assert summary.source_shadow_fraction is not None and any("image-based" in n for n in summary.notes)
    assert axis_difference_in_reference_frame(summary, np.eye(2)) == pytest.approx(0.0, abs=1e-6)
    assert analyse_illumination(image, valid, image, valid, sun_elevation_deg=30.0).regime == "moderate"
    assert analyse_illumination(image, valid, image, valid).regime is None


def _reference(tmp_path, crs: str | None, georef: bool = True, label_ok: bool = True):
    data = (lunar_like((400, 400), 3) * 250 + 1).astype(np.uint8)
    path = write_tif(tmp_path / "ref.tif", data, nodata=0, georef=False)
    info = read_raster_info(path)
    return info.model_copy(
        update={
            "crs": crs,
            "has_geotransform": georef,
            "transform": [5.0, 0.0, 0.0, 0.0, -5.0, 2000.0] if georef else info.transform,
            "pixel_size": [5.0, 5.0] if georef else None,
            "crs_is_geographic": False,
            "crs_label_consistent": label_ok,
        }
    )


def _metadata() -> SceneMetadata:
    corners = {
        "top_left": {"latitude": -85.0, "longitude": 10.0},
        "top_right": {"latitude": -85.0, "longitude": 11.0},
        "bottom_left": {"latitude": -86.0, "longitude": 10.0},
        "bottom_right": {"latitude": -86.0, "longitude": 11.0},
    }
    en = {
        "top_left": {"easting": 200.0, "northing": 1800.0},
        "top_right": {"easting": 700.0, "northing": 1800.0},
        "bottom_left": {"easting": 200.0, "northing": 800.0},
        "bottom_right": {"easting": 700.0, "northing": 800.0},
    }
    return SceneMetadata(xml_path="x.xml", corners=corners, corners_en=en, sun_azimuth_deg=30.0, sun_elevation_deg=4.0)


def _source(tmp_path):
    data = (lunar_like((200, 100), 4) * 250 + 1).astype(np.uint8)
    return read_raster_info(write_tif(tmp_path / "src.tif", data))


def test_lunar_crs_is_recognised_from_its_wkt(tmp_path) -> None:
    check = check_coordinate_system(_reference(tmp_path, MOON_POLAR), None)
    assert check.body == "Moon" and check.crs_kind == "projected" and check.units in ("metre", "meter")
    assert check.semi_major_m == pytest.approx(1737400.0)
    assert "Polar" in (check.projection or "")


def test_earth_crs_disables_the_prior(tmp_path) -> None:
    source, reference = _source(tmp_path), _reference(tmp_path, "EPSG:3857")
    reference = reference.model_copy(update={"crs": 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'})
    overlap = estimate_overlap(source, reference, _metadata(), None, GuidedSearchConfig(), ValidationConfig())
    assert overlap.mode == "full_image" and "not lunar" in overlap.reason
    assert overlap.search_bbox_px == [0, 0, reference.width, reference.height]


def test_missing_metadata_or_crs_falls_back_to_the_whole_image(tmp_path) -> None:
    source, reference = _source(tmp_path), _reference(tmp_path, MOON_POLAR)
    overlap = estimate_overlap(source, reference, None, None, GuidedSearchConfig(), ValidationConfig())
    assert overlap.mode == "full_image" and "no XML metadata" in overlap.reason
    no_crs = estimate_overlap(source, _reference(tmp_path, None), _metadata(), None, GuidedSearchConfig(), ValidationConfig())
    assert no_crs.mode == "full_image" and "no CRS" in no_crs.reason
    no_grid = estimate_overlap(source, _reference(tmp_path, MOON_POLAR, georef=False), _metadata(), None, GuidedSearchConfig(), ValidationConfig())
    assert no_grid.mode == "full_image"


def test_implausible_metadata_scale_is_not_used(tmp_path) -> None:
    source, reference = _source(tmp_path), _reference(tmp_path, MOON_POLAR)
    meta = _metadata()
    meta.corners_en = {k: v.model_copy(update={"easting": v.easting * 100.0, "northing": v.northing * 100.0}) for k, v in meta.corners_en.items()}
    overlap = estimate_overlap(source, reference, meta, None, GuidedSearchConfig(), ValidationConfig())
    assert overlap.mode == "full_image" and "implausible scale" in overlap.reason


def test_search_region_is_the_footprint_plus_a_margin(tmp_path) -> None:
    source, reference = _source(tmp_path), _reference(tmp_path, MOON_POLAR)
    valid = np.ones((400, 400), bool)
    overlap = estimate_overlap(source, reference, _metadata(), valid, GuidedSearchConfig(search_radius_m=50.0), ValidationConfig())
    assert overlap.mode == "metadata" and overlap.prior_affine is not None
    x0, y0, x1, y1 = overlap.footprint_bbox_px
    sx0, sy0, sx1, sy1 = overlap.search_bbox_px
    assert sx0 <= x0 and sy0 <= y0 and sx1 >= x1 and sy1 >= y1
    assert overlap.search_radius_px >= 10.0
    assert overlap.sun_elevation_deg == 4.0
    assert 0.0 < overlap.footprint_fraction_in_reference <= 1.0


def test_xml_is_extended_not_replaced(tmp_path) -> None:
    path = tmp_path / "scene.xml"
    path.write_text(XML_TEMPLATE.format(scene_id="OHRXXD18TEST_V1"), encoding="utf-8")
    meta = parse_scene_xml(path)
    assert meta.sun_azimuth_deg == 45.5 and meta.corners["top_left"].latitude == -85.1
    assert load_raster  # the loader stays the single entry point for rasters
