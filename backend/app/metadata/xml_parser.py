"""Chandrayaan-2 product XML reader.

Only values that are physically present in the file are returned; anything
absent is ``None``. The mentor XMLs repeat many tags inside ``<corners>`` and
``<statistics>``; the first occurrence is used and any disagreement between
repeats is reported instead of being silently resolved.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from pydantic import BaseModel

from app.ingestion.indexer import sensor_from_scene_id

# Output corner key -> tag prefix used in the XML.
CORNER_TAGS = {
    "top_left": "topleft",
    "top_right": "topright",
    "bottom_left": "bottomleft",
    "bottom_right": "bottomright",
}

# Sensor-specific block that wraps corners/statistics in the XML.
SENSOR_BLOCKS = {"pan": "OHRC", "ohr": "OHRC", "iir": "IIRS", "tmc": "TMC-2"}


class LatLon(BaseModel):
    latitude: float
    longitude: float


class EastNorth(BaseModel):
    easting: float
    northing: float


class SceneMetadata(BaseModel):
    xml_path: str
    scene_id: str | None = None
    sensor: str | None = None
    sensor_basis: str | None = None

    resolution_m: float | None = None
    sun_azimuth_deg: float | None = None
    sun_elevation_deg: float | None = None
    solar_incidence_angle_deg: float | None = None
    roll_deg: float | None = None
    pitch_deg: float | None = None
    yaw_deg: float | None = None
    spacecraft_altitude_km: float | None = None

    projection: str | None = None
    area: str | None = None
    orbit_limb_direction: str | None = None
    spacecraft_yaw_direction: str | None = None
    reference_used: str | None = None
    date_of_pass: str | None = None
    start_time_utc: str | None = None
    stop_time_utc: str | None = None
    imaging_orbit_number: int | None = None
    data_type: str | None = None
    product_image_width: int | None = None
    product_image_height: int | None = None

    corners: dict[str, LatLon] | None = None
    corners_refined: dict[str, LatLon] | None = None
    corners_en: dict[str, EastNorth] | None = None
    corners_refined_en: dict[str, EastNorth] | None = None
    statistics: dict[str, float] | None = None

    duplicated_tag_count: int = 0
    duplicate_conflicts: list[str] = []
    raw_fields: dict[str, str] = {}


def _to_float(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        return float(text.strip())
    except ValueError:
        return None


def _to_int(text: str | None) -> int | None:
    value = _to_float(text)
    return int(value) if value is not None and float(value).is_integer() else None


def _flatten(root: ET.Element) -> tuple[dict[str, str], int, list[str]]:
    """Leaf elements as ``path -> first text``, plus duplicate bookkeeping."""
    fields: dict[str, str] = {}
    duplicates = 0
    conflicts: list[str] = []

    def walk(node: ET.Element, prefix: str) -> None:
        nonlocal duplicates
        children = list(node)
        path = f"{prefix}/{node.tag}" if prefix else node.tag
        if not children:
            text = (node.text or "").strip()
            if path in fields:
                duplicates += 1
                if fields[path] != text and path not in conflicts:
                    conflicts.append(path)
            else:
                fields[path] = text
            return
        for child in children:
            walk(child, path)

    for child in root:
        walk(child, "")
    return fields, duplicates, conflicts


def _find(fields: dict[str, str], tag: str) -> str | None:
    """First leaf whose tag name equals ``tag`` (top-level match preferred)."""
    if tag in fields:
        return fields[tag] or None
    for path, value in fields.items():
        if path.rsplit("/", 1)[-1] == tag:
            return value or None
    return None


def _corner_latlon(fields: dict[str, str], suffix: str) -> dict[str, LatLon] | None:
    out: dict[str, LatLon] = {}
    for key, tag in CORNER_TAGS.items():
        lat = _to_float(_find(fields, f"{tag}_latitude{suffix}"))
        lon = _to_float(_find(fields, f"{tag}_longitude{suffix}"))
        if lat is None or lon is None:
            return None
        out[key] = LatLon(latitude=lat, longitude=lon)
    return out


def _corner_en(fields: dict[str, str], suffix: str) -> dict[str, EastNorth] | None:
    """Projected corners.

    The XML names these ``*_latitude_en`` / ``*_longitude_en``. They are metres:
    ``longitude_en`` is the easting (X) and ``latitude_en`` the northing (Y).
    This was confirmed against the mentor rasters - the IIRS polar reference
    has its origin at exactly (topleft_longitude_en, topright_latitude_en).
    """
    out: dict[str, EastNorth] = {}
    for key, tag in CORNER_TAGS.items():
        northing = _to_float(_find(fields, f"{tag}_latitude{suffix}"))
        easting = _to_float(_find(fields, f"{tag}_longitude{suffix}"))
        if northing is None or easting is None:
            return None
        out[key] = EastNorth(easting=easting, northing=northing)
    return out


def _sensor(root: ET.Element, scene_id: str | None) -> tuple[str | None, str | None]:
    # The XML carries no explicit instrument-name tag, so the sensor is
    # inferred and the basis of that inference is reported alongside it.
    if scene_id:
        by_id = sensor_from_scene_id(scene_id)
        if by_id != "UNKNOWN":
            return by_id, f"scene ID prefix '{scene_id[:3]}'"
    for child in root:
        if child.tag in SENSOR_BLOCKS:
            return SENSOR_BLOCKS[child.tag], f"XML block <{child.tag}>"
    return None, None


def parse_scene_xml(xml_path: str | Path) -> SceneMetadata:
    path = Path(xml_path)
    root = ET.parse(path).getroot()
    fields, duplicates, conflicts = _flatten(root)

    scene_id = _find(fields, "job_id") or _find(fields, "level0_dataset")
    sensor, basis = _sensor(root, scene_id or path.stem)

    height = _find(fields, "rad_no_of_scan") or _find(fields, "radiance_qube_image_height")
    stats = {
        key: value
        for key, tag in (
            ("minimum", "minimum"),
            ("maximum", "maximum"),
            ("mean", "mean"),
            ("standard_deviation", "standard_deviation"),
        )
        if (value := _to_float(_find(fields, tag))) is not None
    }

    return SceneMetadata(
        xml_path=str(path),
        scene_id=scene_id,
        sensor=sensor,
        sensor_basis=basis,
        resolution_m=_to_float(_find(fields, "Resolution_in_meter")),
        sun_azimuth_deg=_to_float(_find(fields, "Sun_azimuth_in_degree")),
        sun_elevation_deg=_to_float(_find(fields, "Sun_elevation_in_degree")),
        solar_incidence_angle_deg=_to_float(_find(fields, "Solar_incidence_angle_in_degree")),
        roll_deg=_to_float(_find(fields, "Roll_in_degree")),
        pitch_deg=_to_float(_find(fields, "Pitch_in_degree")),
        yaw_deg=_to_float(_find(fields, "Yaw_in_degree")),
        spacecraft_altitude_km=_to_float(_find(fields, "spacecraft_altitude_in_km")),
        projection=_find(fields, "projection"),
        area=_find(fields, "area"),
        orbit_limb_direction=_find(fields, "orbit_limb_direction"),
        spacecraft_yaw_direction=_find(fields, "spacecraft_yaw_direction"),
        reference_used=_find(fields, "ReferenceUsed"),
        date_of_pass=_find(fields, "dop"),
        start_time_utc=_find(fields, "start_time_utc"),
        stop_time_utc=_find(fields, "stop_time_utc"),
        imaging_orbit_number=_to_int(_find(fields, "imaging_orbit_number")),
        data_type=_find(fields, "data_type"),
        product_image_width=_to_int(_find(fields, "image_width")),
        product_image_height=_to_int(height),
        corners=_corner_latlon(fields, ""),
        corners_refined=_corner_latlon(fields, "_refined"),
        corners_en=_corner_en(fields, "_en"),
        corners_refined_en=_corner_en(fields, "_refined_en"),
        statistics=stats or None,
        duplicated_tag_count=duplicates,
        duplicate_conflicts=conflicts,
        raw_fields=fields,
    )
