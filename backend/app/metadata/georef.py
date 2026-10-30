"""Metadata-predicted source footprint inside the reference raster.

This is an *independent plausibility check* for the image-based registration:
it answers "where does the XML/GCP metadata say the source should land?".
It is never used to drive feature matching, and it is not ground truth - the
XMLs state ``ReferenceUsed = System``, i.e. system-level pointing knowledge.
"""
from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel
from pyproj import Transformer

from app.ingestion.raster import RasterInfo
from app.metadata.xml_parser import SceneMetadata

MOON_RADIUS_M = 1737400.0
_COORD_TOL_DEG = 1e-5


class PredictedFootprint(BaseModel):
    corner_basis: str
    map_coordinate_basis: str
    xml_corner_names: list[str | None]
    source_corners_px: list[list[float]]
    reference_corners_px: list[list[float]]
    affine: list[list[float]]
    mirrored: bool
    scale_x: float
    scale_y: float
    rotation_deg: float
    corners_inside_reference: int
    notes: list[str]


def reference_gsd_m(ref: RasterInfo) -> float | None:
    """Approximate ground size of one reference pixel in metres."""
    if not ref.pixel_size:
        return None
    if ref.crs_is_geographic and ref.crs_label_consistent:
        return math.radians(ref.pixel_size[1]) * MOON_RADIUS_M  # north-south
    return float(ref.pixel_size[1])


def _array_corner_lonlat(src: RasterInfo, meta: SceneMetadata) -> tuple[list, str] | None:
    """[(col, row, lon, lat)] for the array corners, and how they were obtained."""
    if len(src.gcps) >= 3:
        return [(g.col, g.row, g.x, g.y) for g in src.gcps], "GCPs embedded in the source TIFF"
    corners = meta.corners_refined or meta.corners
    if not corners:
        return None
    w, h = src.width, src.height
    layout = {"top_left": (0, 0), "top_right": (w, 0), "bottom_left": (0, h), "bottom_right": (w, h)}
    pts = [(c, r, corners[k].longitude, corners[k].latitude) for k, (c, r) in layout.items()]
    return pts, "XML corners, ASSUMING XML corner names match the array corners (no GCPs in TIFF)"


def _xml_corner_name(meta: SceneMetadata, lon: float, lat: float) -> str | None:
    for corners in (meta.corners_refined, meta.corners):
        if not corners:
            continue
        for name, c in corners.items():
            if abs(c.latitude - lat) < _COORD_TOL_DEG and abs(c.longitude - lon) < _COORD_TOL_DEG:
                return name
    return None


def predict_footprint(src: RasterInfo, ref: RasterInfo, meta: SceneMetadata) -> PredictedFootprint | None:
    if not ref.has_geotransform:
        return None
    found = _array_corner_lonlat(src, meta)
    if found is None:
        return None
    points, corner_basis = found
    notes: list[str] = []

    geographic_grid = bool(ref.crs_is_geographic and ref.crs_label_consistent)
    if ref.crs_label_consistent is False:
        notes.append(
            "Reference CRS is labelled geographic but its geotransform is in metres; "
            "treated as a projected (metre) grid."
        )

    a, b, c, d, e, f = ref.transform
    inv = np.linalg.inv(np.array([[a, b, c], [d, e, f], [0, 0, 1]], dtype=float))
    en = meta.corners_refined_en or meta.corners_en

    src_px, ref_px, names, bases = [], [], [], set()
    for col, row, lon, lat in points:
        name = _xml_corner_name(meta, lon, lat)
        if geographic_grid:
            centre = c + a * ref.width / 2.0
            x = lon + 360.0 * round((centre - lon) / 360.0)
            y = lat
            bases.add("selenographic lon/lat")
        elif name and en:
            x, y = en[name].easting, en[name].northing
            bases.add("XML easting/northing (*_en)")
        else:
            lat0 = -90 if lat < 0 else 90
            tf = Transformer.from_crs(
                f"+proj=longlat +R={MOON_RADIUS_M} +no_defs",
                f"+proj=stere +lat_0={lat0} +lon_0=0 +k=1 +R={MOON_RADIUS_M} +units=m +no_defs",
                always_xy=True,
            )
            x, y = tf.transform(lon, lat)
            bases.add("spherical polar stereographic computed with pyproj (approximate)")
        px = inv @ np.array([x, y, 1.0])
        src_px.append([float(col), float(row)])
        ref_px.append([float(px[0]), float(px[1])])
        names.append(name)

    sp, rp = np.array(src_px), np.array(ref_px)
    sol, *_ = np.linalg.lstsq(np.c_[sp, np.ones(len(sp))], rp, rcond=None)
    m = sol.T  # 2x3, source px -> reference px
    det = float(np.linalg.det(m[:, :2]))
    if det < 0:
        notes.append("Metadata implies the source array is mirrored relative to the reference.")
    inside = int(sum(0 <= x <= ref.width and 0 <= y <= ref.height for x, y in ref_px))

    return PredictedFootprint(
        corner_basis=corner_basis,
        map_coordinate_basis=" + ".join(sorted(bases)),
        xml_corner_names=names,
        source_corners_px=src_px,
        reference_corners_px=ref_px,
        affine=m.tolist(),
        mirrored=det < 0,
        scale_x=float(np.linalg.norm(m[:, 0])),
        scale_y=float(np.linalg.norm(m[:, 1])),
        rotation_deg=float(np.degrees(np.arctan2(m[1, 0], m[0, 0]))),
        corners_inside_reference=inside,
        notes=notes,
    )
