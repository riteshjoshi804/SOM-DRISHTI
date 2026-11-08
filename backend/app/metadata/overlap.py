"""XML/GCP-guided overlap localisation.

Turns the metadata-predicted footprint (``georef.py``) into a *search region*: where
in the reference is it worth looking for a source image, and through which frame?
It is a prior and nothing more. The XMLs state ``ReferenceUsed = System``, and the
official samples are off by kilometres, so the region always carries a safety margin
and every result is still decided by image evidence. When the metadata is missing or
fails these checks, the search falls back to the whole reference.

Nothing here assumes an Earth CRS or an EPSG code. The reference's own WKT is
inspected, and a body other than the Moon disables the prior outright.
"""
from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel
from pyproj import CRS
from pyproj.exceptions import CRSError

from app.config import GuidedSearchConfig, ValidationConfig
from app.geometry.estimation import apply_transform
from app.ingestion.raster import RasterInfo
from app.metadata.georef import MOON_RADIUS_M, PredictedFootprint, predict_footprint, reference_gsd_m
from app.metadata.xml_parser import SceneMetadata

_LUNAR_RADIUS_TOLERANCE = 0.005  # published lunar spheres differ by far less than this


class CoordinateSystemCheck(BaseModel):
    """Whether metadata coordinates and the reference grid can be compared at all."""

    reference_has_crs: bool
    reference_has_geotransform: bool
    datum: str | None = None
    semi_major_m: float | None = None
    body: Literal["Moon", "other", "unknown"] = "unknown"
    crs_kind: Literal["projected", "geographic", "unknown"] = "unknown"
    projection: str | None = None
    latitude_of_origin_deg: float | None = None
    central_meridian_deg: float | None = None
    units: str | None = None
    label_consistent: bool | None = None
    map_coordinate_basis: str | None = None
    compatible: bool = False
    reasons: list[str] = []
    warnings: list[str] = []


class OverlapEstimate(BaseModel):
    mode: Literal["metadata", "full_image"]
    reason: str
    coordinate_system: CoordinateSystemCheck | None = None
    corner_basis: str | None = None
    prior_affine: list[list[float]] | None = None  # 2x3, source px -> reference px
    prior_scale_x: float | None = None
    prior_scale_y: float | None = None
    prior_rotation_deg: float | None = None
    prior_mirrored: bool | None = None
    predicted_corners_px: list[list[float]] | None = None  # source frame corners in reference px
    footprint_bbox_px: list[float] | None = None
    search_bbox_px: list[int] | None = None
    search_radius_px: float | None = None
    reference_gsd_m: float | None = None
    footprint_fraction_in_reference: float | None = None
    footprint_fraction_on_valid_reference: float | None = None
    sun_azimuth_deg: float | None = None
    sun_elevation_deg: float | None = None
    warnings: list[str] = []


def _ellipsoid(crs: CRS) -> tuple[str | None, float | None]:
    try:
        ellipsoid = crs.ellipsoid
        datum = crs.datum.name if crs.datum else None
        return datum, float(ellipsoid.semi_major_metre) if ellipsoid else None
    except (CRSError, AttributeError, ValueError):
        return None, None


def _param(crs: CRS, *names: str) -> float | None:
    operation = crs.coordinate_operation
    if operation is None:
        return None
    for param in operation.params:
        if param.name.lower() in names:
            return float(param.value)
    return None


def check_coordinate_system(reference: RasterInfo, footprint: PredictedFootprint | None) -> CoordinateSystemCheck:
    out = CoordinateSystemCheck(
        reference_has_crs=bool(reference.crs),
        reference_has_geotransform=reference.has_geotransform,
        label_consistent=reference.crs_label_consistent,
        map_coordinate_basis=footprint.map_coordinate_basis if footprint else None,
    )
    reasons: list[str] = []
    if not reference.has_geotransform:
        reasons.append("the reference has no geotransform")
    if not reference.crs:
        reasons.append("the reference declares no CRS, so its body and units cannot be verified")
    else:
        try:
            crs = CRS.from_wkt(reference.crs)
        except CRSError as exc:
            crs = None
            reasons.append(f"the reference CRS cannot be parsed: {exc}")
        if crs is not None:
            out.datum, out.semi_major_m = _ellipsoid(crs)
            if out.semi_major_m is None:
                reasons.append("the reference CRS has no ellipsoid")
            elif abs(out.semi_major_m - MOON_RADIUS_M) / MOON_RADIUS_M <= _LUNAR_RADIUS_TOLERANCE:
                out.body = "Moon"
            else:
                out.body = "other"
                reasons.append(
                    f"the reference CRS is not lunar (semi-major axis {out.semi_major_m:.0f} m, "
                    f"datum {out.datum}); metadata coordinates cannot be used with it"
                )
            out.crs_kind = "projected" if crs.is_projected else "geographic" if crs.is_geographic else "unknown"
            if crs.is_projected and crs.coordinate_operation is not None:
                out.projection = crs.coordinate_operation.method_name
                out.latitude_of_origin_deg = _param(crs, "latitude of natural origin", "latitude of origin", "latitude of standard parallel")
                out.central_meridian_deg = _param(crs, "longitude of natural origin", "longitude of origin")
            axes = crs.axis_info
            out.units = axes[0].unit_name if axes else None

    if reference.crs_label_consistent is False:
        out.warnings.append(
            "The reference CRS is labelled geographic but its geotransform is in metres. The grid is treated as "
            "projected metres, which the XML easting/northing corners match; the label itself is not trusted."
        )
    if footprint is None and not reasons:
        reasons.append("the metadata gives no usable corner coordinates")
    if footprint is not None and footprint.map_coordinate_basis.startswith("spherical"):
        out.warnings.append("Footprint computed from a spherical polar stereographic assumption, not from the XML projected corners.")
    out.reasons = reasons
    out.compatible = not reasons and footprint is not None
    return out


def _fraction_inside(quad: np.ndarray, width: int, height: int) -> float:
    area = abs(cv2.contourArea(quad.astype(np.float32)))
    if area <= 0:
        return 0.0
    frame = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
    shared, _ = cv2.intersectConvexConvex(quad.astype(np.float32), frame)
    return float(min(1.0, shared / area))


def _fraction_on_valid(quad: np.ndarray, valid: np.ndarray) -> float:
    mask = np.zeros(valid.shape, np.uint8)
    cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 1)
    # The polygon may be mostly outside the raster; only its total area is a fair denominator.
    area = abs(cv2.contourArea(quad.astype(np.float32)))
    # The rasterised polygon can hold slightly more pixels than its exact area.
    return float(min(1.0, ((mask > 0) & valid).sum() / area)) if area > 0 else 0.0


def estimate_overlap(
    source: RasterInfo,
    reference: RasterInfo,
    metadata: SceneMetadata | None,
    reference_valid: np.ndarray | None,
    guided: GuidedSearchConfig,
    validation: ValidationConfig,
) -> OverlapEstimate:
    """Search region for ``source`` inside ``reference``; falls back to the whole image."""
    width, height = reference.width, reference.height
    sun = dict(
        sun_azimuth_deg=metadata.sun_azimuth_deg if metadata else None,
        sun_elevation_deg=metadata.sun_elevation_deg if metadata else None,
    )

    def full_image(reason: str, csys: CoordinateSystemCheck | None = None, warnings: list[str] | None = None) -> OverlapEstimate:
        return OverlapEstimate(
            mode="full_image",
            reason=reason,
            coordinate_system=csys,
            search_bbox_px=[0, 0, width, height],
            warnings=warnings or [],
            **sun,
        )

    if metadata is None:
        return full_image("no XML metadata was provided: searching the whole reference")
    try:
        footprint = predict_footprint(source, reference, metadata)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return full_image(f"the metadata footprint could not be computed ({exc}): searching the whole reference")

    csys = check_coordinate_system(reference, footprint)
    if not csys.compatible:
        return full_image(
            "metadata not usable (" + "; ".join(csys.reasons) + "): searching the whole reference", csys, csys.warnings
        )

    affine = np.vstack([np.asarray(footprint.affine), [0.0, 0.0, 1.0]])
    singular = np.linalg.svd(affine[:2, :2], compute_uv=False)
    if singular.min() < validation.min_scale or singular.max() > validation.max_scale:
        return full_image(
            f"the metadata implies an implausible scale ({singular.min():.3f}..{singular.max():.3f}): "
            "searching the whole reference",
            csys,
            csys.warnings,
        )
    if singular.max() / singular.min() > validation.max_anisotropy:
        return full_image(
            f"the metadata implies excessive anisotropy ({singular.max() / singular.min():.2f}): "
            "searching the whole reference",
            csys,
            csys.warnings,
        )

    w, h = source.width, source.height
    corners = apply_transform(affine, np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64))
    inside = _fraction_inside(corners, width, height)
    side = float(np.linalg.norm(corners[1] - corners[0]))
    side = max(side, float(np.linalg.norm(corners[3] - corners[0])))
    gsd = reference_gsd_m(reference)
    radius = guided.search_radius_footprint_fraction * side
    if gsd:
        radius = max(radius, guided.search_radius_m / gsd)

    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    search = [
        int(np.clip(np.floor(x0 - radius), 0, width)),
        int(np.clip(np.floor(y0 - radius), 0, height)),
        int(np.clip(np.ceil(x1 + radius), 0, width)),
        int(np.clip(np.ceil(y1 + radius), 0, height)),
    ]
    if search[2] <= search[0] or search[3] <= search[1]:
        return full_image("the metadata footprint and its safety margin miss the reference: searching the whole reference", csys, csys.warnings)

    return OverlapEstimate(
        mode="metadata",
        reason="metadata footprint accepted as a search prior",
        coordinate_system=csys,
        corner_basis=footprint.corner_basis,
        prior_affine=footprint.affine,
        prior_scale_x=footprint.scale_x,
        prior_scale_y=footprint.scale_y,
        prior_rotation_deg=footprint.rotation_deg,
        prior_mirrored=footprint.mirrored,
        predicted_corners_px=corners.tolist(),
        footprint_bbox_px=[float(x0), float(y0), float(x1), float(y1)],
        search_bbox_px=search,
        search_radius_px=float(radius),
        reference_gsd_m=gsd,
        footprint_fraction_in_reference=inside,
        footprint_fraction_on_valid_reference=_fraction_on_valid(corners, reference_valid) if reference_valid is not None else None,
        warnings=list(csys.warnings) + list(footprint.notes),
        **sun,
    )
