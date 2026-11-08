"""What the metadata and the images say about the illumination of a scene.

The XML gives the Sun azimuth and elevation of the source. It says nothing about the
reference, so the reference's lighting can only be estimated from the image itself.
These numbers describe a scene and explain why it may be hard. Reading them does not
make matching illumination-invariant, and no physical correction is derived from them.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from pydantic import BaseModel

from app.preprocessing.pipeline import shadow_mask

_GRAZING_ELEVATION_DEG = 5.0
_LOW_ELEVATION_DEG = 20.0
_MAX_ANALYSIS_SIDE_PX = 1500  # the statistics do not need full resolution


class IlluminationSummary(BaseModel):
    sun_azimuth_deg: float | None = None
    sun_elevation_deg: float | None = None
    solar_incidence_deg: float | None = None
    regime: str | None = None  # grazing | low | moderate, from the XML elevation
    shadow_length_factor: float | None = None  # shadow length / feature height = 1 / tan(elevation)
    source_shadow_fraction: float | None = None
    reference_shadow_fraction: float | None = None
    source_shading_axis_deg: float | None = None  # image-based, in the source pixel frame
    reference_shading_axis_deg: float | None = None  # image-based, in the reference pixel frame
    shading_axis_strength_source: float | None = None
    shading_axis_strength_reference: float | None = None
    shading_axis_difference_deg: float | None = None  # in the reference frame, once a transform is known
    source_mean_level: float | None = None
    reference_mean_level: float | None = None
    notes: list[str] = []


def shading_axis(display: np.ndarray, valid_mask: np.ndarray) -> tuple[float | None, float]:
    """Dominant axis of image gradients in degrees [0, 180) and its strength in [0, 1].

    Terrain lit from one side has gradients concentrated along the Sun axis. Doubling the
    angle treats opposite gradients (the lit and the shadowed wall of a crater) alike.
    """
    if not valid_mask.any():
        return None, 0.0
    blurred = cv2.GaussianBlur(display.astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    inner = cv2.erode(valid_mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    magnitude = np.hypot(gx, gy) * inner
    angle = np.arctan2(gy, gx)
    total = float(magnitude.sum())
    if total <= 0:
        return None, 0.0
    c = float((magnitude * np.cos(2.0 * angle)).sum()) / total
    s = float((magnitude * np.sin(2.0 * angle)).sum()) / total
    return float((0.5 * math.degrees(math.atan2(s, c))) % 180.0), float(math.hypot(c, s))


def _downscaled(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = min(1.0, _MAX_ANALYSIS_SIDE_PX / max(image.shape))
    if scale == 1.0:
        return image, mask
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    return small, small_mask


def analyse_illumination(
    source_display: np.ndarray,
    source_valid: np.ndarray,
    reference_display: np.ndarray,
    reference_valid: np.ndarray,
    *,
    sun_azimuth_deg: float | None = None,
    sun_elevation_deg: float | None = None,
    solar_incidence_deg: float | None = None,
    shadow_threshold: int = 10,
    shadow_min_area_px: int = 100,
) -> IlluminationSummary:
    summary = IlluminationSummary(
        sun_azimuth_deg=sun_azimuth_deg,
        sun_elevation_deg=sun_elevation_deg,
        solar_incidence_deg=solar_incidence_deg,
    )
    if sun_elevation_deg is not None:
        if sun_elevation_deg < _GRAZING_ELEVATION_DEG:
            summary.regime = "grazing"
            summary.notes.append(
                f"Grazing Sun ({sun_elevation_deg:.1f} deg): long cast shadows move strongly with the Sun, so much of "
                "the image content depends on the illumination rather than on the terrain alone."
            )
        elif sun_elevation_deg < _LOW_ELEVATION_DEG:
            summary.regime = "low"
        else:
            summary.regime = "moderate"
        if sun_elevation_deg > 0.1:
            summary.shadow_length_factor = 1.0 / math.tan(math.radians(sun_elevation_deg))
    summary.notes.append("The XML gives no Sun geometry for the reference; its figures below are image-based estimates.")

    source_small, source_mask = _downscaled(source_display, source_valid)
    reference_small, reference_mask = _downscaled(reference_display, reference_valid)
    for label, image, mask in (("source", source_small, source_mask), ("reference", reference_small, reference_mask)):
        if not mask.any():
            continue
        shadows = shadow_mask(image, mask, shadow_threshold, shadow_min_area_px)
        setattr(summary, f"{label}_shadow_fraction", float(shadows.sum() / mask.sum()))
        setattr(summary, f"{label}_mean_level", float(image[mask].mean()))
        axis, strength = shading_axis(image, mask)
        setattr(summary, f"{label}_shading_axis_deg", axis)
        setattr(summary, f"shading_axis_strength_{label}", strength)
    return summary


def axis_difference_in_reference_frame(
    summary: IlluminationSummary, linear: np.ndarray
) -> float | None:
    """Angle between the two shading axes after rotating the source axis into the reference frame."""
    src, ref = summary.source_shading_axis_deg, summary.reference_shading_axis_deg
    if src is None or ref is None:
        return None
    a = math.radians(src)
    # Gradients transform with the inverse transpose of the linear map.
    gradient = np.linalg.inv(np.asarray(linear, dtype=np.float64)).T @ np.array([math.cos(a), math.sin(a)])
    mapped = math.degrees(math.atan2(gradient[1], gradient[0])) % 180.0
    difference = abs(mapped - ref) % 180.0
    return float(min(difference, 180.0 - difference))
