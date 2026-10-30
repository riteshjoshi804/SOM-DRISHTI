"""Sanity checks on an estimated transformation.

A robust estimator always returns *something*. These checks decide whether
that something is a physically plausible image-to-image mapping.
"""
from __future__ import annotations

import cv2
import numpy as np
from pydantic import BaseModel

from app.config import ValidationConfig
from app.geometry.estimation import apply_transform


class TransformProperties(BaseModel):
    determinant: float | None = None
    scale_major: float | None = None
    scale_minor: float | None = None
    anisotropy: float | None = None
    rotation_deg: float | None = None
    shear_deg: float | None = None
    condition_number: float | None = None
    perspective_ratio: float | None = None
    projected_corners: list[list[float]] | None = None
    footprint_convex: bool | None = None
    fraction_inside_reference: float | None = None


class TransformCheck(BaseModel):
    valid: bool
    reasons: list[str]
    properties: TransformProperties


def _source_corners(width: int, height: int) -> np.ndarray:
    return np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)


def _fraction_inside(quad: np.ndarray, ref_w: int, ref_h: int) -> float:
    """Share of the projected source footprint that falls inside the reference frame."""
    area = abs(cv2.contourArea(quad.astype(np.float32)))
    if area <= 0:
        return 0.0
    frame = np.array([[0, 0], [ref_w, 0], [ref_w, ref_h], [0, ref_h]], dtype=np.float32)
    inter, _ = cv2.intersectConvexConvex(quad.astype(np.float32), frame)
    return float(min(1.0, inter / area))


def check_transform(
    matrix: np.ndarray,
    source_size: tuple[int, int],
    reference_size: tuple[int, int],
    cfg: ValidationConfig,
) -> TransformCheck:
    """Validate a 3x3 source->reference transform. Sizes are (width, height)."""
    reasons: list[str] = []
    props = TransformProperties()
    if matrix is None or matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return TransformCheck(valid=False, reasons=["transformation has non-finite values"], properties=props)
    if abs(matrix[2, 2]) < 1e-12:
        return TransformCheck(valid=False, reasons=["singular transformation"], properties=props)

    m = matrix / matrix[2, 2]
    sw, sh = source_size
    rw, rh = reference_size
    corners = _source_corners(sw, sh)

    # Local linear behaviour, evaluated at the source centre (exact for affine).
    cx, cy = sw / 2.0, sh / 2.0
    w0 = m[2, 0] * cx + m[2, 1] * cy + 1.0
    if abs(w0) < 1e-9:
        return TransformCheck(valid=False, reasons=["singular transformation"], properties=props)
    px, py = (m[0] @ [cx, cy, 1.0]) / w0, (m[1] @ [cx, cy, 1.0]) / w0
    jac = np.array(
        [
            [(m[0, 0] - px * m[2, 0]) / w0, (m[0, 1] - px * m[2, 1]) / w0],
            [(m[1, 0] - py * m[2, 0]) / w0, (m[1, 1] - py * m[2, 1]) / w0],
        ]
    )
    det = float(np.linalg.det(jac))
    sv = np.linalg.svd(jac, compute_uv=False)
    props.determinant = det
    props.scale_major, props.scale_minor = float(sv[0]), float(sv[1])
    props.anisotropy = float(sv[0] / sv[1]) if sv[1] > 1e-12 else float("inf")
    props.condition_number = props.anisotropy
    props.rotation_deg = float(np.degrees(np.arctan2(jac[1, 0], jac[0, 0])))
    col_x, col_y = jac[:, 0], jac[:, 1]
    cos_axes = float(col_x @ col_y / (np.linalg.norm(col_x) * np.linalg.norm(col_y) + 1e-12))
    props.shear_deg = float(abs(90.0 - np.degrees(np.arccos(np.clip(cos_axes, -1, 1)))))

    if sv[1] < 1e-9 or props.condition_number > cfg.max_condition_number:
        reasons.append("singular or near-singular transformation")
    if det <= 0:
        reasons.append("transformation flips orientation (negative determinant)")
    if not (cfg.min_scale <= sv[1] and sv[0] <= cfg.max_scale):
        reasons.append(f"implausible scale ({sv[1]:.3f}..{sv[0]:.3f}, allowed {cfg.min_scale}..{cfg.max_scale})")
    if props.anisotropy > cfg.max_anisotropy:
        reasons.append(f"excessive anisotropy ({props.anisotropy:.2f} > {cfg.max_anisotropy})")
    if props.shear_deg > cfg.max_shear_deg:
        reasons.append(f"excessive shear ({props.shear_deg:.1f} deg > {cfg.max_shear_deg})")

    # Perspective: the homogeneous weight must keep its sign and stay near 1
    # across the whole source image, otherwise the warp folds or explodes.
    weights = m[2, 0] * corners[:, 0] + m[2, 1] * corners[:, 1] + 1.0
    if np.any(weights <= 0):
        reasons.append("excessive perspective distortion (horizon crosses the source image)")
        props.perspective_ratio = float("inf")
    else:
        props.perspective_ratio = float(weights.max() / weights.min())
        if props.perspective_ratio > cfg.max_perspective_ratio:
            reasons.append(
                f"excessive perspective distortion (ratio {props.perspective_ratio:.2f} > {cfg.max_perspective_ratio})"
            )

    quad = apply_transform(m, corners)
    props.projected_corners = quad.tolist()
    if np.all(np.isfinite(quad)):
        props.footprint_convex = bool(cv2.isContourConvex(quad.astype(np.float32)))
        if not props.footprint_convex:
            reasons.append("projected footprint is not convex")
        else:
            props.fraction_inside_reference = _fraction_inside(quad, rw, rh)
            if props.fraction_inside_reference < cfg.min_projected_overlap:
                reasons.append(
                    f"absurd translation: only {props.fraction_inside_reference:.1%} of the source lands inside the reference"
                )
    else:
        reasons.append("projected footprint has non-finite coordinates")

    return TransformCheck(valid=not reasons, reasons=reasons, properties=props)
