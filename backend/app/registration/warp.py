"""Resample the source into the reference pixel grid. Inputs are never modified."""
from __future__ import annotations

import warnings
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import Affine

from app.ingestion.raster import RasterInfo


def is_affine(matrix: np.ndarray) -> bool:
    return bool(np.allclose(matrix[2] / matrix[2, 2], [0.0, 0.0, 1.0], atol=1e-12))


def warp_to_reference(
    image: np.ndarray,
    matrix: np.ndarray,
    reference_shape: tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
    border_value: float = 0,
) -> np.ndarray:
    """Warp ``image`` with the 3x3 source->reference ``matrix`` onto a (h, w) canvas."""
    h, w = reference_shape
    m = matrix / matrix[2, 2]
    if is_affine(m):
        return cv2.warpAffine(
            image, m[:2], (w, h), flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value
        )
    return cv2.warpPerspective(
        image, m, (w, h), flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value
    )


def warp_mask_to_reference(valid_mask: np.ndarray, matrix: np.ndarray, reference_shape: tuple[int, int]) -> np.ndarray:
    warped = warp_to_reference(valid_mask.astype(np.uint8), matrix, reference_shape, cv2.INTER_NEAREST, 0)
    return warped.astype(bool)


def source_overlap_mask(
    source_valid: np.ndarray, reference_valid: np.ndarray, matrix: np.ndarray
) -> np.ndarray:
    """Source pixels that are valid and land on a valid reference pixel."""
    h, w = source_valid.shape
    m = matrix / matrix[2, 2]
    flags = cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
    ref = reference_valid.astype(np.uint8)
    if is_affine(m):
        pulled = cv2.warpAffine(ref, m[:2], (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        pulled = cv2.warpPerspective(ref, m, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return source_valid & pulled.astype(bool)


def smooth_maps(evaluate, width: int, height: int, x0: int = 0, y0: int = 0, step: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Dense float32 coordinate maps from a smooth mapping evaluated on a coarse grid."""
    xs = np.unique(np.r_[np.arange(0, width, step), width - 1])
    ys = np.unique(np.r_[np.arange(0, height, step), height - 1])
    gx, gy = np.meshgrid(xs + x0, ys + y0)
    mapped = evaluate(np.c_[gx.ravel(), gy.ravel()].astype(np.float64)).reshape(len(ys), len(xs), 2)
    # Bilinear upsampling at exact pixel positions, one axis at a time.
    fx = np.interp(np.arange(width), xs, np.arange(len(xs), dtype=np.float64))
    fy = np.interp(np.arange(height), ys, np.arange(len(ys), dtype=np.float64))
    i0, j0 = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
    i1, j1 = np.minimum(i0 + 1, len(xs) - 1), np.minimum(j0 + 1, len(ys) - 1)
    wx, wy = (fx - i0)[None, :, None], (fy - j0)[:, None, None]
    top = mapped[j0][:, i0] * (1.0 - wx) + mapped[j0][:, i1] * wx
    bottom = mapped[j1][:, i0] * (1.0 - wx) + mapped[j1][:, i1] * wx
    dense = (top * (1.0 - wy) + bottom * wy).astype(np.float32)
    return dense[:, :, 0], dense[:, :, 1]


def warp_with_inverse(
    image: np.ndarray,
    inverse,
    reference_shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    interpolation: int = cv2.INTER_LINEAR,
    border_value: float = 0,
) -> np.ndarray:
    """Warp through ``inverse`` (reference px -> source px), touching only ``bbox`` of the canvas."""
    h, w = reference_shape
    x0, y0, x1, y1 = int(max(0, bbox[0])), int(max(0, bbox[1])), int(min(w, bbox[2])), int(min(h, bbox[3]))
    out = np.full((h, w), border_value, dtype=image.dtype)
    if x1 <= x0 or y1 <= y0:
        return out
    map_x, map_y = smooth_maps(inverse, x1 - x0, y1 - y0, x0, y0)
    out[y0:y1, x0:x1] = cv2.remap(image, map_x, map_y, interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
    return out


def source_overlap_mask_smooth(source_valid: np.ndarray, reference_valid: np.ndarray, forward) -> np.ndarray:
    """Source pixels that are valid and land on a valid reference pixel, for any smooth model."""
    h, w = source_valid.shape
    map_x, map_y = smooth_maps(forward, w, h)
    pulled = cv2.remap(reference_valid.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return source_valid & pulled.astype(bool)


def write_registered_geotiff(
    path: str | Path,
    source_data: np.ndarray,
    source_valid: np.ndarray,
    matrix: np.ndarray,
    reference: RasterInfo,
    *,
    warp=None,
    warp_mask=None,
    matcher: str = "SIFT",
) -> list[str]:
    """Registered source with its *original* values, on the reference grid.

    The reference's geotransform is copied verbatim, so the output is exactly as well (or as
    badly) georeferenced as the reference itself. Its CRS is copied only when its label agrees
    with the geotransform: an inconsistent label would publish coordinates that mean something
    else. ``warp`` and ``warp_mask`` replace the matrix warp for models that are not a matrix.
    Returns notes about what was left out.
    """
    shape = (reference.height, reference.width)
    is_float = source_data.dtype.kind == "f"
    warped = warp(source_data) if warp is not None else warp_to_reference(source_data, matrix, shape, cv2.INTER_LINEAR)
    mask = warp_mask(source_valid) if warp_mask is not None else warp_mask_to_reference(source_valid, matrix, shape)
    nodata = float("nan") if is_float else 0
    notes: list[str] = []
    warped = np.where(mask, warped, np.nan if is_float else 0).astype(source_data.dtype)

    profile = dict(
        driver="GTiff",
        height=shape[0],
        width=shape[1],
        count=1,
        dtype=str(source_data.dtype),
        nodata=nodata,
        compress="deflate",
        tiled=True,
    )
    if reference.has_geotransform:
        profile["transform"] = Affine(*reference.transform)
    if reference.crs and reference.crs_label_consistent is not False:
        profile["crs"] = CRS.from_wkt(reference.crs)
    elif reference.crs:
        notes.append("The GeoTIFF carries no CRS: the reference CRS label contradicts its geotransform, and none was invented.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)  # an unlabelled grid is intended above
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(warped, 1)
            dst.update_tags(
                SOM_DRISHTI=f"Source registered to the reference grid by {matcher} + robust estimation",
                SOURCE_TO_REFERENCE_MATRIX=" ".join(f"{v:.10g}" for v in matrix.ravel()),
            )
    return notes
