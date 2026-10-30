"""Resample the source into the reference pixel grid. Inputs are never modified."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.crs import CRS
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


def write_registered_geotiff(
    path: str | Path,
    source_data: np.ndarray,
    source_valid: np.ndarray,
    matrix: np.ndarray,
    reference: RasterInfo,
) -> None:
    """Registered source with its *original* values, on the reference grid.

    The reference's CRS and geotransform are copied verbatim, so the output is
    exactly as well (or as badly) georeferenced as the reference itself.
    """
    shape = (reference.height, reference.width)
    is_float = source_data.dtype.kind == "f"
    warped = warp_to_reference(source_data, matrix, shape, cv2.INTER_LINEAR)
    mask = warp_mask_to_reference(source_valid, matrix, shape)
    nodata = float("nan") if is_float else 0
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
    if reference.crs:
        profile["crs"] = CRS.from_wkt(reference.crs)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(warped, 1)
        dst.update_tags(
            SOM_DRISHTI="Source registered to the reference grid by SIFT + robust estimation",
            SOURCE_TO_REFERENCE_MATRIX=" ".join(f"{v:.10g}" for v in matrix.ravel()),
        )
