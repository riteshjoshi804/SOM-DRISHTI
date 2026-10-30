"""Scientific raster loading.

The array returned in ``RasterData.data`` is exactly what is stored in the
TIFF (uint8 / uint16 / float32). Normalisation for display or matching is done
elsewhere and never written back here.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rasterio
from pydantic import BaseModel
from rasterio.errors import NotGeoreferencedWarning


class GroundControlPoint(BaseModel):
    row: float
    col: float
    x: float
    y: float


class RasterInfo(BaseModel):
    path: str
    width: int
    height: int
    band_count: int
    dtype: str
    nodata: float | None
    crs: str | None
    crs_is_geographic: bool | None
    transform: list[float]
    has_geotransform: bool
    pixel_size: list[float] | None
    bounds: list[float] | None
    gcps: list[GroundControlPoint]
    gcp_crs: str | None
    crs_label_consistent: bool | None
    tags: dict[str, str]


class RasterStats(BaseModel):
    valid_pixels: int
    valid_fraction: float
    nan_count: int
    inf_count: int
    zero_count: int
    negative_count: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    std: float | None
    percentiles: dict[str, float]
    histogram_counts: list[int]
    histogram_edges: list[float]


@dataclass
class RasterData:
    info: RasterInfo
    data: np.ndarray  # original values, 2-D
    valid_mask: np.ndarray  # bool, True = usable pixel


def _crs_label_consistent(crs, transform, width: int, height: int) -> bool | None:
    """False when a geographic CRS is attached to a grid that cannot be degrees."""
    if crs is None or transform.is_identity:
        return None
    if not crs.is_geographic:
        return True
    xs = (transform.c, transform.c + transform.a * width)
    ys = (transform.f, transform.f + transform.e * height)
    return max(abs(v) for v in xs) <= 720 and max(abs(v) for v in ys) <= 90.0001


def read_raster_info(path: str | Path) -> RasterInfo:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as ds:
            t = ds.transform
            gcps, gcp_crs = ds.gcps
            has_gt = not t.is_identity
            return RasterInfo(
                path=str(path),
                width=ds.width,
                height=ds.height,
                band_count=ds.count,
                dtype=str(ds.dtypes[0]),
                nodata=None if ds.nodata is None else float(ds.nodata),
                crs=ds.crs.to_wkt() if ds.crs else None,
                crs_is_geographic=bool(ds.crs.is_geographic) if ds.crs else None,
                transform=[t.a, t.b, t.c, t.d, t.e, t.f],
                has_geotransform=has_gt,
                pixel_size=[abs(t.a), abs(t.e)] if has_gt else None,
                bounds=list(ds.bounds) if has_gt else None,
                gcps=[GroundControlPoint(row=g.row, col=g.col, x=g.x, y=g.y) for g in gcps],
                gcp_crs=gcp_crs.to_wkt() if gcp_crs else None,
                crs_label_consistent=_crs_label_consistent(ds.crs, t, ds.width, ds.height),
                tags={k: str(v) for k, v in ds.tags().items()},
            )


def border_connected_zero_fill(data: np.ndarray, min_area_px: int = 64) -> np.ndarray:
    """Mask of zero-valued regions that touch the image border.

    Rasters without a declared NoData value still carry zero fill around the
    swath. Zeros *inside* the image (hard lunar shadows) are real data and are
    left alone.
    """
    zeros = (data == 0).astype(np.uint8)
    if not zeros.any():
        return np.zeros(data.shape, dtype=bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(zeros, connectivity=8)
    h, w = data.shape
    x, y, bw, bh, area = (stats[:, i] for i in range(5))
    fill = ((x == 0) | (y == 0) | (x + bw == w) | (y + bh == h)) & (area >= min_area_px)
    fill[0] = False  # label 0 is the non-zero background
    return fill[labels]


def compute_valid_mask(data: np.ndarray, nodata: float | None, min_border_fill_area_px: int = 64) -> np.ndarray:
    """True where a pixel holds usable data (not NoData / NaN / Inf / border fill)."""
    valid = np.isfinite(data) if data.dtype.kind == "f" else np.ones(data.shape, dtype=bool)
    if nodata is not None:
        valid &= data != nodata
    valid &= ~border_connected_zero_fill(np.where(valid, data, 0), min_border_fill_area_px)
    return valid


def load_raster(path: str | Path, band: int = 1, min_border_fill_area_px: int = 64) -> RasterData:
    info = read_raster_info(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as ds:
            data = ds.read(band)
    mask = compute_valid_mask(data, info.nodata, min_border_fill_area_px)
    return RasterData(info=info, data=data, valid_mask=mask)


def compute_stats(data: np.ndarray, valid_mask: np.ndarray, bins: int = 64) -> RasterStats:
    is_float = data.dtype.kind == "f"
    values = data[valid_mask].astype(np.float64)
    percentiles: dict[str, float] = {}
    counts: list[int] = []
    edges: list[float] = []
    if values.size:
        for p, v in zip((1, 5, 50, 95, 99), np.percentile(values, (1, 5, 50, 95, 99))):
            percentiles[f"p{p}"] = float(v)
        # Histogram over the 0.5-99.5 % range so a few outliers do not flatten it.
        lo, hi = np.percentile(values, (0.5, 99.5))
        if hi > lo:
            c, e = np.histogram(values, bins=bins, range=(lo, hi))
            counts, edges = c.astype(int).tolist(), e.astype(float).tolist()
    return RasterStats(
        valid_pixels=int(values.size),
        valid_fraction=float(values.size / data.size) if data.size else 0.0,
        nan_count=int(np.isnan(data).sum()) if is_float else 0,
        inf_count=int(np.isinf(data).sum()) if is_float else 0,
        zero_count=int((data == 0).sum()),
        negative_count=int((values < 0).sum()),
        minimum=float(values.min()) if values.size else None,
        maximum=float(values.max()) if values.size else None,
        mean=float(values.mean()) if values.size else None,
        std=float(values.std()) if values.size else None,
        percentiles=percentiles,
        histogram_counts=counts,
        histogram_edges=edges,
    )
