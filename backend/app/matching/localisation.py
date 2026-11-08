"""Finding where a source lies on the reference, without any learned matcher.

Rims and slopes stay in place whichever side of them is lit, so gradient magnitude correlates
across a changed Sun where intensity does not (see appearance.py). Each source tile, rotated into
the reference frame by the metadata orientation, is correlated against its whole search region:

* **By FFT**, so the whole region costs about as much as a few learned image pairs.
* **Over valid pixels only** (masked correlation, Padfield 2012), so a tile may hang over the edge
  of the reference. One official scene lies mostly off its reference, and a search that needs whole
  tiles on the raster cannot find it.
* **Counting only tiles that share no source pixels.** A look-alike can fool one tile. Fooling several
  at one offset is what chance does not do, so only that counts.

This supplies a starting point and nothing else. If it finds nothing, the matcher's own sweep runs.
If it is wrong, the tile passes and the verification reject it. No reported figure comes from here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy import fft as sfft

from app.config import GuidedSearchConfig
from app.geometry.estimation import apply_transform
from app.matching.appearance import gradient_map
from app.matching.tiling import crop_scaled, erode, rectified_tile

if TYPE_CHECKING:
    from app.matching.guided import Hypothesis, ImageView

ProgressFn = Callable[[str, float], None]

_FFT_WORKERS = 4
_MIN_FOV_PX = 256.0
_MAX_FOV_PX = 512.0
_PEAKS_PER_MAP = 3
_NMS_PX = 16


@dataclass
class TilePeak:
    tile: int
    angle_deg: float
    offset: np.ndarray  # reference px, relative to the hypothesis
    ncc: float
    z: float
    overlap: float  # share of the tile's valid pixels on valid reference pixels


@dataclass
class Located:
    hypothesis: "Hypothesis"
    offset: np.ndarray  # reference px, relative to the hypothesis
    centre: np.ndarray  # source px of the strongest tile that agrees
    tiles: int  # tiles sharing no source pixels that agree on the offset
    tested: int
    best_z: float
    angle_deg: float
    seconds: float


@dataclass
class HypothesisScan:
    hypothesis: "Hypothesis"
    centres: np.ndarray
    extent: np.ndarray  # source px; tiles further apart than this share no pixel
    peaks: list[TilePeak] = field(default_factory=list)
    tested: int = 0


class _MaskedSearch:
    """The reference side of a masked correlation, whose transforms are shared by every template."""

    def __init__(self, image: np.ndarray, mask: np.ndarray, template_shape: tuple[int, int]) -> None:
        self.shape_in = image.shape
        self.template_shape = template_shape
        height = image.shape[0] + template_shape[0] - 1
        width = image.shape[1] + template_shape[1] - 1
        self.shape = (sfft.next_fast_len(height, real=True), sfft.next_fast_len(width, real=True))
        self.out = (height, width)
        valid = (mask > 0).astype(np.float64)
        values = image.astype(np.float64) * valid
        self.f_values = sfft.rfft2(values, self.shape, workers=_FFT_WORKERS)
        self.f_squares = sfft.rfft2(values * values, self.shape, workers=_FFT_WORKERS)
        self.f_mask = sfft.rfft2(valid, self.shape, workers=_FFT_WORKERS)

    def correlate(self, template: np.ndarray, mask: np.ndarray, min_overlap_px: float) -> tuple[np.ndarray, np.ndarray]:
        """(ncc, overlap in pixels) per placement; NaN where too little overlaps or either side is flat.

        Index (y, x) puts the template's top-left corner at search pixel (x - tw + 1, y - th + 1),
        so placements hanging over any edge of the search image are included.
        """
        valid = (mask[::-1, ::-1] > 0).astype(np.float64)
        values = template[::-1, ::-1].astype(np.float64) * valid
        f_values = sfft.rfft2(values, self.shape, workers=_FFT_WORKERS)
        f_squares = sfft.rfft2(values * values, self.shape, workers=_FFT_WORKERS)
        f_mask = sfft.rfft2(valid, self.shape, workers=_FFT_WORKERS)

        def inverse(spectrum: np.ndarray) -> np.ndarray:
            return sfft.irfft2(spectrum, self.shape, workers=_FFT_WORKERS)[: self.out[0], : self.out[1]]

        overlap = np.rint(inverse(self.f_mask * f_mask))
        with np.errstate(divide="ignore", invalid="ignore"):
            sum_s = inverse(self.f_values * f_mask)
            sum_t = inverse(self.f_mask * f_values)
            cov = inverse(self.f_values * f_values) - sum_s * sum_t / overlap
            var_s = inverse(self.f_squares * f_mask) - sum_s * sum_s / overlap
            var_t = inverse(self.f_mask * f_squares) - sum_t * sum_t / overlap
            ncc = cov / np.sqrt(var_s * var_t)
        ncc[(overlap < min_overlap_px) | (var_s <= 1e-6 * overlap) | (var_t <= 1e-6 * overlap)] = np.nan
        return np.clip(ncc, -1.0, 1.0), overlap


def masked_ncc(
    search: np.ndarray, search_mask: np.ndarray, template: np.ndarray, template_mask: np.ndarray, min_overlap_px: float = 16.0
) -> tuple[np.ndarray, np.ndarray]:
    """Normalised cross-correlation over the pixels valid in both images, at every placement."""
    return _MaskedSearch(search, search_mask, template.shape).correlate(template, template_mask, min_overlap_px)


def _strongest(score: np.ndarray, count: int, radius: int) -> list[tuple[int, int]]:
    """Up to ``count`` strongest positions, at least ``radius`` apart. ``score`` is overwritten."""
    found = []
    for _ in range(count):
        y, x = np.unravel_index(np.argmax(score), score.shape)
        if not np.isfinite(score[y, x]) or score[y, x] == -np.inf:
            break
        found.append((int(y), int(x)))
        score[max(0, y - radius) : y + radius + 1, max(0, x - radius) : x + radius + 1] = -np.inf
    return found


def tile_field_of_view(source_size: tuple[int, int], linear: np.ndarray) -> float:
    """Side of a tile in reference pixels: the strip's width, within limits, with room for three tiles."""
    w, h = source_size
    sx, sy = float(np.linalg.norm(linear[:, 0])), float(np.linalg.norm(linear[:, 1]))
    short_side, long_side = min(w * sx, h * sy), max(w * sx, h * sy)
    return float(np.clip(min(short_side, long_side / 3.6), min(_MIN_FOV_PX, long_side / 3.6), _MAX_FOV_PX))


def tile_extent(linear: np.ndarray, fov: float) -> np.ndarray:
    """Source-pixel extent (x, y) of a square reference-frame window; centres this far apart share no pixel."""
    return fov * np.abs(np.linalg.inv(linear)).sum(axis=1)


def tile_centres(source_size: tuple[int, int], linear: np.ndarray, fov: float, step_fraction: float, max_tiles: int) -> np.ndarray:
    """Tile centres (source px), ``step_fraction`` of a tile's extent apart, spread over the whole image."""

    def axis(length: int, extent: float) -> np.ndarray:
        if length <= extent:
            return np.array([length / 2.0])
        count = int(np.floor((length - extent) / (extent * step_fraction))) + 1
        return np.linspace(extent / 2.0, length - extent / 2.0, count) if count > 1 else np.array([length / 2.0])

    extent = tile_extent(linear, fov)
    xs, ys = axis(source_size[0], float(extent[0])), axis(source_size[1], float(extent[1]))
    centres = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)
    if len(centres) > max_tiles:
        centres = centres[np.linspace(0, len(centres) - 1, max_tiles).round().astype(int)]
    return centres


def scan_hypothesis(
    source: "ImageView",
    reference: "ImageView",
    hypothesis: "Hypothesis",
    radius_px: float,
    guided: GuidedSearchConfig,
) -> HypothesisScan:
    """Correlate every independent tile over its search region; keep each map's strongest peaks."""
    linear = hypothesis.linear
    fov = tile_field_of_view(source.size, linear)
    scale = min(1.0, guided.correlation_tile_px / fov)
    size = int(round(fov * scale))
    centres = tile_centres(source.size, linear, fov, guided.correlation_step_fraction, guided.correlation_max_tiles)
    scan = HypothesisScan(hypothesis, centres, tile_extent(linear, fov))
    half = (size - 1) / 2.0
    for index, centre in enumerate(centres):
        predicted = apply_transform(hypothesis.matrix, centre[None])[0]
        reach = radius_px + fov / 2.0
        box = [predicted[0] - reach, predicted[1] - reach, predicted[0] + reach, predicted[1] + reach]
        window = crop_scaled(reference.image, reference.mask, *[int(round(v)) for v in box], scale=scale)
        if window.image.size == 0 or min(window.image.shape) < 8 or not (window.mask > 0).any():
            continue
        grad = gradient_map(window.image)
        search = _MaskedSearch(grad, erode(window.mask, 3), (size, size))
        measured = False
        for angle in guided.correlation_angles_deg:
            a = np.radians(angle)
            turned = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ linear * scale
            tile = rectified_tile(source.image, source.mask, turned, centre, (size, size))
            mask = erode(tile.mask, 3)
            usable = float((mask > 0).sum())
            if usable < guided.correlation_min_valid_fraction * size * size:
                continue
            ncc, overlap = search.correlate(gradient_map(tile.image), mask, guided.correlation_min_overlap * usable)
            # Noise on a correlation shrinks with the square root of the pixels compared, so this keeps
            # a tile hanging mostly over the edge from out-scoring one that is fully compared.
            score = ncc * np.sqrt(overlap / usable)
            y_idx, x_idx = np.indices(score.shape, sparse=True)
            positions = window.map.patch_to_original(np.array([[0.0, 0.0], [1.0, 1.0]]))
            step, origin = positions[1] - positions[0], positions[0]
            cx = origin[0] + (x_idx - size + 1 + half) * step[0]
            cy = origin[1] + (y_idx - size + 1 + half) * step[1]
            score[(np.abs(cx - predicted[0]) > radius_px) | (np.abs(cy - predicted[1]) > radius_px)] = np.nan
            finite = score[np.isfinite(score)]
            if finite.size < 16:
                continue
            measured = True
            median = float(np.median(finite))
            spread = 1.4826 * float(np.median(np.abs(finite - median))) + 1e-9
            raw = np.nan_to_num(score, nan=-np.inf)
            for y, x in _strongest(raw.copy(), _PEAKS_PER_MAP, _NMS_PX):
                offset = np.array([cx[0, x], cy[y, 0]]) - predicted
                scan.peaks.append(TilePeak(index, float(angle), offset, float(ncc[y, x]), (float(score[y, x]) - median) / spread, float(overlap[y, x] / usable)))
        if measured:
            scan.tested += 1
    return scan


def _independent(members: list[TilePeak], centres: np.ndarray, extent: np.ndarray) -> list[TilePeak]:
    """The strongest peaks whose tiles share no source pixel."""
    kept: list[TilePeak] = []
    for peak in sorted(members, key=lambda m: -m.z):
        if all(np.any(np.abs(centres[k.tile] - centres[peak.tile]) >= extent) for k in kept):
            kept.append(peak)
    return kept


def best_cluster(scan: HypothesisScan, min_z: float, agreement_px: float) -> tuple[int, float, np.ndarray, list[TilePeak]]:
    """The offset most independent tiles agree on: (tiles, mean z of its three strongest, offset, their peaks)."""
    strong = [p for p in scan.peaks if p.z >= min_z]
    best: tuple[int, float, np.ndarray, list[TilePeak]] = (0, 0.0, np.zeros(2), [])
    for seed in strong:
        per_tile: dict[int, TilePeak] = {}
        for p in strong:
            if np.linalg.norm(p.offset - seed.offset) <= agreement_px and (p.tile not in per_tile or p.z > per_tile[p.tile].z):
                per_tile[p.tile] = p
        members = _independent(list(per_tile.values()), scan.centres, scan.extent)
        top = sorted((m.z for m in members), reverse=True)[:3]
        key = (len(members), float(np.mean(top)))
        if key > best[:2]:
            best = (key[0], key[1], np.median([m.offset for m in members], axis=0), members)
    return best


def locate_by_correlation(
    source: "ImageView",
    reference: "ImageView",
    hypotheses: list["Hypothesis"],
    radius_px: float,
    guided: GuidedSearchConfig,
    say: ProgressFn | None = None,
) -> tuple[Located | None, list[str]]:
    """The location several independent tiles agree on, under the best orientation, or None."""
    t0 = time.perf_counter()
    log: list[str] = []
    found: Located | None = None
    for number, hypothesis in enumerate(hypotheses):
        if say is not None:
            say(f"gradient correlation: {hypothesis.name}", number / len(hypotheses))
        scan = scan_hypothesis(source, reference, hypothesis, radius_px, guided)
        tiles, mean_z, offset, members = best_cluster(scan, guided.correlation_min_z, guided.correlation_agreement_px)
        log.append(
            f"correlation: [{hypothesis.name}] {scan.tested} independent tiles, best agreement {tiles} tiles, "
            f"mean z {mean_z:.1f}, offset {np.round(offset).tolist()}"
        )
        if tiles < guided.correlation_min_tiles or mean_z < guided.correlation_min_mean_z:
            continue
        if found is None or (tiles, mean_z) > (found.tiles, found.best_z):
            strongest = max(members, key=lambda p: p.z)
            found = Located(hypothesis, offset, scan.centres[strongest.tile], tiles, scan.tested, mean_z, strongest.angle_deg, 0.0)
    seconds = time.perf_counter() - t0
    if found is not None:
        found.seconds = seconds
    log.append(f"correlation: {'location found' if found else 'no location agreed'} in {seconds:.1f} s")
    return found, log
