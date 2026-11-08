"""Search strategy of the learned matchers: anchor, then tiles along the whole image.

SuperPoint + LightGlue and LoFTR are neither rotation invariant nor built for one image
being a small part of another, and the official scenes are rotated up to 118 degrees
and sit kilometres away from their metadata position. So:

1. **Orientation hypotheses** come from the metadata prior (with its 180-degree
   alternative, because one official scene is reversed relative to its GCPs), or from a
   rotation sweep when there is no usable metadata.
2. **Location.** A correlation search that uses no matcher comes first (localisation.py).
   If it finds nothing, or its location leads nowhere, the anchor sweep runs: a source
   tile, rotated into the reference frame, slides over the search region at reduced
   resolution. Inside a correct window the displacements of true matches agree, so a
   translation vote separates right from wrong windows. A candidate only counts once
   another, independent source tile confirms the same offset.
3. **Tile passes.** All tiles are matched at full resolution around the current
   estimate. The first pass tracks the along-track deviation from tile to tile; later
   passes re-centre on the verified model, which raises the yield.
4. **Appearance check.** Every match must also pass a gradient patch comparison that is
   independent of the matcher (see appearance.py), which removes look-alike craters and
   matches in black shadow before any geometry is estimated.
5. **Decoy controls.** The final pass is repeated on deliberately wrong ground. Learned
   matchers always return something, and the expected-chance formula does not account
   for model search. Only the measured null result says how much agreement means nothing.

Window positions are jittered randomly, so bias tied to the position inside a window
cannot masquerade as agreement between tiles.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from app.config import GeometryConfig, GuidedSearchConfig, ValidationConfig
from app.geometry.estimation import apply_transform
from app.geometry.verification import VerifiedGeometry, verify_candidates
from app.matching.appearance import AppearanceSummary, gradient_map, patch_zncc, summarise
from app.matching.base import MatcherResult, PairMatcher, PairMatches, failed_result
from app.matching.localisation import locate_by_correlation
from app.matching.tiling import Patch, apply_h, crop_scaled, rectified_tile
from app.metadata.overlap import OverlapEstimate

ProgressFn = Callable[[str, float], None]


@dataclass
class ImageView:
    """What a matcher sees: an 8-bit representation and where features are allowed."""

    image: np.ndarray  # uint8
    mask: np.ndarray  # uint8, non-zero = usable

    @property
    def size(self) -> tuple[int, int]:
        return self.image.shape[1], self.image.shape[0]


@dataclass
class Hypothesis:
    name: str
    matrix: np.ndarray  # 3x3 prior, source -> reference, original pixels
    mirrored: bool
    from_metadata: bool

    @property
    def linear(self) -> np.ndarray:
        return self.matrix[:2, :2]

    def shifted(self, offset: np.ndarray) -> np.ndarray:
        out = self.matrix.copy()
        out[:2, 2] += offset
        return out


@dataclass
class Peak:
    votes: int
    displacement: np.ndarray
    second: int


@dataclass
class Anchor:
    hypothesis: Hypothesis
    offset: np.ndarray  # reference px, relative to the hypothesis
    centre: np.ndarray  # source px of the tile that found it
    votes: int
    second: int
    confirmations: int
    pairs_used: int
    seconds: float
    log: list[str] = field(default_factory=list)
    method: str = "matcher sweep"


@dataclass
class TilePass:
    points0: np.ndarray  # original source px
    points1: np.ndarray  # original reference px
    confidence: np.ndarray
    candidates: int  # before confidence filtering
    keypoints0: int | None
    keypoints1: int | None
    tiles: list[dict]
    pairs: int
    seconds: float
    appearance: AppearanceSummary | None = None


@dataclass
class GuidedResult:
    result: MatcherResult
    geometry: VerifiedGeometry | None
    hypothesis: Hypothesis | None
    trace: dict
    scale: float = 1.0


def appearance_filter(
    pair: PairMatches, image0: np.ndarray, image1: np.ndarray, half: int, guided: GuidedSearchConfig
) -> tuple[PairMatches, np.ndarray]:
    """The matches whose patches look alike, and the score of every match (NaN = not measurable)."""
    if not guided.appearance_filter or len(pair) == 0:
        return pair, np.full(len(pair), np.nan)
    scores = patch_zncc(gradient_map(image0), gradient_map(image1), pair.points0, pair.points1, half)
    keep = np.nan_to_num(scores, nan=-1.0) >= guided.appearance_min_zncc
    filtered = PairMatches(
        pair.points0[keep], pair.points1[keep], pair.confidence[keep], pair.candidates, pair.keypoints0, pair.keypoints1
    )
    return filtered, scores


def vote_translation(points0: np.ndarray, points1: np.ndarray, bin_px: float) -> Peak:
    """Strongest common displacement ``points1 - points0`` and the runner-up elsewhere."""
    if len(points0) == 0:
        return Peak(0, np.zeros(2), 0)
    d = np.asarray(points1, dtype=np.float64) - np.asarray(points0, dtype=np.float64)
    best, best_d = 0, np.zeros(2)
    # Two half-bin-shifted grids, so a cluster on a bin edge is not split in two.
    for shift in ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)):
        cells = np.floor(d / bin_px + np.array(shift)).astype(np.int64)
        unique, votes = np.unique(cells, axis=0, return_counts=True)
        i = int(np.argmax(votes))
        if votes[i] > best:
            best = int(votes[i])
            best_d = np.median(d[np.all(cells == unique[i], axis=1)], axis=0)
    rest = d[np.linalg.norm(d - best_d, axis=1) > 2.0 * bin_px]
    second = 0
    if len(rest):
        second = int(np.unique(np.floor(rest / bin_px).astype(np.int64), axis=0, return_counts=True)[1].max())
    return Peak(best, best_d, second)


def rotation_matrix_about(width: int, height: int, angle_deg: float, mirror: bool) -> np.ndarray:
    """Source -> source map rotating about the image centre, optionally after a left-right flip."""
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    rot = np.array([[c, -s, cx - c * cx + s * cy], [s, c, cy - s * cx - c * cy], [0.0, 0.0, 1.0]])
    if mirror:
        rot = rot @ np.array([[-1.0, 0.0, width - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return rot


def build_hypotheses(
    overlap: OverlapEstimate, source_size: tuple[int, int], reference_size: tuple[int, int], guided: GuidedSearchConfig
) -> list[Hypothesis]:
    """Orientation priors to try, most likely first."""
    w, h = source_size
    if overlap.mode == "metadata" and overlap.prior_affine is not None:
        prior = np.vstack([np.asarray(overlap.prior_affine, dtype=np.float64), [0.0, 0.0, 1.0]])
        mirrored = bool(overlap.prior_mirrored)
        reversed_prior = prior @ rotation_matrix_about(w, h, 180.0, False)
        return [
            Hypothesis("metadata", prior, mirrored, True),
            Hypothesis("metadata rotated 180 deg", reversed_prior, mirrored, True),
        ]
    # Nothing fixes the orientation: a coarse rotation sweep, placed at the reference centre.
    rw, rh = reference_size
    angles = [0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0]
    out: list[Hypothesis] = []
    for mirror in (False, True):
        for angle in angles:
            frame = rotation_matrix_about(w, h, angle, mirror)
            centre = apply_transform(frame, np.array([[(w - 1) / 2.0, (h - 1) / 2.0]]))[0]
            matrix = frame.copy()
            matrix[:2, 2] += np.array([(rw - 1) / 2.0, (rh - 1) / 2.0]) - centre
            out.append(Hypothesis(f"sweep {angle:g} deg{' mirrored' if mirror else ''}", matrix, mirror, False))
    return out[: guided.max_orientation_hypotheses]


def working_scale(base: float, strip_px: float, min_strip_px: float, limit: float = 2.0) -> float:
    """Scale the matcher works at; narrow strips are enlarged so that enough coarse cells cover them."""
    if strip_px <= 0:
        return base
    return float(np.clip(min_strip_px / strip_px, base, max(limit, base)))


def plan_tile_centres(source_size: tuple[int, int], linear: np.ndarray, tile_ref_px: float, step_ref_px: float) -> np.ndarray:
    """Tile centres (source px) covering the source when viewed at the reference scale."""
    w, h = source_size
    sx, sy = float(np.linalg.norm(linear[:, 0])), float(np.linalg.norm(linear[:, 1]))

    def axis(length: int, scale: float) -> np.ndarray:
        span = length * scale
        # A sliver beyond one tile is not worth a second, almost identical row of tiles.
        if span <= 1.1 * tile_ref_px:
            return np.array([(length - 1) / 2.0])
        count = int(np.ceil((span - tile_ref_px) / step_ref_px)) + 1
        half = tile_ref_px / (2.0 * scale)
        return np.linspace(half, length - 1 - half, count)

    xs, ys = axis(w, sx), axis(h, sy)
    return np.array([[x, y] for y in ys for x in xs], dtype=np.float64)


def texture_score(patch: Patch) -> float:
    """Local structure on usable pixels; used only to decide which tiles to try first."""
    if not patch.mask.any():
        return 0.0
    gx = cv2.Sobel(patch.image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch.image, cv2.CV_32F, 0, 1, ksize=3)
    return float((cv2.magnitude(gx, gy) * (patch.mask > 0)).mean())


def _window_offsets(radius: float, step: float) -> np.ndarray:
    """Offsets on a grid out to ``radius``, nearest first (the prior is most likely right)."""
    count = int(np.ceil(radius / step))
    values = np.arange(-count, count + 1) * step
    grid = np.array([[x, y] for y in values for x in values], dtype=np.float64)
    return grid[np.argsort(np.linalg.norm(grid, axis=1), kind="stable")]


def _centred_window(
    centre: np.ndarray, fov: float, reference_size: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    """A ``fov``-sized window around ``centre``, pushed back inside the raster when it can be."""
    rw, rh = reference_size
    x0 = int(round(centre[0] - fov / 2.0))
    y0 = int(round(centre[1] - fov / 2.0))
    side = int(round(fov))
    x0 = int(np.clip(x0, 0, max(0, rw - side)))
    y0 = int(np.clip(y0, 0, max(0, rh - side)))
    x1, y1 = min(rw, x0 + side), min(rh, y0 + side)
    if x1 - x0 < 32 or y1 - y0 < 32:
        return None
    return x0, y0, x1, y1


def _strict_window(centre: np.ndarray, fov: float, reference_size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """A ``fov``-sized window at exactly ``centre``, or None if it does not fit in the raster."""
    rw, rh = reference_size
    side = int(round(fov))
    x0, y0 = int(round(centre[0] - fov / 2.0)), int(round(centre[1] - fov / 2.0))
    if x0 < 0 or y0 < 0 or x0 + side > rw or y0 + side > rh:
        return None
    return x0, y0, x0 + side, y0 + side


def _window_touches(box: tuple[int, int, int, int], true_centre: np.ndarray, tile_fov: float) -> bool:
    """Would this window contain any part of the tile's true ground (with a safety margin)?"""
    reach = 0.75 * tile_fov  # half the tile's diagonal, near enough for any in-plane rotation
    x0, y0, x1, y1 = box
    return x0 - reach < true_centre[0] < x1 + reach and y0 - reach < true_centre[1] < y1 + reach


def _window_ground(centre: np.ndarray, side: float, reference: ImageView) -> float:
    """Share of a square window that holds valid reference pixels."""
    rw, rh = reference.size
    x0, y0 = int(round(centre[0] - side / 2.0)), int(round(centre[1] - side / 2.0))
    x1, y1 = int(round(centre[0] + side / 2.0)), int(round(centre[1] + side / 2.0))
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, rw), min(y1, rh)
    if cx1 <= cx0 or cy1 <= cy0:
        return 0.0
    return float((reference.mask[cy0:cy1:8, cx0:cx1:8] > 0).sum()) * 64.0 / float((x1 - x0) * (y1 - y0))


def _tile_ground(patch: Patch, prior: np.ndarray, offset: np.ndarray, reference: ImageView) -> float:
    """Share of a tile's valid pixels that would land on valid reference pixels."""
    ys, xs = np.nonzero(patch.mask[::8, ::8])
    if not len(xs):
        return 0.0
    points = apply_transform(prior, patch.map.patch_to_original(np.c_[xs * 8.0, ys * 8.0])) + offset
    rw, rh = reference.size
    px, py = np.rint(points[:, 0]).astype(np.int64), np.rint(points[:, 1]).astype(np.int64)
    inside = (px >= 0) & (py >= 0) & (px < rw) & (py < rh)
    return float((reference.mask[py[inside], px[inside]] > 0).sum() / len(points))


def _spread_first(tiles: list[tuple[float, np.ndarray, Patch]], count: int, axis: int) -> list[tuple[float, np.ndarray, Patch]]:
    """The best-textured tile of each along-track segment first, then the rest in their order.

    A strip can lie mostly off its reference, so the few tiles a sweep can afford must not all
    come from one stretch of it. ``tiles`` must already be sorted by texture.
    """
    if len(tiles) <= 1:
        return tiles
    along = np.array([t[1][axis] for t in tiles])
    span = max(float(along.max() - along.min()), 1e-6)
    segments = np.minimum(((along - along.min()) / span * count).astype(int), count - 1)
    _, first = np.unique(segments, return_index=True)  # the first occurrence is the best textured
    chosen = sorted(first.tolist())
    return [tiles[i] for i in chosen] + [t for i, t in enumerate(tiles) if i not in set(chosen)]


def _implied_offset(
    tile: Patch, window: Patch, peak: Peak, source_centre: np.ndarray, prior: np.ndarray
) -> np.ndarray:
    """Offset from the prior implied by a translation peak, in reference pixels."""
    h, w = tile.image.shape
    centre_tile = np.array([[(w - 1) / 2.0, (h - 1) / 2.0]])
    position = window.map.patch_to_original(centre_tile + peak.displacement)[0]
    return position - apply_transform(prior, source_centre[None])[0]


def sweep_anchor(
    matcher: PairMatcher,
    source: ImageView,
    reference: ImageView,
    hypotheses: list[Hypothesis],
    search_radius_px: float,
    guided: GuidedSearchConfig,
    say: ProgressFn,
) -> tuple[Anchor | None, list[str], float]:
    """First location supported by two independent source tiles, or None. Returns (anchor, log, scale)."""
    t0 = time.perf_counter()
    log: list[str] = []
    pairs = 0
    budget = guided.anchor_max_pairs
    short_side = min(source.size) * float(np.mean(np.linalg.svd(hypotheses[0].linear, compute_uv=False)))
    scale = working_scale(guided.anchor_scale, short_side, 160.0)
    size = guided.anchor_tile_px
    fov = size / scale
    step = max(1.0, fov * guided.anchor_step_fraction)
    bin_px = guided.anchor_vote_bin_px
    log.append(f"anchor: working scale {scale:.3f}, window {fov:.0f} px, step {step:.0f} px, radius {search_radius_px:.0f} px")

    prepared: dict[int, list[tuple[float, np.ndarray, Patch]]] = {}

    def tiles_for(index: int) -> list[tuple[float, np.ndarray, Patch]]:
        if index not in prepared:
            hypothesis = hypotheses[index]
            linear = hypothesis.linear * scale
            centres = plan_tile_centres(source.size, hypothesis.linear, fov, fov)
            built = []
            for centre in centres:
                patch = rectified_tile(source.image, source.mask, linear, centre, (size, size))
                if (patch.mask > 0).mean() >= guided.min_tile_valid_fraction:
                    built.append((texture_score(patch), centre, patch))
            built.sort(key=lambda item: -item[0])
            prepared[index] = _spread_first(built, guided.anchor_max_tiles, 1 if source.size[1] >= source.size[0] else 0)
        return prepared[index]

    offsets = _window_offsets(search_radius_px, step)
    max_rounds = guided.anchor_max_tiles

    def confirm(index: int, centre: np.ndarray, offset: np.ndarray) -> int:
        nonlocal pairs
        hypothesis = hypotheses[index]
        found = 0
        others = [t for t in tiles_for(index) if np.linalg.norm(t[1] - centre) >= 0.75 * fov]
        # A tile can only confirm where the offset puts it on ground of the reference. Pushing its window
        # back inside the raster would compare it with ground it does not show and reject a true location.
        others = [t for t in others if _tile_ground(t[2], hypothesis.matrix, offset, reference) >= guided.anchor_confirm_min_ground]
        if not others:
            log.append(f"  no other tile has ground at offset {np.round(offset).tolist()}")
        for _, other, patch in others[: guided.anchor_confirm_tiles]:
            position = apply_transform(hypothesis.matrix, other[None])[0] + offset
            box = _centred_window(position, fov * 1.25, reference.size)
            if box is None:
                continue
            window = crop_scaled(reference.image, reference.mask, *box, scale=scale)
            if (window.mask > 0).mean() < 0.3:
                continue
            pair = matcher.match_pair(patch.image, window.image, patch.mask, window.mask)
            pairs += 1
            pair, _ = appearance_filter(pair, patch.image, window.image, guided.anchor_patch_radius_px, guided)
            peak = vote_translation(pair.points0, pair.points1, bin_px)
            implied = _implied_offset(patch, window, peak, other, hypothesis.matrix)
            agrees = peak.votes >= max(4, guided.anchor_min_peak // 2) and np.linalg.norm(implied - offset) <= guided.anchor_agreement_px
            log.append(f"  confirm tile at {np.round(other).tolist()}: {peak.votes} votes, offset {np.round(implied).tolist()}{' agrees' if agrees else ''}")
            if agrees:
                found += 1
                if found >= guided.anchor_min_confirmations:
                    return found
        return found

    for round_index in range(max_rounds):
        for h_index, hypothesis in enumerate(hypotheses):
            tiles = tiles_for(h_index)
            if round_index >= len(tiles):
                continue
            _, centre, patch = tiles[round_index]
            predicted = apply_transform(hypothesis.matrix, centre[None])[0]
            seen: set[tuple[int, int, int, int]] = set()
            best = Peak(0, np.zeros(2), 0)
            tried = 0
            for k, offset in enumerate(offsets):
                if pairs >= budget:
                    log.append(f"anchor: budget of {budget} image pairs exhausted")
                    return None, log, scale
                box = _centred_window(predicted + offset, fov, reference.size)
                if box is None or box in seen:
                    continue
                seen.add(box)
                window = crop_scaled(reference.image, reference.mask, *box, scale=scale)
                if (window.mask > 0).mean() < 0.4:
                    continue
                pair = matcher.match_pair(patch.image, window.image, patch.mask, window.mask)
                pairs += 1
                tried += 1
                pair, _ = appearance_filter(pair, patch.image, window.image, guided.anchor_patch_radius_px, guided)
                if k % 20 == 0:
                    say(f"anchor sweep: {hypothesis.name}, tile {round_index + 1}, {pairs} pairs", min(0.95, pairs / budget))
                peak = vote_translation(pair.points0, pair.points1, bin_px)
                if peak.votes > best.votes:
                    best = peak
                if peak.votes < guided.anchor_min_peak or peak.votes < 2 * peak.second:
                    continue
                implied = _implied_offset(patch, window, peak, centre, hypothesis.matrix)
                log.append(f"  candidate [{hypothesis.name}] tile {round_index} at {np.round(centre).tolist()}: {peak.votes} votes (second {peak.second}), offset {np.round(implied).tolist()}")
                confirmations = confirm(h_index, centre, implied)
                if confirmations >= guided.anchor_min_confirmations:
                    seconds = time.perf_counter() - t0
                    log.append(f"anchor found after {pairs} image pairs, {seconds:.0f} s")
                    return (
                        Anchor(hypothesis, implied, centre, peak.votes, peak.second, confirmations, pairs, seconds, log),
                        log,
                        scale,
                    )
            log.append(f"anchor: [{hypothesis.name}] tile {round_index} at {np.round(centre).tolist()}: {tried} windows, best peak {best.votes}")
    log.append(f"anchor: no location confirmed after {pairs} image pairs")
    return None, log, scale


def _interp_track(track: list[tuple[np.ndarray, np.ndarray]], centre: np.ndarray, axis: int) -> np.ndarray:
    """Residual to expect at ``centre`` from the tiles verified so far (linear between neighbours)."""
    if not track:
        return np.zeros(2)
    ordered = sorted(track, key=lambda item: item[0][axis])
    along = np.array([t[0][axis] for t in ordered])
    values = np.array([t[1] for t in ordered])
    return np.array([np.interp(centre[axis], along, values[:, 0]), np.interp(centre[axis], along, values[:, 1])])


def match_tiles(
    matcher: PairMatcher,
    source: ImageView,
    reference: ImageView,
    predict: Callable[[np.ndarray], np.ndarray],
    local_linear: Callable[[np.ndarray], np.ndarray],
    centres: np.ndarray,
    guided: GuidedSearchConfig,
    *,
    scale: float,
    margin_px: float,
    rng: np.random.Generator,
    shifts: np.ndarray | None = None,
    track_from: np.ndarray | None = None,
    say: ProgressFn | None = None,
) -> TilePass:
    """Match every tile around its predicted position and collect correspondences in original pixels.

    With ``track_from`` (a source point) tiles run outward from it and each corrects the
    prediction for the next, which follows along-track deviations larger than the margin.
    ``shifts`` (one reference-pixel vector per tile, NaN = skip) puts each window on known-wrong
    ground, for the null control.
    """
    t0 = time.perf_counter()
    size = guided.fine_tile_px
    fov = size / scale
    axis = 1 if source.size[1] >= source.size[0] else 0
    order = np.arange(len(centres))
    if track_from is not None:
        order = np.argsort(np.linalg.norm(centres - track_from, axis=1), kind="stable")
    all0, all1, conf, all_scores, tiles = [], [], [], [], []
    half = guided.appearance_patch_radius_px
    candidates = 0
    keypoints0: int | None = None
    keypoints1: int | None = None
    track: list[tuple[np.ndarray, np.ndarray]] = []
    pairs = 0

    for n, i in enumerate(order):
        centre = centres[i]
        linear = local_linear(centre) * scale
        patch = rectified_tile(source.image, source.mask, linear, centre, (size, size))
        record: dict = {"tile": int(i), "source_centre": np.round(centre, 1).tolist(), "matches": 0, "votes": 0}
        if (patch.mask > 0).mean() < guided.min_tile_valid_fraction:
            record["skipped"] = "no valid source pixels"
            tiles.append(record)
            continue
        position = predict(centre[None])[0]
        if track_from is not None:
            position = position + _interp_track(track, centre, axis)
        jitter = rng.uniform(-guided.fine_jitter_px, guided.fine_jitter_px, 2) if guided.fine_jitter_px else np.zeros(2)
        if shifts is None:
            # With partial overlap some tiles have no ground here. Pushing their window back inside
            # the raster would only show the matcher terrain that cannot correspond.
            if _window_ground(position, fov + margin_px, reference) < guided.min_tile_ground_fraction:
                record["skipped"] = "no reference ground at the predicted position"
                tiles.append(record)
                continue
            box = _centred_window(position + jitter, fov + 2.0 * margin_px, reference.size)
        else:
            # A decoy window is never pushed back inside the raster: that could put it on the tile's
            # true ground and turn the null control into real matches.
            box = None
            if np.all(np.isfinite(shifts[i])):
                box = _strict_window(position + shifts[i] + jitter, fov + 2.0 * margin_px, reference.size)
                if box is not None and _window_touches(box, position, fov):
                    box = None
        if box is None:
            record["skipped"] = "window outside the reference"
            tiles.append(record)
            continue
        window = crop_scaled(reference.image, reference.mask, *box, scale=scale)
        if (window.mask > 0).mean() < 0.1:
            record["skipped"] = "no valid reference pixels"
            tiles.append(record)
            continue
        pair = matcher.match_pair(patch.image, window.image, patch.mask, window.mask)
        pairs += 1
        candidates += pair.candidates
        if pair.keypoints0 is not None:
            keypoints0 = (keypoints0 or 0) + pair.keypoints0
        if pair.keypoints1 is not None:
            keypoints1 = (keypoints1 or 0) + pair.keypoints1
        record["matches_before_appearance"] = len(pair)
        pair, scores = appearance_filter(pair, patch.image, window.image, half, guided)
        all_scores.append(scores)
        src_pts = patch.map.patch_to_original(pair.points0)
        ref_pts = window.map.patch_to_original(pair.points1)
        all0.append(src_pts)
        all1.append(ref_pts)
        conf.append(pair.confidence)
        record["matches"] = len(pair)
        if len(pair):
            peak = vote_translation(predict(src_pts), ref_pts, guided.tile_vote_bin_px)
            record["votes"] = peak.votes
            record["residual"] = np.round(peak.displacement, 1).tolist()
            if track_from is not None and peak.votes >= guided.tile_min_votes and peak.votes >= 2 * peak.second:
                track.append((centre, peak.displacement))
        tiles.append(record)
        if say is not None:
            say(f"tile {n + 1}/{len(order)}: {len(pair)} matches, {record['votes']} agree", (n + 1) / len(order))

    tiles.sort(key=lambda r: r["tile"])
    if all0:
        points0, points1, confidence = np.concatenate(all0), np.concatenate(all1), np.concatenate(conf)
    else:
        points0, points1, confidence = np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
    appearance = None
    if guided.appearance_filter:
        appearance = summarise(np.concatenate(all_scores) if all_scores else np.zeros(0), guided.appearance_min_zncc, half)
    return TilePass(
        points0, points1, confidence, candidates, keypoints0, keypoints1, tiles, pairs, time.perf_counter() - t0, appearance
    )


def _pass_result(
    matcher: PairMatcher, tile_pass: TilePass, runtime: float, extras: dict | None = None
) -> MatcherResult:
    return MatcherResult(
        matcher_name=matcher.name,
        source_points=tile_pass.points0,
        reference_points=tile_pass.points1,
        confidence_scores=tile_pass.confidence,
        rank=1.0 - tile_pass.confidence,
        candidate_count=tile_pass.candidates,
        accepted_count=len(tile_pass.points0),
        runtime_seconds=runtime,
        device=matcher.device,
        source_keypoints=tile_pass.keypoints0,
        reference_keypoints=tile_pass.keypoints1,
        extras=extras or {},
    )


def _predictor(geometry: VerifiedGeometry) -> tuple[Callable, Callable]:
    if geometry.strip is not None:
        strip = geometry.strip
        return strip.apply, lambda c: strip.jacobian(c[None])[0]
    matrix = geometry.matrix
    return (lambda p: apply_transform(matrix, p)), (lambda c: matrix[:2, :2])


def guided_search(
    matcher: PairMatcher,
    source: ImageView,
    reference: ImageView,
    overlap: OverlapEstimate,
    guided: GuidedSearchConfig,
    geometry_cfg: GeometryConfig,
    validation: ValidationConfig,
    *,
    seed: int = 0,
    say: ProgressFn | None = None,
) -> GuidedResult:
    """Anchor -> tile passes -> verified geometry, for one learned matcher."""
    t0 = time.perf_counter()
    report: ProgressFn = say or (lambda message, fraction: None)
    rng = np.random.default_rng(seed)
    hypotheses = build_hypotheses(overlap, source.size, reference.size, guided)
    radius = overlap.search_radius_px if overlap.mode == "metadata" and overlap.search_radius_px else float(max(reference.size))
    trace: dict = {"search_mode": overlap.mode, "search_radius_px": radius, "hypotheses": [h.name for h in hypotheses], "anchor_log": []}

    def describe(anchor: Anchor) -> dict:
        return {
            "method": anchor.method,
            "hypothesis": anchor.hypothesis.name,
            "offset_from_prior_px": np.round(anchor.offset, 1).tolist(),
            "offset_from_prior_m": float(np.linalg.norm(anchor.offset) * overlap.reference_gsd_m) if overlap.reference_gsd_m else None,
            "votes": anchor.votes,
            "second_peak": anchor.second,
            "confirmations": anchor.confirmations,
            "image_pairs": anchor.pairs_used,
            "seconds": round(anchor.seconds, 1),
        }

    def passes(anchor: Anchor) -> tuple[TilePass | None, VerifiedGeometry | None, float]:
        return _tile_passes(matcher, source, reference, anchor, guided, geometry_cfg, validation, rng, report, trace)

    # The location search shared by every matcher comes first. Anything it gets wrong must not end the
    # run: a location that yields no acceptable geometry only costs its passes, then the matcher searches.
    best_pass: TilePass | None = None
    best_geometry: VerifiedGeometry | None = None
    anchor: Anchor | None = None
    scale = 1.0
    if guided.correlation_search and overlap.mode == "metadata":
        located, located_log = locate_by_correlation(
            source, reference, hypotheses, radius, guided, lambda message, fraction: report(message, 0.05 * fraction)
        )
        trace["anchor_log"].extend(located_log)
        if located is not None:
            anchor = Anchor(
                located.hypothesis, located.offset, located.centre, located.tiles, 0, located.tiles - 1, 0, located.seconds, [],
                f"gradient correlation, {located.tiles} of {located.tested} tiles, mean z {located.best_z:.1f}",
            )  # fmt: skip
            trace["anchor"] = describe(anchor)
            best_pass, best_geometry, scale = passes(anchor)
            if best_geometry is None or not best_geometry.valid or len(best_geometry.inlier_idx) < validation.learned_min_inliers:
                trace["anchor_log"].append("correlation: this location gave no acceptable geometry; searching with the matcher")
                trace["abandoned_anchor"] = trace.pop("anchor")
                trace["abandoned_passes"] = trace.pop("passes")
                best_pass, best_geometry, anchor = None, None, None

    if anchor is None:
        anchor, anchor_log, anchor_scale = sweep_anchor(
            matcher, source, reference, hypotheses, radius, guided,
            lambda message, fraction: report(message, 0.05 + 0.45 * fraction),
        )  # fmt: skip
        trace["anchor_log"].extend(anchor_log)
        trace["anchor_scale"] = anchor_scale
        if anchor is None:
            elapsed = time.perf_counter() - t0
            out = failed_result(
                matcher.name,
                "no location in the search region was confirmed by two independent source tiles",
                device=matcher.device,
                runtime=elapsed,
            )
            out.extras = {"trace": trace}
            return GuidedResult(out, None, None, trace)
        trace["anchor"] = describe(anchor)
        best_pass, best_geometry, scale = passes(anchor)

    elapsed = time.perf_counter() - t0
    extras = {"trace": trace, "appearance": best_pass.appearance.model_dump() if best_pass.appearance else None}
    out = _pass_result(matcher, best_pass, elapsed, extras)
    return GuidedResult(out, best_geometry, anchor.hypothesis, trace, scale)


def _tile_passes(
    matcher: PairMatcher,
    source: ImageView,
    reference: ImageView,
    anchor: Anchor,
    guided: GuidedSearchConfig,
    geometry_cfg: GeometryConfig,
    validation: ValidationConfig,
    rng: np.random.Generator,
    report: ProgressFn,
    trace: dict,
) -> tuple[TilePass | None, VerifiedGeometry | None, float]:
    """Full-resolution passes from an anchor. Returns the best pass, its geometry and the working scale."""
    hypothesis = anchor.hypothesis
    start = hypothesis.shifted(anchor.offset)
    linear = start[:2, :2]
    short_side = min(source.size) * float(np.mean(np.linalg.svd(linear, compute_uv=False)))
    scale = working_scale(guided.fine_scale, short_side, 200.0)
    fov = guided.fine_tile_px / scale
    centres = plan_tile_centres(source.size, linear, fov, fov * guided.fine_step_fraction)
    trace["fine_scale"] = scale
    trace["tile_count"] = len(centres)
    trace["passes"] = []

    predict: Callable = lambda p: apply_transform(start, p)
    local: Callable = lambda c: linear
    best_pass: TilePass | None = None
    best_geometry: VerifiedGeometry | None = None
    size = (source.size, reference.size)

    for number in range(guided.fine_passes):
        first = number == 0
        margin = 2.0 * guided.fine_margin_px if first else float(guided.fine_margin_px)
        tile_pass = match_tiles(
            matcher, source, reference, predict, local, centres, guided,
            scale=scale, margin_px=margin, rng=rng, track_from=anchor.centre if first else None,
            say=lambda message, fraction, n=number: report(f"pass {n + 1}: {message}", 0.5 + 0.45 * (n + fraction) / guided.fine_passes),
        )  # fmt: skip
        geometry, log = verify_candidates(
            tile_pass.points0, tile_pass.points1, 1.0 - tile_pass.confidence, *size,
            geometry_cfg, validation, guided, mirrored=hypothesis.mirrored, start=start,
        )  # fmt: skip
        inliers = len(geometry.inlier_idx) if geometry is not None else 0
        trace["passes"].append(
            {
                "pass": number + 1,
                "margin_px": margin,
                "image_pairs": tile_pass.pairs,
                "candidates": tile_pass.candidates,
                "accepted": len(tile_pass.points0),
                "appearance": tile_pass.appearance.model_dump() if tile_pass.appearance else None,
                "inliers": inliers,
                "model": geometry.kind if geometry is not None else None,
                "valid": geometry.valid if geometry is not None else None,
                "seconds": round(tile_pass.seconds, 1),
                "log": log,
                "tiles": tile_pass.tiles,
            }
        )
        # A valid model always beats an invalid one, however many points the latter claims.
        better = geometry is not None and (
            best_geometry is None or (geometry.valid, inliers) > (best_geometry.valid, len(best_geometry.inlier_idx))
        )
        if better:
            best_pass, best_geometry = tile_pass, geometry
        elif best_pass is None:
            best_pass = tile_pass
        if geometry is None or not geometry.valid:
            break
        predict, local = _predictor(geometry)
        start = geometry.matrix
    return best_pass, best_geometry, scale


@dataclass
class DecoyResult:
    kind: str  # constant: one shift for all tiles; scattered: an independent wrong window per tile
    shift_px: list[float] | None
    tiles_used: int
    candidates: int  # after the same filtering as a real run
    candidates_before_appearance: int
    inliers: int
    valid: bool
    model: str | None
    seconds: float


def _pick_decoy_shift(
    centres: np.ndarray,
    predict: Callable,
    reference: ImageView,
    fov: float,
    rng: np.random.Generator,
    trials: int = 60,
) -> tuple[np.ndarray, int] | None:
    """A constant shift that puts as many tiles as possible on valid but wrong ground."""
    positions = predict(centres)
    rw, rh = reference.size
    valid = reference.mask > 0
    best, best_used = None, 0
    minimum = 2.0 * fov
    maximum = max(rw, rh)
    for _ in range(trials):
        angle = rng.uniform(0.0, 2.0 * np.pi)
        length = rng.uniform(minimum, maximum)
        shift = length * np.array([np.cos(angle), np.sin(angle)])
        used = 0
        for position in positions:
            box = _strict_window(position + shift, fov, (rw, rh))
            if box is None or _window_touches(box, position, fov):
                continue
            x0, y0, x1, y1 = box
            if valid[y0:y1:8, x0:x1:8].mean() >= 0.4:
                used += 1
        if used > best_used:
            best, best_used = shift, used
    if best is None:
        return None
    return best, best_used


def _scattered_shifts(
    centres: np.ndarray,
    predict: Callable,
    reference: ImageView,
    fov: float,
    side: float,
    jitter: float,
    rng: np.random.Generator,
    trials: int = 300,
) -> np.ndarray:
    """An independent wrong-ground window for each tile; NaN where the reference has none."""
    positions = predict(centres)
    rw, rh = reference.size
    valid = reference.mask > 0
    half = side / 2.0 + jitter
    out = np.full((len(positions), 2), np.nan)
    if rw <= 2 * half or rh <= 2 * half:
        return out
    for i, position in enumerate(positions):
        for _ in range(trials):
            centre = np.array([rng.uniform(half, rw - half), rng.uniform(half, rh - half)])
            box = _strict_window(centre, side + 2.0 * jitter, (rw, rh))
            if box is None or _window_touches(box, position, fov):
                continue
            x0, y0, x1, y1 = box
            if valid[y0:y1:8, x0:x1:8].mean() >= 0.4:
                out[i] = centre - position
                break
    return out


def decoy_pass(
    matcher: PairMatcher,
    source: ImageView,
    reference: ImageView,
    geometry: VerifiedGeometry,
    centres: np.ndarray,
    scale: float,
    shifts: np.ndarray,
    kind: str,
    guided: GuidedSearchConfig,
    geometry_cfg: GeometryConfig,
    validation: ValidationConfig,
    rng: np.random.Generator,
    say: ProgressFn | None = None,
) -> tuple[DecoyResult, TilePass]:
    """One tile pass on known-wrong ground, verified exactly as a real pass is."""
    t0 = time.perf_counter()
    predict, local = _predictor(geometry)
    tile_pass = match_tiles(
        matcher, source, reference, predict, local, centres, guided,
        scale=scale, margin_px=float(guided.fine_margin_px), rng=rng, shifts=shifts, say=say,
    )  # fmt: skip
    start = None
    constant = None
    if kind == "constant":
        constant = shifts[np.flatnonzero(np.all(np.isfinite(shifts), axis=1))[0]]
        start = geometry.matrix.copy()
        start[:2, 2] += constant
    decoy, _ = verify_candidates(
        tile_pass.points0, tile_pass.points1, 1.0 - tile_pass.confidence, source.size, reference.size,
        geometry_cfg, validation, guided, mirrored=geometry.mirrored, start=start,
    )  # fmt: skip
    used = sum(1 for t in tile_pass.tiles if "skipped" not in t)
    before = sum(t.get("matches_before_appearance", 0) for t in tile_pass.tiles)
    return (
        DecoyResult(
            kind=kind,
            shift_px=None if constant is None else np.round(constant, 1).tolist(),
            tiles_used=used,
            candidates=len(tile_pass.points0),
            candidates_before_appearance=before,
            inliers=len(decoy.inlier_idx) if decoy is not None else 0,
            valid=bool(decoy.valid) if decoy is not None else False,
            model=decoy.kind if decoy is not None else None,
            seconds=round(time.perf_counter() - t0, 1),
        ),
        tile_pass,
    )


def run_decoy_controls(
    matcher: PairMatcher,
    source: ImageView,
    reference: ImageView,
    outcome: GuidedResult,
    guided: GuidedSearchConfig,
    geometry_cfg: GeometryConfig,
    validation: ValidationConfig,
    *,
    seed: int = 0,
    say: ProgressFn | None = None,
) -> list[DecoyResult]:
    """Repeat the final tile pass on ground known to be wrong and verify it the same way.

    A constant shift tests whether a coherent wrong geometry can emerge, and may grow a strip
    model from the shifted prior as a real run would, making it a worst case. On a reference
    barely larger than the footprint a constant shift leaves most tiles without a window, so
    the scattered form, with its own wrong window for each tile, is used instead.
    """
    geometry = outcome.geometry
    if geometry is None or outcome.hypothesis is None or guided.decoy_controls <= 0:
        return []
    rng = np.random.default_rng(seed + 7919)
    predict, _ = _predictor(geometry)
    scale = outcome.scale
    fov = guided.fine_tile_px / scale
    centres = plan_tile_centres(source.size, geometry.matrix[:2, :2], fov, fov * guided.fine_step_fraction)
    side = fov + 2.0 * guided.fine_margin_px
    results: list[DecoyResult] = []
    for number in range(guided.decoy_controls):
        chosen = _pick_decoy_shift(centres, predict, reference, side + 2.0 * guided.fine_jitter_px, rng)
        if chosen is not None and chosen[1] >= 0.5 * len(centres):
            shifts, kind = np.tile(chosen[0], (len(centres), 1)), "constant"
        else:
            shifts = _scattered_shifts(centres, predict, reference, fov, side, float(guided.fine_jitter_px), rng)
            kind = "scattered"
        if not np.any(np.all(np.isfinite(shifts), axis=1)):
            results.append(DecoyResult(kind, None, 0, 0, 0, 0, False, "no wrong ground available in the reference", 0.0))
            continue
        progress = None
        if say is not None:
            progress = lambda message, fraction, n=number: say(f"decoy {n + 1}: {message}", (n + fraction) / guided.decoy_controls)
        result, _ = decoy_pass(
            matcher, source, reference, geometry, centres, scale, shifts, kind, guided, geometry_cfg, validation, rng, progress
        )
        results.append(result)
    return results


__all__ = [
    "Anchor",
    "DecoyResult",
    "GuidedResult",
    "Hypothesis",
    "ImageView",
    "Peak",
    "apply_h",
    "appearance_filter",
    "build_hypotheses",
    "decoy_pass",
    "guided_search",
    "match_tiles",
    "plan_tile_centres",
    "run_decoy_controls",
    "sweep_anchor",
    "vote_translation",
    "working_scale",
]
