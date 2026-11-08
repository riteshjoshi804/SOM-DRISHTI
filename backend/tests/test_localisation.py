"""The matcher-independent location search, and the handling of strips lying partly off the reference."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.config import PipelineConfig
from app.evaluation.synthetic import crater_terrain, render
from app.matching import guided as guided_module
from app.matching.base import PairMatches
from app.matching.guided import (
    Hypothesis,
    ImageView,
    _spread_first,
    _tile_ground,
    _window_ground,
    guided_search,
    match_tiles,
    plan_tile_centres,
)
from app.matching.localisation import (
    HypothesisScan,
    Located,
    TilePeak,
    best_cluster,
    locate_by_correlation,
    masked_ncc,
    scan_hypothesis,
    tile_centres,
    tile_extent,
    tile_field_of_view,
)
from app.matching.tiling import Patch, crop_map
from app.metadata.overlap import OverlapEstimate


@pytest.fixture(scope="module")
def field() -> np.ndarray:
    rng = np.random.default_rng(3)
    return cv2.GaussianBlur(rng.standard_normal((200, 240)).astype(np.float32), (0, 0), 2.0)


def _terrain_image(size: tuple[int, int], seed: int, sun: tuple[float, float] = (40.0, 25.0)) -> np.ndarray:
    return np.clip(render(crater_terrain(size, seed), *sun) * 254.0 + 1.0, 1, 255).astype(np.uint8)


def _view(image: np.ndarray) -> ImageView:
    return ImageView(image, (image > 0).astype(np.uint8) * 255)


def _hypothesis(offset_x: float, offset_y: float) -> Hypothesis:
    return Hypothesis("metadata", np.array([[1.0, 0.0, offset_x], [0.0, 1.0, offset_y], [0.0, 0.0, 1.0]]), False, True)


# ------------------------------------------------------------------------------- masked correlation
def test_masked_ncc_finds_a_template_where_it_was_cut(field: np.ndarray) -> None:
    template = field[60:110, 90:150].copy()
    ncc, overlap = masked_ncc(field, np.ones_like(field), template, np.ones_like(template))
    y, x = np.unravel_index(np.nanargmax(ncc), ncc.shape)
    h, w = template.shape
    assert (x - w + 1, y - h + 1) == (90, 60)  # the documented meaning of an index
    assert ncc[y, x] == pytest.approx(1.0, abs=1e-6) and overlap[y, x] == template.size


def test_masked_ncc_sees_a_template_hanging_over_the_edge(field: np.ndarray) -> None:
    """The case that makes a strip mostly off its reference findable at all."""
    h, w = field.shape
    cut = field[40:100, w - 30 : w]  # the visible part of a template 60 wide
    template = np.zeros((60, 60), np.float32)
    template[:, :30] = cut
    mask = np.zeros_like(template)
    mask[:, :30] = 1
    ncc, overlap = masked_ncc(field, np.ones_like(field), template, mask)
    # Placing it on the wrong side with the same overlap must not score: only its valid half can agree.
    y, x = np.unravel_index(np.nanargmax(ncc), ncc.shape)
    assert (x - 59, y - 59) == (w - 30, 40) and ncc[y, x] > 0.999 and overlap[y, x] == 30 * 60


def test_masked_ncc_ignores_whatever_invalid_pixels_contain(field: np.ndarray) -> None:
    template = field[60:110, 90:150].copy()
    mask = np.ones_like(template)
    mask[10:30, 10:40] = 0
    spoiled = template.copy()
    spoiled[mask == 0] = 1000.0
    search_mask = np.ones_like(field)
    search_mask[150:, :] = 0
    spoiled_search = field.copy()
    spoiled_search[search_mask == 0] = -1000.0
    clean, _ = masked_ncc(field, search_mask, template, mask)
    dirty, _ = masked_ncc(spoiled_search, search_mask, spoiled, mask)
    assert np.nanmax(np.abs(np.nan_to_num(clean) - np.nan_to_num(dirty))) < 1e-5


def test_masked_ncc_reports_nothing_where_too_little_overlaps(field: np.ndarray) -> None:
    template = field[60:110, 90:150].copy()
    ncc, overlap = masked_ncc(field, np.ones_like(field), template, np.ones_like(template), min_overlap_px=500.0)
    assert np.isnan(ncc[overlap < 500]).all() and np.isfinite(ncc[overlap >= template.size]).all()


def test_masked_ncc_is_nan_on_flat_patches() -> None:
    flat = np.full((64, 64), 7.0, np.float32)
    ncc, _ = masked_ncc(flat, np.ones_like(flat), flat[:16, :16], np.ones((16, 16), np.float32))
    assert np.isnan(ncc).all()


# ------------------------------------------------------------------------------- tiles and agreement
@pytest.mark.parametrize("angle", [0.0, 7.0, 39.0, 118.0])
def test_tile_extent_really_separates_windows(angle: float) -> None:
    """Two tiles an extent apart must not look at any common source pixel, whatever the rotation."""
    a = np.radians(angle)
    linear = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    fov = 200.0
    extent = tile_extent(linear, fov)
    inverse = np.linalg.inv(linear)
    corners = fov / 2.0 * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
    footprint = corners @ inverse.T
    assert footprint[:, 1].max() - footprint[:, 1].min() == pytest.approx(extent[1])


def test_tile_centres_span_the_strip_and_respect_the_limit() -> None:
    centres = tile_centres((600, 5000), np.eye(2), 512.0, 0.5, 20)
    assert centres[:, 1].min() == pytest.approx(256.0) and centres[:, 1].max() == pytest.approx(5000 - 256.0)
    assert (centres[:, 0] == 300.0).all()
    assert len(tile_centres((600, 50000), np.eye(2), 512.0, 0.5, 12)) == 12


def test_field_of_view_follows_the_strip_within_limits() -> None:
    assert tile_field_of_view((600, 5000), np.eye(2)) == 512.0  # wide strip: the upper limit
    assert tile_field_of_view((300, 5000), np.eye(2)) == 300.0  # its own width
    assert tile_field_of_view((100, 5000), np.eye(2)) == 256.0  # never below what can correlate
    assert tile_field_of_view((420, 1000), np.eye(2)) == pytest.approx(1000 / 3.6)  # room for three tiles


def _scan(*peaks: tuple[int, float, float, float]) -> HypothesisScan:
    centres = np.array([[100.0, y] for y in (100.0, 200.0, 300.0, 700.0, 1300.0)])
    scan = HypothesisScan(_hypothesis(0, 0), centres, np.array([50.0, 400.0]))
    scan.peaks = [TilePeak(t, 0.0, np.array([x, y]), 0.3, z, 1.0) for t, x, y, z in peaks]
    return scan


def test_overlapping_tiles_count_once() -> None:
    """Tiles 0, 1 and 2 share source pixels, so together they are one witness."""
    tiles, _, _, members = best_cluster(_scan((0, 10.0, 10.0, 9.0), (1, 12.0, 8.0, 8.0), (2, 11.0, 9.0, 7.0)), 4.5, 120.0)
    assert tiles == 1 and members[0].tile == 0


def test_independent_tiles_at_one_offset_are_counted() -> None:
    tiles, mean_z, offset, _ = best_cluster(_scan((0, 10.0, 10.0, 9.0), (3, 30.0, -5.0, 7.0), (4, 900.0, 900.0, 20.0)), 4.5, 120.0)
    assert tiles == 2 and mean_z == pytest.approx(8.0) and np.allclose(offset, [20.0, 2.5])


def test_weak_peaks_do_not_count() -> None:
    assert best_cluster(_scan((0, 10.0, 10.0, 4.0), (3, 10.0, 10.0, 4.4)), 4.5, 120.0)[0] == 0


def test_the_right_offset_wins_across_a_changed_sun_without_a_matcher() -> None:
    """Whether a location is accepted is a matter of the limit calibrated on real data, tested below."""
    reference = _terrain_image((1400, 900), 5)
    source = _terrain_image((1400, 900), 5, sun=(80.0, 20.0))[300:1300, 300:600]
    truth = (300.0, 300.0)
    hypothesis = _hypothesis(truth[0] + 180.0, truth[1] - 140.0)  # metadata wrong by (180, -140)
    guided = PipelineConfig().guided
    scan = scan_hypothesis(_view(source), _view(reference), hypothesis, 400.0, guided)
    tiles, _, offset, _ = best_cluster(scan, guided.correlation_min_z, guided.correlation_agreement_px)
    assert tiles >= 2 and np.linalg.norm(offset - np.array([-180.0, 140.0])) < 6.0


def test_location_is_found_when_most_of_the_source_is_off_the_reference() -> None:
    terrain = _terrain_image((1400, 1000), 8)
    reference = terrain[:, 300:]  # the source's left part has no ground here
    source = terrain[100:1300, 100:600]
    hypothesis = _hypothesis(-200.0 + 60.0, 100.0 + 40.0)  # true (-200, 100), metadata wrong by (60, 40)
    found, log = locate_by_correlation(_view(source), _view(reference), [hypothesis], 200.0, PipelineConfig().guided)
    assert found is not None, log
    assert np.linalg.norm(found.offset - np.array([-60.0, -40.0])) < 6.0


def test_unrelated_terrain_gives_no_location() -> None:
    reference = _terrain_image((1400, 900), 11)
    source = _terrain_image((1000, 300), 12)
    found, log = locate_by_correlation(_view(source), _view(reference), [_hypothesis(300.0, 200.0)], 200.0, PipelineConfig().guided)
    assert found is None, log


# --------------------------------------------------------------------- partial overlap in the search
def test_window_ground_measures_the_share_on_valid_pixels() -> None:
    mask = np.zeros((400, 400), np.uint8)
    mask[:, :200] = 255
    view = ImageView(np.zeros((400, 400), np.uint8), mask)
    assert _window_ground(np.array([100.0, 200.0]), 160.0, view) == pytest.approx(1.0, abs=0.02)
    assert _window_ground(np.array([200.0, 200.0]), 160.0, view) == pytest.approx(0.5, abs=0.06)
    assert _window_ground(np.array([300.0, 200.0]), 160.0, view) == 0.0
    assert _window_ground(np.array([-500.0, 200.0]), 160.0, view) == 0.0


def test_tile_ground_follows_the_offset() -> None:
    reference = ImageView(np.ones((300, 300), np.uint8), np.full((300, 300), 255, np.uint8))
    patch = Patch(np.ones((100, 100), np.uint8), np.full((100, 100), 255, np.uint8), crop_map(0.0, 0.0))
    prior = np.eye(3)
    assert _tile_ground(patch, prior, np.array([50.0, 50.0]), reference) == 1.0
    assert _tile_ground(patch, prior, np.array([-1000.0, 0.0]), reference) == 0.0
    half = _tile_ground(patch, prior, np.array([-50.0, 0.0]), reference)
    assert 0.4 < half < 0.6


def test_sweep_tries_every_stretch_of_the_strip_before_repeating_one() -> None:
    def tile(y: float) -> tuple:
        return (0.0, np.array([100.0, y]), None)

    # All the texture is at the far end, but a sweep must also try the other end.
    ordered = [tile(y) for y in (3900.0, 3800.0, 3700.0, 3600.0, 200.0, 1500.0, 2500.0)]
    spread = _spread_first(ordered, 4, 1)
    assert [t[1][1] for t in spread[:4]] == [3900.0, 200.0, 1500.0, 2500.0]
    assert len(spread) == len(ordered)


class _EmptyMatcher:
    name = "EMPTY"
    device = "cpu"

    def __init__(self) -> None:
        self.calls = 0

    def match_pair(self, image0, image1, mask0=None, mask1=None) -> PairMatches:
        self.calls += 1
        return PairMatches(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0), 0, None, None)


def test_every_tile_position_of_a_strip_takes_part_in_the_sweep() -> None:
    """Texture does not tell which tiles have ground, so by default none may be left out."""
    rng = np.random.default_rng(0)
    source = _view(rng.integers(1, 255, (4800, 620), dtype=np.uint8))
    reference = _view(rng.integers(1, 255, (1000, 1000), dtype=np.uint8))
    guided = PipelineConfig().guided
    hypothesis = _hypothesis(200.0, 200.0)
    _, log, scale = guided_module.sweep_anchor(_EmptyMatcher(), source, reference, [hypothesis], 100.0, guided, lambda m, f: None)
    fov = guided.anchor_tile_px / scale
    positions = plan_tile_centres(source.size, hypothesis.linear, fov, fov)
    assert len(positions) == guided.anchor_max_tiles == 8
    swept = [line for line in log if line.startswith("anchor: [metadata] tile")]
    assert len(swept) == len(positions)
    for centre in positions:
        assert any(str(np.round(centre).tolist()) in line for line in swept), centre


def test_tiles_without_reference_ground_are_skipped_not_pushed_onto_other_terrain() -> None:
    source = _view(np.full((2000, 400), 100, np.uint8))
    reference = _view(np.full((1000, 1000), 100, np.uint8))
    guided = PipelineConfig().guided
    guided.fine_tile_px = 320
    matcher = _EmptyMatcher()
    # The lower half of the source maps below the bottom of the reference.
    centres = plan_tile_centres(source.size, np.eye(2), 320.0, 240.0)
    tile_pass = match_tiles(
        matcher, source, reference, lambda p: p + np.array([300.0, 0.0]), lambda c: np.eye(2), centres, guided,
        scale=1.0, margin_px=32.0, rng=np.random.default_rng(0),
    )  # fmt: skip
    skipped = [t for t in tile_pass.tiles if t.get("skipped")]
    assert skipped and all(t["source_centre"][1] > 1000.0 for t in skipped)
    assert matcher.calls == len(tile_pass.tiles) - len(skipped)


def _overlap(radius: float) -> OverlapEstimate:
    return OverlapEstimate(mode="metadata", reason="test", prior_affine=[[1.0, 0.0, 10.0], [0.0, 1.0, 10.0]], prior_mirrored=False, search_radius_px=radius)


def test_a_location_that_leads_nowhere_hands_over_to_the_matcher_sweep(monkeypatch) -> None:
    source = _view(_terrain_image((600, 200), 2))
    reference = _view(_terrain_image((800, 500), 3))
    cfg = PipelineConfig()
    cfg.guided.fine_tile_px = 192
    swept = []

    def false_location(src, ref, hypotheses, radius, guided, say=None):
        return Located(hypotheses[0], np.array([5.0, 5.0]), np.array([100.0, 300.0]), 3, 5, 12.0, 0.0, 0.1), ["correlation: test"]

    def no_sweep(matcher, src, ref, hypotheses, radius, guided, say):
        swept.append(True)
        return None, ["anchor: none"], 1.0

    monkeypatch.setattr(guided_module, "locate_by_correlation", false_location)
    monkeypatch.setattr(guided_module, "sweep_anchor", no_sweep)
    out = guided_search(_EmptyMatcher(), source, reference, _overlap(50.0), cfg.guided, cfg.geometry, cfg.validation)
    assert swept and out.geometry is None
    assert out.trace["abandoned_anchor"]["method"].startswith("gradient correlation")
    assert "anchor" not in out.trace and out.trace["abandoned_passes"]


def test_the_location_search_can_be_turned_off(monkeypatch) -> None:
    source = _view(_terrain_image((600, 200), 2))
    reference = _view(_terrain_image((800, 500), 3))
    cfg = PipelineConfig()
    cfg.guided.correlation_search = False
    cfg.guided.anchor_max_pairs = 2

    def must_not_run(*args, **kwargs):
        raise AssertionError("the location search ran although it was turned off")

    monkeypatch.setattr(guided_module, "locate_by_correlation", must_not_run)
    out = guided_search(_EmptyMatcher(), source, reference, _overlap(50.0), cfg.guided, cfg.geometry, cfg.validation)
    assert out.geometry is None


def test_no_metadata_means_no_correlation_search(monkeypatch) -> None:
    """Without a metadata orientation there are too many orientations for it to be cheap."""
    source = _view(_terrain_image((400, 160), 2))
    reference = _view(_terrain_image((500, 500), 3))
    cfg = PipelineConfig()
    cfg.guided.anchor_max_pairs = 1

    def must_not_run(*args, **kwargs):
        raise AssertionError("the location search ran without a metadata orientation")

    monkeypatch.setattr(guided_module, "locate_by_correlation", must_not_run)
    overlap = OverlapEstimate(mode="full_image", reason="test")
    out = guided_search(_EmptyMatcher(), source, reference, overlap, cfg.guided, cfg.geometry, cfg.validation)
    assert out.geometry is None
