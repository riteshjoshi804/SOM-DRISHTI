"""RIFT-style matcher (experimental): phase-congruency features with maximum-index-map descriptors.

Follows Li, Hu & Ai, "RIFT: Multi-modal image matching based on radiation-variation
insensitive feature transform" (IEEE TIP 2020): features are detected on phase
congruency rather than intensity, and each is described by a histogram of the
*maximum index map* (which log-Gabor orientation responds most strongly at each
pixel). Both depend on local structure rather than on image contrast, which is the
property a Sun-angle or modality change attacks.

This is a compact in-house implementation, not the authors' code, and it has one
important limit: the descriptor has no rotation handling beyond the reference frame.
It is therefore only meaningful on tiles already rotated into the reference frame,
which the guided search provides. AUTO never selects it by default.
"""
from __future__ import annotations

import numpy as np

from app.config import RiftConfig
from app.matching.base import RIFT, MatcherStatus, PairMatches, empty_pair
from app.matching.tiling import keep_inside_mask
from app.preprocessing.phase_congruency import log_gabor_responses, phase_congruency_from_responses


def rift_status() -> MatcherStatus:
    return MatcherStatus(
        name=RIFT,
        label="RIFT (experimental)",
        available=True,
        experimental=True,
        device="cpu",
        filtering="Mutual nearest neighbour on maximum-index-map descriptors",
        implementation="In-house implementation following Li, Hu & Ai (2020); not the authors' code",
    )


def _detect(pc: np.ndarray, mask: np.ndarray | None, cfg: RiftConfig) -> np.ndarray:
    """Local maxima of phase congruency with non-maximal suppression, strongest first."""
    import cv2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cfg.nms_radius_px + 1, 2 * cfg.nms_radius_px + 1))
    dilated = cv2.dilate(pc, kernel)
    peaks = (pc == dilated) & (pc >= cfg.min_phase_congruency)
    if mask is not None:
        peaks &= mask > 0
    ys, xs = np.nonzero(peaks)
    order = np.argsort(-pc[ys, xs], kind="stable")[: cfg.max_keypoints]
    return np.c_[xs[order], ys[order]].astype(np.float64)


def _max_index_map(responses: np.ndarray) -> np.ndarray:
    """Per pixel, the orientation whose summed log-Gabor amplitude is largest."""
    return np.argmax(responses.sum(axis=1), axis=0).astype(np.int64)


def _describe(points: np.ndarray, mim: np.ndarray, orientations: int, cfg: RiftConfig) -> tuple[np.ndarray, np.ndarray]:
    """Grid histograms of the maximum index map around each point; L2-normalised."""
    h, w = mim.shape
    r = cfg.patch_radius_px
    cells = cfg.grid_cells
    keep = (points[:, 0] >= r) & (points[:, 1] >= r) & (points[:, 0] < w - r) & (points[:, 1] < h - r)
    kept = np.flatnonzero(keep)
    side = 2 * r + 1
    edges = (np.arange(side) * cells // side).astype(np.int64)
    cell_index = edges[:, None] * cells + edges[None, :]
    bins = cells * cells * orientations
    out = np.zeros((len(kept), bins), np.float32)
    for row, i in enumerate(kept):
        x, y = int(points[i, 0]), int(points[i, 1])
        patch = mim[y - r : y + r + 1, x - r : x + r + 1]
        out[row] = np.bincount((cell_index * orientations + patch).ravel(), minlength=bins)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norm, 1e-6), kept


class RiftMatcher:
    name = RIFT
    device = "cpu"

    def __init__(self, cfg: RiftConfig | None = None) -> None:
        self.cfg = cfg or RiftConfig()

    def _features(self, image: np.ndarray, mask: np.ndarray | None):
        responses = log_gabor_responses(image, self.cfg.scales, self.cfg.orientations)
        pc = phase_congruency_from_responses(responses)
        points = _detect(pc, mask, self.cfg)
        if len(points) == 0:
            return points, np.zeros((0, 1), np.float32)
        descriptors, kept = _describe(points, _max_index_map(np.abs(responses)), self.cfg.orientations, self.cfg)
        return points[kept], descriptors

    def match_pair(
        self, image0: np.ndarray, image1: np.ndarray, mask0: np.ndarray | None = None, mask1: np.ndarray | None = None
    ) -> PairMatches:
        p0, d0 = self._features(image0, mask0)
        p1, d1 = self._features(image1, mask1)
        if len(p0) < 2 or len(p1) < 2:
            return empty_pair(len(p0), len(p1))
        similarity = d0 @ d1.T
        forward = similarity.argmax(axis=1)
        backward = similarity.argmax(axis=0)
        mutual = np.flatnonzero(backward[forward] == np.arange(len(p0)))
        scores = similarity[mutual, forward[mutual]].astype(np.float64)
        keep = scores >= self.cfg.min_similarity
        mutual, scores = mutual[keep], scores[keep]
        points0, points1 = p0[mutual], p1[forward[mutual]]
        inside = keep_inside_mask(points0, mask0) & keep_inside_mask(points1, mask1)
        return PairMatches(points0[inside], points1[inside], scores[inside], len(mutual), len(p0), len(p1))
