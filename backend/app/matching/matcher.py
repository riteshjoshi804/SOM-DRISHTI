"""Descriptor matching: KNN (k=2) + Lowe ratio test, source -> reference."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.config import MatchConfig
from app.matching.sift import Features


@dataclass
class MatchSet:
    source_idx: np.ndarray  # (M,) indices into the source features
    reference_idx: np.ndarray  # (M,) indices into the reference features
    distances: np.ndarray  # (M,) descriptor distance of the best neighbour
    ratios: np.ndarray  # (M,) best / second-best distance
    raw_candidates: int  # source keypoints that received a 2-NN answer
    ratio_filtered: int  # survivors of the ratio test (before de-duplication)
    matcher_used: str

    def __len__(self) -> int:
        return int(self.source_idx.shape[0])


def empty_matches(raw: int = 0, matcher: str = "none") -> MatchSet:
    z = np.zeros(0, np.int64)
    return MatchSet(z, z.copy(), np.zeros(0, np.float32), np.zeros(0, np.float32), raw, 0, matcher)


def _knn(src: np.ndarray, ref: np.ndarray, cfg: MatchConfig) -> tuple[list, str]:
    use_flann = cfg.matcher == "flann" or (
        cfg.matcher == "auto" and float(len(src)) * float(len(ref)) > cfg.bf_max_pairs
    )
    if use_flann:
        matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=cfg.flann_trees), dict(checks=cfg.flann_checks)
        )
        return matcher.knnMatch(src, ref, k=2), "flann_kdtree"
    return cv2.BFMatcher(cv2.NORM_L2).knnMatch(src, ref, k=2), "brute_force_l2"


def match_features(source: Features, reference: Features, cfg: MatchConfig | None = None) -> MatchSet:
    cfg = cfg or MatchConfig()
    if len(source) < 2 or len(reference) < 2:
        return empty_matches()

    knn, used = _knn(source.descriptors, reference.descriptors, cfg)
    pairs = [m for m in knn if len(m) == 2]
    if not pairs:
        return empty_matches(0, used)

    q = np.fromiter((m[0].queryIdx for m in pairs), np.int64, len(pairs))
    t = np.fromiter((m[0].trainIdx for m in pairs), np.int64, len(pairs))
    d1 = np.fromiter((m[0].distance for m in pairs), np.float32, len(pairs))
    d2 = np.fromiter((m[1].distance for m in pairs), np.float32, len(pairs))
    ratio = d1 / np.maximum(d2, 1e-12)

    keep = ratio < cfg.ratio_threshold
    q, t, d1, ratio = q[keep], t[keep], d1[keep], ratio[keep]
    ratio_filtered = int(keep.sum())

    if cfg.unique_reference and len(t):
        # Several source keypoints may pick the same reference keypoint; at most
        # one of them can be right, so keep the closest.
        order = np.lexsort((d1, t))
        first = np.ones(len(order), bool)
        first[1:] = t[order][1:] != t[order][:-1]
        sel = np.sort(order[first])
        q, t, d1, ratio = q[sel], t[sel], d1[sel], ratio[sel]

    return MatchSet(q, t, d1, ratio, len(pairs), ratio_filtered, used)
