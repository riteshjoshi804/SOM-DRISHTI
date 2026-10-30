"""Tiled SIFT extraction.

Large lunar rasters are processed tile by tile. This bounds memory (SIFT
up-samples its first octave), lets tiles run in parallel, and - because the
feature budget is per tile - spreads keypoints over the whole image instead
of letting a few high-contrast crater rims take the entire budget.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

from app.config import SiftConfig


@dataclass
class Features:
    points: np.ndarray  # (N, 2) float32, x/y in full-image pixels
    sizes: np.ndarray  # (N,) keypoint diameter
    angles: np.ndarray  # (N,) orientation in degrees
    responses: np.ndarray  # (N,)
    descriptors: np.ndarray  # (N, 128) float32

    def __len__(self) -> int:
        return int(self.points.shape[0])


def empty_features() -> Features:
    return Features(
        points=np.zeros((0, 2), np.float32),
        sizes=np.zeros(0, np.float32),
        angles=np.zeros(0, np.float32),
        responses=np.zeros(0, np.float32),
        descriptors=np.zeros((0, 128), np.float32),
    )


def to_root_sift(descriptors: np.ndarray) -> np.ndarray:
    """RootSIFT (Arandjelovic & Zisserman 2012): L1-normalise then square-root.

    Euclidean distance on the result equals the Hellinger kernel on the
    original histograms, which is more tolerant of contrast changes.
    """
    if descriptors.size == 0:
        return descriptors
    d = descriptors / (np.abs(descriptors).sum(axis=1, keepdims=True) + 1e-7)
    return np.sqrt(d, out=d)


def _tile_origins(length: int, tile: int) -> list[tuple[int, int]]:
    """Split ``length`` into near-equal spans no longer than ``tile``."""
    count = max(1, int(np.ceil(length / tile)))
    edges = np.linspace(0, length, count + 1).round().astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(count)]


def _extract_tile(image: np.ndarray, mask: np.ndarray, x0: int, x1: int, y0: int, y1: int, cfg: SiftConfig):
    h, w = image.shape
    pad = cfg.tile_overlap
    px0, py0, px1, py1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    # Detection is limited to the tile core; the padding only provides context
    # for descriptors near the tile edge, so no keypoint is produced twice.
    core = np.zeros((py1 - py0, px1 - px0), np.uint8)
    core[y0 - py0 : y1 - py0, x0 - px0 : x1 - px0] = mask[y0:y1, x0:x1]
    if not core.any():
        return None
    sift = cv2.SIFT_create(
        nfeatures=cfg.nfeatures,
        nOctaveLayers=cfg.n_octave_layers,
        contrastThreshold=cfg.contrast_threshold,
        edgeThreshold=cfg.edge_threshold,
        sigma=cfg.sigma,
    )
    keypoints, descriptors = sift.detectAndCompute(image[py0:py1, px0:px1], core)
    if not keypoints:
        return None
    pts = np.array([k.pt for k in keypoints], np.float32) + np.array([px0, py0], np.float32)
    return (
        pts,
        np.array([k.size for k in keypoints], np.float32),
        np.array([k.angle for k in keypoints], np.float32),
        np.array([k.response for k in keypoints], np.float32),
        descriptors,
    )


def extract_sift(image: np.ndarray, detect_mask: np.ndarray, cfg: SiftConfig | None = None) -> Features:
    """Detect SIFT keypoints inside ``detect_mask`` (uint8, non-zero = allowed)."""
    cfg = cfg or SiftConfig()
    if image.dtype != np.uint8:
        raise ValueError("SIFT input must be a uint8 image (run preprocessing first)")
    h, w = image.shape
    jobs = [(x0, x1, y0, y1) for y0, y1 in _tile_origins(h, cfg.tile_size) for x0, x1 in _tile_origins(w, cfg.tile_size)]
    workers = cfg.max_workers or max(1, min(8, (os.cpu_count() or 2) // 2))

    def run(job):
        return _extract_tile(image, detect_mask, *job, cfg)

    if workers == 1 or len(jobs) == 1:
        parts = [run(j) for j in jobs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(run, jobs))  # map keeps tile order -> deterministic output
    parts = [p for p in parts if p is not None]
    if not parts:
        return empty_features()

    descriptors = np.concatenate([p[4] for p in parts]).astype(np.float32)
    if cfg.root_sift:
        descriptors = to_root_sift(descriptors)
    return Features(
        points=np.concatenate([p[0] for p in parts]),
        sizes=np.concatenate([p[1] for p in parts]),
        angles=np.concatenate([p[2] for p in parts]),
        responses=np.concatenate([p[3] for p in parts]),
        descriptors=descriptors,
    )
