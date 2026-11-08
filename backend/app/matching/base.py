"""The contract every matcher fulfils, whatever produced its correspondences.

Coordinate convention (used everywhere in the project):

* points are ``(x, y)`` in **original image pixels**, with the centre of a pixel at
  integer coordinates (the OpenCV convention);
* ``source_points[i]`` corresponds to ``reference_points[i]``, and transformations
  always map **source -> reference**.

A matcher working on a crop, a resized copy or a rotated tile must convert back
before returning (see ``app.matching.tiling.PatchMap``). The registration pipeline
never needs to know which matcher ran.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel

MatcherStatusName = Literal["OK", "FAILED", "UNAVAILABLE"]

SIFT = "SIFT"
SUPERPOINT_LIGHTGLUE = "SUPERPOINT_LIGHTGLUE"
LOFTR = "LOFTR"
RIFT = "RIFT"
AUTO = "AUTO"


@dataclass
class MatcherResult:
    """Standard output of a matcher branch, before any geometric verification."""

    matcher_name: str
    source_points: np.ndarray  # (N, 2) float64, original source pixels
    reference_points: np.ndarray  # (N, 2) float64, original reference pixels
    confidence_scores: np.ndarray  # (N,) float64, higher = better, comparable only within one matcher
    rank: np.ndarray  # (N,) float64, lower = more distinctive; drives the seed search
    candidate_count: int  # correspondences proposed before matcher-specific filtering
    accepted_count: int  # correspondences that passed matcher-specific filtering (== N)
    runtime_seconds: float
    device: str
    status: MatcherStatusName = "OK"
    failure_reason: str | None = None
    source_keypoints: int | None = None  # None for detector-free matchers
    reference_keypoints: int | None = None
    extras: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.source_points = np.asarray(self.source_points, dtype=np.float64).reshape(-1, 2)
        self.reference_points = np.asarray(self.reference_points, dtype=np.float64).reshape(-1, 2)
        self.confidence_scores = np.asarray(self.confidence_scores, dtype=np.float64).reshape(-1)
        self.rank = np.asarray(self.rank, dtype=np.float64).reshape(-1)
        n = len(self.source_points)
        if not (len(self.reference_points) == len(self.confidence_scores) == len(self.rank) == n):
            raise ValueError("MatcherResult arrays must have the same length")

    def __len__(self) -> int:
        return len(self.source_points)


def failed_result(
    matcher_name: str, reason: str, *, device: str = "n/a", status: MatcherStatusName = "FAILED", runtime: float = 0.0
) -> MatcherResult:
    """A branch that produced nothing, with the reason it did not."""
    empty = np.zeros((0, 2))
    return MatcherResult(
        matcher_name=matcher_name,
        source_points=empty,
        reference_points=empty.copy(),
        confidence_scores=np.zeros(0),
        rank=np.zeros(0),
        candidate_count=0,
        accepted_count=0,
        runtime_seconds=runtime,
        device=device,
        status=status,
        failure_reason=reason,
    )


@dataclass
class PairMatches:
    """Raw output of one image pair, in the pixel coordinates of those two images."""

    points0: np.ndarray  # (N, 2) float64
    points1: np.ndarray  # (N, 2) float64
    confidence: np.ndarray  # (N,) float64
    candidates: int  # before confidence filtering
    keypoints0: int | None = None
    keypoints1: int | None = None

    def __len__(self) -> int:
        return len(self.points0)


def empty_pair(keypoints0: int | None = None, keypoints1: int | None = None) -> PairMatches:
    return PairMatches(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0), 0, keypoints0, keypoints1)


@runtime_checkable
class PairMatcher(Protocol):
    """Matches two uint8 grayscale images; masks are uint8 with non-zero = usable."""

    name: str
    device: str

    def match_pair(
        self, image0: np.ndarray, image1: np.ndarray, mask0: np.ndarray | None = None, mask1: np.ndarray | None = None
    ) -> PairMatches: ...


class WeightInfo(BaseModel):
    name: str
    source_url: str
    license: str
    present: bool
    path: str | None = None
    size_bytes: int | None = None


class MatcherStatus(BaseModel):
    """What the API and the UI report about a matcher. ``available`` is checked, never assumed."""

    name: str
    label: str
    available: bool
    reason: str | None = None
    experimental: bool = False
    needs_download: bool = False  # dependencies are installed, weights not yet downloaded
    device: str | None = None
    filtering: str
    implementation: str
    weights: list[WeightInfo] = []
