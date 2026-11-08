"""Which matchers exist, whether they can actually run, and how to create them."""
from __future__ import annotations

import numpy as np

from app.config import PipelineConfig, SiftConfig
from app.matching.base import (
    LOFTR,
    RIFT,
    SIFT,
    SUPERPOINT_LIGHTGLUE,
    MatcherStatus,
    PairMatcher,
    PairMatches,
    empty_pair,
)
from app.matching.matcher import match_features
from app.matching.models import ModelUnavailable, dependency_missing, weight_info
from app.matching.sift import extract_sift

LEARNED = (SUPERPOINT_LIGHTGLUE, LOFTR)


class SiftPairMatcher:
    """SIFT / RootSIFT with the Lowe ratio test behind the common pair interface."""

    name = SIFT
    device = "cpu"

    def __init__(self, cfg: PipelineConfig | None = None) -> None:
        self.cfg = cfg or PipelineConfig()

    def match_pair(
        self, image0: np.ndarray, image1: np.ndarray, mask0: np.ndarray | None = None, mask1: np.ndarray | None = None
    ) -> PairMatches:
        def full(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
            return np.full(image.shape, 255, np.uint8) if mask is None else (mask > 0).astype(np.uint8) * 255

        features0 = extract_sift(image0, full(image0, mask0), self.cfg.sift)
        features1 = extract_sift(image1, full(image1, mask1), self.cfg.sift)
        if len(features0) < 2 or len(features1) < 2:
            return empty_pair(len(features0), len(features1))
        matches = match_features(features0, features1, self.cfg.match)
        return PairMatches(
            features0.points[matches.source_idx].astype(np.float64),
            features1.points[matches.reference_idx].astype(np.float64),
            1.0 - matches.ratios.astype(np.float64),
            matches.raw_candidates,
            len(features0),
            len(features1),
        )


def _device_name(preference: str) -> str | None:
    if dependency_missing("torch"):
        return None
    from app.matching.models import resolve_device

    try:
        return resolve_device(preference)
    except ModelUnavailable:
        return None


def matcher_statuses(cfg: PipelineConfig | None = None) -> list[MatcherStatus]:
    """Real availability of every matcher: installed packages and weights on disk.

    ``available`` means it can run right now. ``needs_download`` means the packages are
    there and the weights will be fetched from their official source on first use.
    """
    cfg = cfg or PipelineConfig()
    device = _device_name(cfg.device)
    statuses = [
        MatcherStatus(
            name=SIFT,
            label="SIFT / RootSIFT",
            available=True,
            device="cpu",
            filtering="KNN (k=2) + Lowe ratio test",
            implementation="OpenCV",
        )
    ]

    def learned(name: str, label: str, filtering: str, implementation: str, modules: tuple[str, ...], keys: tuple[str, ...]) -> MatcherStatus:
        missing = dependency_missing(*modules)
        weights = [weight_info(k) for k in keys]
        absent = [w.name for w in weights if not w.present]
        if missing:
            return MatcherStatus(
                name=name, label=label, available=False, reason=f"Python package '{missing}' is not installed",
                filtering=filtering, implementation=implementation, weights=weights,
            )  # fmt: skip
        if absent and not cfg.allow_model_download:
            return MatcherStatus(
                name=name, label=label, available=False, reason="weights not downloaded: " + ", ".join(absent),
                device=device, filtering=filtering, implementation=implementation, weights=weights,
            )  # fmt: skip
        return MatcherStatus(
            name=name, label=label, available=not absent, needs_download=bool(absent),
            reason=("weights will be downloaded on first use: " + ", ".join(absent)) if absent else None,
            device=device, filtering=filtering, implementation=implementation, weights=weights,
        )  # fmt: skip

    statuses.append(
        learned(
            SUPERPOINT_LIGHTGLUE,
            "SuperPoint + LightGlue",
            "LightGlue match confidence",
            "transformers (SuperPoint) + Kornia (LightGlue)",
            ("torch", "kornia", "transformers"),
            ("superpoint", "lightglue"),
        )
    )
    statuses.append(
        learned(LOFTR, "LoFTR", "LoFTR coarse match confidence", "Kornia (detector-free)", ("torch", "kornia"), ("loftr",))
    )
    from app.matching.rift import rift_status

    statuses.append(rift_status())
    return statuses


def runnable_matchers(cfg: PipelineConfig | None = None) -> list[str]:
    """Names AUTO may try: everything that can run now or fetch its weights, except experimental ones."""
    out = []
    for status in matcher_statuses(cfg):
        if (status.available or status.needs_download) and not status.experimental:
            out.append(status.name)
    return out


def create_pair_matcher(name: str, cfg: PipelineConfig | None = None) -> PairMatcher:
    """Instantiate a matcher; raises ``ModelUnavailable`` when it cannot run."""
    cfg = cfg or PipelineConfig()
    if name == SIFT:
        return SiftPairMatcher(cfg)
    if name == SUPERPOINT_LIGHTGLUE:
        missing = dependency_missing("torch", "kornia", "transformers")
        if missing:
            raise ModelUnavailable(f"Python package '{missing}' is not installed")
        from app.matching.superpoint_lightglue import SuperPointLightGlue

        return SuperPointLightGlue(cfg.superpoint, cfg.lightglue, cfg.device, allow_download=cfg.allow_model_download)
    if name == LOFTR:
        missing = dependency_missing("torch", "kornia")
        if missing:
            raise ModelUnavailable(f"Python package '{missing}' is not installed")
        from app.matching.loftr import Loftr

        return Loftr(cfg.loftr, cfg.device, allow_download=cfg.allow_model_download)
    if name == RIFT:
        from app.matching.rift import RiftMatcher

        return RiftMatcher(cfg.rift)
    raise ValueError(f"Unknown matcher: {name}")


__all__ = ["LEARNED", "SiftConfig", "SiftPairMatcher", "create_pair_matcher", "matcher_statuses", "runnable_matchers"]
