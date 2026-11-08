"""LoFTR: detector-free matching (Kornia, pretrained 'outdoor' weights).

This branch shares nothing with SuperPoint or SIFT. LoFTR's own coarse matching
score is the filter; neither the Lowe ratio test nor the LightGlue threshold apply.

The pretrained models place the centre of pixel (x, y) at (x + 0.5, y + 0.5), so
0.5 is subtracted from their output to reach the project convention (pixel centres
at integer coordinates). Forgetting this would be a constant half-pixel bias, which
is exactly the size of error a sub-pixel claim has to rule out.
"""
from __future__ import annotations

import numpy as np

from app.config import LoftrConfig
from app.matching.base import LOFTR, PairMatches, empty_pair
from app.matching.models import load_loftr, resolve_device
from app.matching.tiling import keep_inside_mask

MIN_SIDE = 64  # below this the 1/8 coarse map has too few cells to match anything


class Loftr:
    name = LOFTR

    def __init__(self, cfg: LoftrConfig | None = None, device: str = "auto", *, allow_download: bool = True) -> None:
        self.cfg = cfg or LoftrConfig()
        self.device = resolve_device(device)
        self.model = load_loftr(self.device, allow_download=allow_download)

    def match_pair(
        self, image0: np.ndarray, image1: np.ndarray, mask0: np.ndarray | None = None, mask1: np.ndarray | None = None
    ) -> PairMatches:
        import torch

        if min(image0.shape) < MIN_SIDE or min(image1.shape) < MIN_SIDE:
            return empty_pair()
        if mask0 is not None and not mask0.any() or mask1 is not None and not mask1.any():
            return empty_pair()

        def tensor(image: np.ndarray):
            return torch.from_numpy(image.astype(np.float32) / 255.0)[None, None].to(self.device)

        data = {"image0": tensor(image0), "image1": tensor(image1)}
        # Both masks or neither: LoFTR only uses them to exclude padded coarse cells.
        if mask0 is not None and mask1 is not None:
            data["mask0"] = torch.from_numpy(mask0 > 0)[None].to(self.device)
            data["mask1"] = torch.from_numpy(mask1 > 0)[None].to(self.device)
        with torch.inference_mode():
            out = self.model(data)

        p0 = out["keypoints0"].cpu().numpy().astype(np.float64) - 0.5
        p1 = out["keypoints1"].cpu().numpy().astype(np.float64) - 0.5
        conf = out["confidence"].cpu().numpy().astype(np.float64)
        candidates = len(p0)
        keep = conf >= self.cfg.confidence_threshold
        keep &= keep_inside_mask(p0, mask0) & keep_inside_mask(p1, mask1)
        return PairMatches(p0[keep], p1[keep], conf[keep], candidates)
