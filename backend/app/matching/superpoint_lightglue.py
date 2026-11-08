"""SuperPoint keypoints + LightGlue matching.

SuperPoint comes from the maintained Hugging Face port of the Magic Leap network,
LightGlue from Kornia's port of the ETH CVG model (both are on PyPI). LightGlue's
own match confidence is the filter here; the SIFT Lowe ratio test does not apply.

Large images are extracted in overlapping tiles so that memory stays bounded; a
keypoint keeps the descriptor computed in the tile where it lies furthest from
the border. Coordinates are integers in the input image's pixel frame.
"""
from __future__ import annotations

import numpy as np

from app.config import LightGlueConfig, SuperPointConfig
from app.matching.base import SUPERPOINT_LIGHTGLUE, PairMatches, empty_pair
from app.matching.models import load_lightglue, load_superpoint, resolve_device
from app.matching.tiling import keep_inside_mask, spans

STRIDE = 8  # SuperPoint downsamples by 8; inputs are cropped to multiples of it


class SuperPointLightGlue:
    name = SUPERPOINT_LIGHTGLUE

    def __init__(
        self,
        superpoint: SuperPointConfig | None = None,
        lightglue: LightGlueConfig | None = None,
        device: str = "auto",
        *,
        allow_download: bool = True,
    ) -> None:
        self.sp_cfg = superpoint or SuperPointConfig()
        self.lg_cfg = lightglue or LightGlueConfig()
        self.device = resolve_device(device)
        self.superpoint = load_superpoint(self.device, allow_download=allow_download)
        self.lightglue = load_lightglue(self.device, allow_download=allow_download)

    def extract(self, image: np.ndarray, mask: np.ndarray | None = None, max_keypoints: int | None = None):
        """(points (N, 2), scores (N,), descriptors (N, 256)) inside ``mask``, strongest first."""
        import torch

        limit = self.sp_cfg.max_keypoints if max_keypoints is None else max_keypoints
        h, w = image.shape
        tile, overlap = self.sp_cfg.tile_px, self.sp_cfg.tile_overlap_px
        decoder = self.superpoint.keypoint_decoder
        decoder.keypoint_threshold = self.sp_cfg.keypoint_threshold
        decoder.nms_radius = self.sp_cfg.nms_radius

        points, scores, descriptors = [], [], []
        step = max(STRIDE, tile - overlap)
        rows, columns = spans(h, tile, step), spans(w, tile, step)
        for r, (y0, y1) in enumerate(rows):
            for c, (x0, x1) in enumerate(columns):
                ch, cw = (y1 - y0) // STRIDE * STRIDE, (x1 - x0) // STRIDE * STRIDE
                if ch < 3 * STRIDE or cw < 3 * STRIDE:
                    continue
                if mask is not None and not mask[y0:y0 + ch, x0:x0 + cw].any():
                    continue
                array = image[y0:y0 + ch, x0:x0 + cw].astype(np.float32) / 255.0
                tensor = torch.from_numpy(array)[None, None].repeat(1, 3, 1, 1).to(self.device)
                with torch.inference_mode():
                    out = self.superpoint(tensor)
                n = int(out.mask[0].sum())
                if n == 0:
                    continue
                # The model returns coordinates relative to the input size; they were integers.
                local = (out.keypoints[0, :n].cpu().numpy() * np.array([cw, ch])).round()
                score = out.scores[0, :n].cpu().numpy()
                descriptor = out.descriptors[0, :n].cpu().numpy()
                # Each keypoint belongs to exactly one tile: overlaps are split down their middle.
                # (Tiles can overlap by far more than ``overlap``, because the last one sits flush
                # with the image edge.)
                x_low = 0 if c == 0 else (columns[c - 1][1] + x0) // 2
                x_high = w if c == len(columns) - 1 else (x1 + columns[c + 1][0]) // 2
                y_low = 0 if r == 0 else (rows[r - 1][1] + y0) // 2
                y_high = h if r == len(rows) - 1 else (y1 + rows[r + 1][0]) // 2
                absolute = local + [x0, y0]
                inner = (
                    (absolute[:, 0] >= x_low) & (absolute[:, 0] < x_high) & (absolute[:, 1] >= y_low) & (absolute[:, 1] < y_high)
                )
                points.append(absolute[inner])
                scores.append(score[inner])
                descriptors.append(descriptor[inner])
        if not points:
            return np.zeros((0, 2)), np.zeros(0), np.zeros((0, 256), np.float32)

        pts, sc, de = np.concatenate(points), np.concatenate(scores), np.concatenate(descriptors)
        keep = keep_inside_mask(pts, mask)
        pts, sc, de = pts[keep], sc[keep], de[keep]
        order = np.argsort(-sc, kind="stable")
        if limit and limit > 0:
            order = order[:limit]
        return pts[order], sc[order], de[order]

    def match_features(self, kp0, de0, size0, kp1, de1, size1) -> tuple[np.ndarray, np.ndarray]:
        """Index pairs (M, 2) and confidences (M,). ``size`` is (w, h)."""
        import torch

        if len(kp0) < 2 or len(kp1) < 2:
            return np.zeros((0, 2), np.int64), np.zeros(0)
        conf = self.lightglue.conf
        conf.depth_confidence = self.lg_cfg.depth_confidence
        conf.width_confidence = self.lg_cfg.width_confidence
        data = {
            "image0": {
                "keypoints": torch.from_numpy(kp0.astype(np.float32))[None].to(self.device),
                "descriptors": torch.from_numpy(de0)[None].to(self.device),
                "image_size": torch.tensor([list(size0)], device=self.device),
            },
            "image1": {
                "keypoints": torch.from_numpy(kp1.astype(np.float32))[None].to(self.device),
                "descriptors": torch.from_numpy(de1)[None].to(self.device),
                "image_size": torch.tensor([list(size1)], device=self.device),
            },
        }
        with torch.inference_mode():
            out = self.lightglue(data)
        return out["matches"][0].cpu().numpy(), out["scores"][0].cpu().numpy().astype(np.float64)

    def match_pair(
        self, image0: np.ndarray, image1: np.ndarray, mask0: np.ndarray | None = None, mask1: np.ndarray | None = None
    ) -> PairMatches:
        kp0, _, de0 = self.extract(image0, mask0)
        kp1, _, de1 = self.extract(image1, mask1)
        if len(kp0) < 2 or len(kp1) < 2:
            return empty_pair(len(kp0), len(kp1))
        size0, size1 = (image0.shape[1], image0.shape[0]), (image1.shape[1], image1.shape[0])
        matches, scores = self.match_features(kp0, de0, size0, kp1, de1, size1)
        candidates = len(matches)
        keep = scores >= self.lg_cfg.filter_threshold
        matches, scores = matches[keep], scores[keep]
        return PairMatches(kp0[matches[:, 0]], kp1[matches[:, 1]], scores, candidates, len(kp0), len(kp1))
