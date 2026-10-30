"""Visual artifacts. All drawings are down-scaled copies; metrics never use them."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.config import VisualizationConfig
from app.evaluation.metrics import SpatialCoverage

GREEN = (80, 220, 100)  # BGR
RED = (70, 70, 235)
YELLOW = (60, 220, 250)
CYAN = (230, 210, 60)
WHITE = (255, 255, 255)
MIN_SHORT_SIDE = 360


def fit_scale(width: int, height: int, max_side: int) -> float:
    """Scale that fits the long side in ``max_side`` without shrinking narrow strips to slivers."""
    scale = min(1.0, max_side / max(width, height))
    if min(width, height) * scale < MIN_SHORT_SIDE:
        scale = min(1.0, MIN_SHORT_SIDE / min(width, height), 2.0 * max_side / max(width, height))
    return scale


def _resize(image: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return image.copy()
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def save_image(path: Path, image: np.ndarray, jpeg_quality: int = 88) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if path.suffix.lower() in (".jpg", ".jpeg") else []
    if not cv2.imwrite(str(path), image, params):
        raise OSError(f"Could not write {path}")


def preview(display: np.ndarray, cfg: VisualizationConfig) -> np.ndarray:
    return _resize(display, fit_scale(display.shape[1], display.shape[0], cfg.max_side_px))


def draw_keypoints(display: np.ndarray, points: np.ndarray, cfg: VisualizationConfig, limit: int = 4000) -> np.ndarray:
    scale = fit_scale(display.shape[1], display.shape[0], cfg.max_side_px)
    canvas = _bgr(_resize(display, scale))
    if len(points) > limit:  # even subsample, for legibility only
        points = points[np.linspace(0, len(points) - 1, limit).astype(int)]
    for x, y in points * scale:
        cv2.circle(canvas, (int(round(x)), int(round(y))), 2, GREEN, 1, cv2.LINE_AA)
    return canvas


def _sample_along_track(src_pts: np.ndarray, limit: int) -> np.ndarray:
    """Up to ``limit`` indices, spread evenly along the longer source axis."""
    n = len(src_pts)
    if n <= limit:
        return np.arange(n)
    axis = 1 if np.ptp(src_pts[:, 1]) >= np.ptp(src_pts[:, 0]) else 0
    order = np.argsort(src_pts[:, axis], kind="stable")
    return order[np.linspace(0, n - 1, limit).astype(int)]


def draw_matches(
    src_display: np.ndarray,
    ref_display: np.ndarray,
    src_pts: np.ndarray,
    ref_pts: np.ndarray,
    cfg: VisualizationConfig,
    colour: tuple[int, int, int],
    label: str,
) -> np.ndarray:
    """Source and reference side by side at one common scale, with match lines."""
    gap = 24
    sh, sw = src_display.shape
    rh, rw = ref_display.shape
    scale = min(1.0, cfg.max_side_px / (sw + rw + gap), cfg.max_side_px / max(sh, rh))
    if sw * scale < 160:  # keep a narrow strip readable
        scale = min(1.0, 160 / sw, 2.0 * cfg.max_side_px / max(sh, rh))
    s, r = _bgr(_resize(src_display, scale)), _bgr(_resize(ref_display, scale))
    header = 34
    canvas = np.full((max(s.shape[0], r.shape[0]) + header, s.shape[1] + gap + r.shape[1], 3), 24, np.uint8)
    canvas[header : header + s.shape[0], : s.shape[1]] = s
    x_off = s.shape[1] + gap
    canvas[header : header + r.shape[0], x_off : x_off + r.shape[1]] = r

    total = len(src_pts)
    idx = _sample_along_track(np.asarray(src_pts), cfg.max_match_lines) if total else np.zeros(0, int)
    for i in idx:
        p = (int(round(src_pts[i][0] * scale)), int(round(src_pts[i][1] * scale)) + header)
        q = (int(round(ref_pts[i][0] * scale)) + x_off, int(round(ref_pts[i][1] * scale)) + header)
        cv2.line(canvas, p, q, colour, 1, cv2.LINE_AA)
        cv2.circle(canvas, p, 3, colour, -1, cv2.LINE_AA)
        cv2.circle(canvas, q, 3, colour, -1, cv2.LINE_AA)
    shown = f"showing {len(idx)} of {total}" if total > len(idx) else f"{total} shown"
    cv2.putText(canvas, f"{label}: {shown}", (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
    return canvas


def draw_coverage(
    src_display: np.ndarray, coverage: SpatialCoverage, inlier_pts: np.ndarray, cfg: VisualizationConfig
) -> np.ndarray:
    """Source image with the coverage grid: green = occupied, red = empty, grey = not counted."""
    h, w = src_display.shape
    scale = fit_scale(w, h, cfg.max_side_px)
    canvas = _bgr(_resize(src_display, scale))
    ch, cw = canvas.shape[:2]
    tint = canvas.copy()
    ys = np.linspace(0, ch, coverage.grid_rows + 1).round().astype(int)
    xs = np.linspace(0, cw, coverage.grid_cols + 1).round().astype(int)
    for r in range(coverage.grid_rows):
        for c in range(coverage.grid_cols):
            valid = coverage.cell_valid[r][c]
            in_overlap = coverage.cell_in_overlap[r][c] if coverage.cell_in_overlap else valid
            if not valid or not in_overlap:
                colour = (110, 110, 110)
            elif coverage.cell_counts[r][c] > 0:
                colour = GREEN
            else:
                colour = RED
            cv2.rectangle(tint, (xs[c], ys[r]), (xs[c + 1], ys[r + 1]), colour, -1)
    canvas = cv2.addWeighted(tint, 0.28, canvas, 0.72, 0)
    for x in xs:
        cv2.line(canvas, (min(x, cw - 1), 0), (min(x, cw - 1), ch - 1), WHITE, 1)
    for y in ys:
        cv2.line(canvas, (0, min(y, ch - 1)), (cw - 1, min(y, ch - 1)), WHITE, 1)
    for x, y in np.asarray(inlier_pts).reshape(-1, 2) * scale:
        cv2.circle(canvas, (int(round(x)), int(round(y))), 4, YELLOW, -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(round(x)), int(round(y))), 4, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def draw_footprint(
    ref_display: np.ndarray,
    estimated_quad: np.ndarray | None,
    predicted_quad: np.ndarray | None,
    cfg: VisualizationConfig,
) -> np.ndarray:
    """Reference with the estimated (green) and metadata-predicted (yellow) source footprints."""
    scale = fit_scale(ref_display.shape[1], ref_display.shape[0], cfg.max_side_px)
    canvas = _bgr(_resize(ref_display, scale))
    for quad, colour, width in ((predicted_quad, YELLOW, 2), (estimated_quad, GREEN, 3)):
        if quad is not None and np.all(np.isfinite(quad)):
            pts = np.round(np.asarray(quad) * scale).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], True, colour, width, cv2.LINE_AA)
    return canvas


def make_overlay(
    ref_display: np.ndarray,
    registered: np.ndarray,
    registered_mask: np.ndarray,
    quad: np.ndarray,
    cfg: VisualizationConfig,
) -> np.ndarray:
    """Colour overlay: reference in red, registered source in cyan (green + blue).

    Terrain that lines up appears grey; misalignment shows as red/cyan fringes.
    Outside the registered footprint the reference is shown dimmed.
    """
    h, w = ref_display.shape
    margin = 0.06 * max(np.ptp(quad[:, 0]), np.ptp(quad[:, 1]))
    x0 = int(np.clip(np.floor(quad[:, 0].min() - margin), 0, w - 1))
    x1 = int(np.clip(np.ceil(quad[:, 0].max() + margin), x0 + 1, w))
    y0 = int(np.clip(np.floor(quad[:, 1].min() - margin), 0, h - 1))
    y1 = int(np.clip(np.ceil(quad[:, 1].max() + margin), y0 + 1, h))
    ref, reg, mask = ref_display[y0:y1, x0:x1], registered[y0:y1, x0:x1], registered_mask[y0:y1, x0:x1]
    dim = (ref * 0.45).astype(np.uint8)
    overlay = np.dstack([np.where(mask, reg, dim), np.where(mask, reg, dim), np.where(mask, ref, dim)])
    return _resize(overlay, fit_scale(overlay.shape[1], overlay.shape[0], cfg.max_side_px))
