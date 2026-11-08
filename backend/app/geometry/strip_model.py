"""Smooth along-track model for pushbroom strips.

Measured so far on one official OHRC scene (2470502), the tile-by-tile displacement
from a global affine reaches over a hundred pixels along the 25 km strip: a single
affine fits one part of the strip and calls the rest outliers. Long IIRS strips should
add map-projection scale change on top. A pushbroom image is acquired one scan line
at a time, each with its own position and attitude, which suggests the model

    x' = s_x(u) + v * (a_x + b_x * u)
    y' = s_y(u) + v * (a_y + b_y * u)

with ``u`` the along-track and ``v`` the cross-track source coordinate, and ``s`` a
cubic P-spline (B-splines with a second-difference penalty). A very large penalty
reduces ``s`` to a straight line, which makes this exactly the affine model; the
penalty is chosen by along-track block cross-validation, so the data decide how
much flexibility is warranted and gaps are bridged smoothly instead of being
fitted to noise.

A strip model is an extension of the global transformations, not a replacement:
it is only adopted when verification finds it clearly better (see verification.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel
from scipy.interpolate import BSpline

from app.config import ValidationConfig

_DEGREE = 3
_CV_TOLERANCE = 0.10  # relative slack when preferring a smoother model
_CV_FLOOR_PX = 0.1  # differences below this are not meaningful at this matching accuracy


class StripCheck(BaseModel):
    valid: bool
    reasons: list[str]
    scale_min: float | None = None
    scale_max: float | None = None
    max_anisotropy: float | None = None
    max_shear_deg: float | None = None
    determinant_sign: int | None = None
    max_deviation_from_affine_px: float | None = None
    fraction_inside_reference: float | None = None


@dataclass
class StripModel:
    """Maps source pixels to reference pixels; picklable and JSON-serialisable."""

    axis: int  # 0: x is along-track, 1: y is along-track
    source_size: tuple[int, int]  # (w, h)
    knots: np.ndarray  # full knot vector over u in [0, 1]
    spline_x: np.ndarray  # B-spline coefficients of s_x
    spline_y: np.ndarray
    cross_x: np.ndarray  # (a_x, b_x)
    cross_y: np.ndarray
    smoothness: float

    # ------------------------------------------------------------------ evaluation
    def _uv(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        along = self.source_size[self.axis]
        across = self.source_size[1 - self.axis]
        u = pts[:, self.axis] / max(along - 1, 1)
        v = (pts[:, 1 - self.axis] - (across - 1) / 2.0) / max((across - 1) / 2.0, 1.0)
        return u, v

    def apply(self, points: np.ndarray) -> np.ndarray:
        u, v = self._uv(points)
        basis = _basis(u, self.knots)
        x = basis @ self.spline_x + v * (self.cross_x[0] + self.cross_x[1] * u)
        y = basis @ self.spline_y + v * (self.cross_y[0] + self.cross_y[1] * u)
        return np.c_[x, y]

    def errors(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        return np.linalg.norm(self.apply(src) - np.asarray(dst, dtype=np.float64), axis=1)

    def jacobian(self, points: np.ndarray, step: float = 2.0) -> np.ndarray:
        """(N, 2, 2) local linear map at each source point, by central differences."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        dx = (self.apply(pts + [step, 0]) - self.apply(pts - [step, 0])) / (2 * step)
        dy = (self.apply(pts + [0, step]) - self.apply(pts - [0, step])) / (2 * step)
        return np.stack([dx, dy], axis=2)

    @property
    def spacing_px(self) -> float:
        """Along-track knot spacing this model was built with."""
        intervals = max(1, len(self.knots) - 2 * _DEGREE - 1)
        return self.source_size[self.axis] / intervals

    def best_affine(self, n: int = 41) -> np.ndarray:
        """Least-squares 3x3 affine approximation over the source frame."""
        w, h = self.source_size
        grid = np.stack(np.meshgrid(np.linspace(0, w - 1, n), np.linspace(0, h - 1, n)), axis=-1).reshape(-1, 2)
        target = self.apply(grid)
        sol, *_ = np.linalg.lstsq(np.c_[grid, np.ones(len(grid))], target, rcond=None)
        return np.vstack([sol.T, [0.0, 0.0, 1.0]])

    # ------------------------------------------------------------------ inverse
    def inverse(self, targets: np.ndarray, iterations: int = 12) -> np.ndarray:
        """Source pixels for reference pixels ``targets`` (damped Newton from the affine)."""
        tgt = np.asarray(targets, dtype=np.float64).reshape(-1, 2)
        affine = self.best_affine()
        inv = np.linalg.inv(affine)
        cur = np.c_[tgt, np.ones(len(tgt))] @ inv.T
        cur = cur[:, :2]
        lin_inv = inv[:2, :2]
        w, h = self.source_size
        for _ in range(iterations):
            # Points are clamped only for evaluation so that the spline is not extrapolated wildly.
            clamped = np.c_[np.clip(cur[:, 0], -w * 0.2, w * 1.2), np.clip(cur[:, 1], -h * 0.2, h * 1.2)]
            residual = tgt - self.apply(clamped)
            cur = clamped + residual @ lin_inv.T
            if np.max(np.abs(residual)) < 1e-3:
                break
        return cur

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {
            "model": "strip_spline",
            "form": "x' = s_x(u) + v*(a_x + b_x*u); y' = s_y(u) + v*(a_y + b_y*u); s = cubic B-spline",
            "along_track_axis": "x" if self.axis == 0 else "y",
            "source_size": list(self.source_size),
            "u_definition": "along-track pixel / (along-track size - 1)",
            "v_definition": "(cross-track pixel - centre) / centre",
            "knots": self.knots.tolist(),
            "spline_x": self.spline_x.tolist(),
            "spline_y": self.spline_y.tolist(),
            "cross_x": self.cross_x.tolist(),
            "cross_y": self.cross_y.tolist(),
            "smoothness": self.smoothness,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StripModel":
        return cls(
            axis=0 if data["along_track_axis"] == "x" else 1,
            source_size=tuple(data["source_size"]),
            knots=np.asarray(data["knots"], float),
            spline_x=np.asarray(data["spline_x"], float),
            spline_y=np.asarray(data["spline_y"], float),
            cross_x=np.asarray(data["cross_x"], float),
            cross_y=np.asarray(data["cross_y"], float),
            smoothness=float(data["smoothness"]),
        )


def _basis(u: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Dense B-spline design matrix; u outside [0, 1] uses the end polynomial pieces."""
    u = np.asarray(u, dtype=np.float64)
    count = len(knots) - _DEGREE - 1
    eye = np.eye(count)
    return BSpline(knots, eye, _DEGREE, extrapolate=True)(u)


def make_knots(along_px: float, spacing_px: float) -> np.ndarray:
    """Equally spaced knots that run past both ends of [0, 1].

    The second-difference penalty only reduces the spline to a straight line when the knots
    are uniform (Eilers & Marx). With knots clamped at the ends, a straight line does not
    have evenly spaced coefficients, so the penalty forces a steep slope wherever no data
    push back. On official scene 2470502 that produced a local scale of 3 at both unsupported
    ends of the strip, at every smoothness, while the supported part stayed near 1.
    """
    intervals = max(1, int(round(along_px / spacing_px)))
    return np.arange(-_DEGREE, intervals + _DEGREE + 1, dtype=np.float64) / intervals


def _penalty(count: int) -> np.ndarray:
    """Second-difference penalty on the spline coefficients."""
    if count < 3:
        return np.zeros((0, count))
    return np.diff(np.eye(count), n=2, axis=0)


def _solve(design: np.ndarray, penalty: np.ndarray, target: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    a, b = design, target
    if weights is not None:
        a = design * weights[:, None]
        b = target * weights
    lhs = np.vstack([a, penalty])
    rhs = np.r_[b, np.zeros(len(penalty))]
    return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def balancing_weights(along: np.ndarray, length: float, spacing_px: float) -> np.ndarray:
    """Point weights that stop one dense stretch of the strip from deciding the whole fit.

    On official scene 2470502 one tile holds over 500 matches and its neighbours a few dozen,
    so with equal point weights the fit follows that tile and misses the shape of the curve.
    A group of n points gets a total weight growing with sqrt(n): a dense group still counts
    for more, but not a hundred times more.
    """
    groups = max(1, int(round(length / spacing_px)))
    labels = np.clip((np.asarray(along) / length * groups).astype(np.int64), 0, groups - 1)
    counts = np.bincount(labels, minlength=groups)
    return counts[labels].astype(np.float64) ** -0.25


def fit_strip(
    src: np.ndarray,
    dst: np.ndarray,
    source_size: tuple[int, int],
    spacing_px: float,
    smoothness: float,
) -> StripModel | None:
    """Least-squares strip model through points believed to correspond."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    axis = 1 if source_size[1] >= source_size[0] else 0
    if len(src) < 8:
        return None
    weights = balancing_weights(src[:, axis], source_size[axis], spacing_px)
    knots = make_knots(source_size[axis], spacing_px)
    model = StripModel(axis, source_size, knots, np.zeros(0), np.zeros(0), np.zeros(2), np.zeros(2), smoothness)
    u, v = model._uv(src)
    basis = _basis(u, knots)
    count = basis.shape[1]
    design = np.c_[basis, v, v * u]
    # The penalty is scaled by the data weight so that one smoothness value means the same
    # thing for a sparse and a dense set of tie points.
    scale = np.sqrt(max(smoothness, 0.0) * max(float((weights**2).sum()), 1.0) / max(count, 1))
    penalty = np.c_[_penalty(count) * scale, np.zeros((max(count - 2, 0), 2))]
    try:
        cx = _solve(design, penalty, dst[:, 0], weights)
        cy = _solve(design, penalty, dst[:, 1], weights)
    except np.linalg.LinAlgError:
        return None
    if not (np.all(np.isfinite(cx)) and np.all(np.isfinite(cy))):
        return None
    model.spline_x, model.cross_x = cx[:count], cx[count:]
    model.spline_y, model.cross_y = cy[:count], cy[count:]
    return model


def choose_smoothness(
    src: np.ndarray,
    dst: np.ndarray,
    source_size: tuple[int, int],
    spacing_px: float,
    candidates: list[float],
    blocks: int = 6,
) -> tuple[float, dict[float, float]]:
    """Penalty with the lowest error on along-track blocks left out of the fit.

    Contiguous blocks are held out, not random points: neighbouring tie points are
    strongly correlated, so a random split would always favour the most flexible model.
    """
    axis = 1 if source_size[1] >= source_size[0] else 0
    edges = np.linspace(0, source_size[axis], blocks + 1)
    labels = np.clip(np.digitize(src[:, axis], edges) - 1, 0, blocks - 1)
    scores: dict[float, float] = {}
    for lam in candidates:
        errors = []
        for b in range(blocks):
            test = labels == b
            if not test.any() or (~test).sum() < 8:
                continue
            model = fit_strip(src[~test], dst[~test], source_size, spacing_px, lam)
            if model is None:
                continue
            # Each block counts the same, however many points it holds.
            errors.append(float(np.sqrt(np.mean(model.errors(src[test], dst[test]) ** 2))))
        scores[lam] = float(np.mean(errors)) if errors else float("inf")
    # The smoothest model close to the best error, not the lowest error outright: a flexible fit
    # that wins by a hair on correlated points also wiggles wherever there is no support.
    lowest = min(scores.values())
    tolerance = lowest * (1.0 + _CV_TOLERANCE) + _CV_FLOOR_PX
    admissible = [lam for lam in candidates if scores[lam] <= tolerance]
    return max(admissible) if admissible else min(candidates, key=lambda lam: scores[lam]), scores


def leave_block_out_errors(
    src: np.ndarray,
    dst: np.ndarray,
    source_size: tuple[int, int],
    fit,
    blocks: int = 6,
) -> list[np.ndarray]:
    """Errors on each along-track block left out of the fit (one array per block tested).

    ``fit(train_src, train_dst)`` returns a function mapping source points to reference points,
    or None. This asks the question that matters for a long strip: does the model predict
    ground it has not seen?
    """
    axis = 1 if source_size[1] >= source_size[0] else 0
    edges = np.linspace(0, source_size[axis], blocks + 1)
    labels = np.clip(np.digitize(src[:, axis], edges) - 1, 0, blocks - 1)
    errors: list[np.ndarray] = []
    for b in range(blocks):
        test = labels == b
        if test.sum() < 3 or (~test).sum() < 8:
            continue
        predict = fit(src[~test], dst[~test])
        if predict is None:
            continue
        errors.append(np.linalg.norm(predict(src[test]) - dst[test], axis=1))
    return errors


def track_residuals(
    src: np.ndarray,
    residual: np.ndarray,
    source_size: tuple[int, int],
    bin_px: float,
    min_votes: int,
    spacing_px: float,
    max_slope: float = 0.08,
) -> list[tuple[float, np.ndarray, int]]:
    """Follow the along-track deviation from a starting model through the candidates.

    Candidates are grouped along-track; each group votes for its displacement. Wrong
    matches scatter while correct ones agree, so groups have sharp peaks when anything
    correct is present. Starting from the strongest peak, the track extends in both
    directions and accepts in each group the strongest peak compatible with a smooth
    trajectory, so a false peak cannot pull the curve away.

    Returns ``(along-track pixel, displacement, votes)``, ordered along-track.
    """
    axis = 1 if source_size[1] >= source_size[0] else 0
    along = source_size[axis]
    count = max(1, int(round(along / spacing_px)))
    edges = np.linspace(0, along, count + 1)
    labels = np.clip(np.digitize(src[:, axis], edges) - 1, 0, count - 1)

    peaks: list[list[tuple[int, np.ndarray]]] = []
    for g in range(count):
        d = residual[labels == g]
        found: list[tuple[int, np.ndarray]] = []
        if len(d):
            for shift in ((0.0, 0.0), (0.5, 0.5)):
                cells = np.floor(d / bin_px + shift).astype(np.int64)
                unique, votes = np.unique(cells, axis=0, return_counts=True)
                for i in np.argsort(-votes)[:6]:
                    if votes[i] < 2:
                        break
                    found.append((int(votes[i]), np.median(d[np.all(cells == unique[i], axis=1)], axis=0)))
        found.sort(key=lambda p: -p[0])
        peaks.append(found)

    strongest = [p[0][0] if p else 0 for p in peaks]
    start = int(np.argmax(strongest))
    if strongest[start] < min_votes:
        return []
    centre = (edges[:-1] + edges[1:]) / 2.0
    chosen: dict[int, tuple[int, np.ndarray]] = {start: peaks[start][0]}

    for direction in (1, -1):
        last = start
        g = start + direction
        while 0 <= g < count:
            allowed = max_slope * abs(centre[g] - centre[last]) + 2.0 * bin_px
            for votes, value in peaks[g]:
                if votes < max(2, min_votes // 2):
                    break
                if np.linalg.norm(value - chosen[last][1]) <= allowed:
                    chosen[g] = (votes, value)
                    last = g
                    break
            g += direction
    return [(float(centre[g]), chosen[g][1], chosen[g][0]) for g in sorted(chosen)]


def check_strip(
    model: StripModel,
    reference_size: tuple[int, int],
    cfg: ValidationConfig,
    expected_sign: int,
    samples: int = 200,
) -> StripCheck:
    """A flexible model must still be a physically plausible, fold-free mapping everywhere."""
    w, h = model.source_size
    along = np.linspace(0, model.source_size[model.axis] - 1, samples)
    across = np.linspace(0, model.source_size[1 - model.axis] - 1, 5)
    aa, cc = np.meshgrid(along, across)
    grid = np.c_[aa.ravel(), cc.ravel()] if model.axis == 0 else np.c_[cc.ravel(), aa.ravel()]
    jac = model.jacobian(grid)
    if not np.all(np.isfinite(jac)):
        return StripCheck(valid=False, reasons=["strip model has non-finite values"])

    sv = np.linalg.svd(jac, compute_uv=False)
    det = np.linalg.det(jac)
    col_x, col_y = jac[:, :, 0], jac[:, :, 1]
    cos_axes = np.sum(col_x * col_y, axis=1) / (np.linalg.norm(col_x, axis=1) * np.linalg.norm(col_y, axis=1) + 1e-12)
    shear = np.abs(90.0 - np.degrees(np.arccos(np.clip(cos_axes, -1, 1))))
    anis = sv[:, 0] / np.maximum(sv[:, 1], 1e-12)

    reasons: list[str] = []
    if np.any(np.sign(det) != expected_sign):
        reasons.append("strip model folds or changes orientation along the strip")
    if sv.min() < cfg.min_scale or sv.max() > cfg.max_scale:
        reasons.append(f"implausible local scale ({sv.min():.3f}..{sv.max():.3f})")
    if anis.max() > cfg.max_anisotropy:
        reasons.append(f"excessive local anisotropy ({anis.max():.2f} > {cfg.max_anisotropy})")
    if shear.max() > cfg.max_shear_deg:
        reasons.append(f"excessive local shear ({shear.max():.1f} deg > {cfg.max_shear_deg})")

    affine = model.best_affine()
    affine_pred = np.c_[grid, np.ones(len(grid))] @ affine[:2].T
    deviation = float(np.linalg.norm(model.apply(grid) - affine_pred, axis=1).max())

    mapped = model.apply(grid)
    rw, rh = reference_size
    inside = float(np.mean((mapped[:, 0] >= 0) & (mapped[:, 0] < rw) & (mapped[:, 1] >= 0) & (mapped[:, 1] < rh)))
    if inside < cfg.min_projected_overlap:
        reasons.append(f"only {inside:.1%} of the strip lands inside the reference")
    return StripCheck(
        valid=not reasons,
        reasons=reasons,
        scale_min=float(sv.min()),
        scale_max=float(sv.max()),
        max_anisotropy=float(anis.max()),
        max_shear_deg=float(shear.max()),
        determinant_sign=int(np.sign(det.mean())),
        max_deviation_from_affine_px=deviation,
        fraction_inside_reference=inside,
    )
