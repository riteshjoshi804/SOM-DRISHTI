"""Three kinds of evidence, kept strictly apart.

A. **Geometric consistency**: residuals of the points used for the fit (``metrics.py``).
B. **Held-out validation** (here): the transformation is estimated without some of the
   automatically matched correspondences and tested on them. This is real out-of-sample
   evidence that the model predicts unseen matches, but it is not ground truth, because
   those points come from the same matcher with the same systematic errors.
C. **Independent control points** (here): points identified independently by a person or
   another source, used only for evaluation and never in a fit. Only this can be
   called independent accuracy, and no such points ship with the official sample.

Synthetic ground truth lives in ``synthetic.py`` and is always labelled as synthetic.
"""
from __future__ import annotations

import csv
import io
import math
from typing import Callable

import numpy as np
from pydantic import BaseModel

from app.config import GeometryConfig, GuidedSearchConfig, HoldoutConfig, ValidationConfig
from app.geometry.estimation import apply_transform, fit_least_squares
from app.geometry.strip_model import fit_strip, leave_block_out_errors
from app.geometry.verification import VerifiedGeometry, verify_candidates


class HoldoutResult(BaseModel):
    label: str = "held-out automatic correspondences (not independent ground truth)"
    status: str  # ok | insufficient
    reason: str | None = None
    model: str | None = None
    fraction: float | None = None
    repeats_requested: int = 0
    repeats_completed: int = 0
    gate_px: float | None = None
    # Measured on the points each split used to fit its model.
    training_reprojection_rmse_px: float | None = None
    training_median_px: float | None = None
    # Measured on candidates that split did not use.
    heldout_candidates: int = 0
    heldout_within_gate: int = 0
    heldout_within_gate_fraction: float | None = None
    expected_chance_within_gate: float | None = None
    heldout_transfer_rmse_px: float | None = None
    heldout_median_px: float | None = None
    heldout_p90_px: float | None = None
    heldout_within_3px: int = 0
    # Contiguous blocks left out in turn: the harder, extrapolating test.
    block_label: str = "along-track blocks of verified inliers left out in turn (extrapolation across unseen ground)"
    block_rmse_px: float | None = None
    block_median_px: float | None = None
    block_p90_px: float | None = None
    block_count: int = 0


class ControlPoint(BaseModel):
    source_x: float
    source_y: float
    reference_x: float
    reference_y: float
    id: str | None = None
    note: str | None = None


class ControlPointEvaluation(BaseModel):
    label: str = "independent control points (evaluation only, never used in a fit)"
    provenance: str
    count: int
    rmse_px: float | None
    median_px: float | None
    p90_px: float | None
    max_px: float | None
    per_point: list[dict]
    warnings: list[str] = []


def stratified_split(
    points: np.ndarray, size: tuple[int, int], rows: int, cols: int, fraction: float, rng: np.random.Generator
) -> np.ndarray:
    """Boolean mask of held-out points, drawn cell by cell so they span the image like the rest."""
    w, h = size
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    r = np.clip((pts[:, 1] / h * rows).astype(np.int64), 0, rows - 1)
    c = np.clip((pts[:, 0] / w * cols).astype(np.int64), 0, cols - 1)
    cell = r * cols + c
    held = np.zeros(len(pts), bool)
    for value in np.unique(cell):
        members = np.flatnonzero(cell == value)
        if len(members) < 2:
            continue  # a lone point stays in the fit, or that cell would be unconstrained
        wanted = fraction * len(members)
        count = int(np.floor(wanted)) + int(rng.random() < wanted - np.floor(wanted))
        count = min(count, len(members) - 1)
        if count > 0:
            held[rng.choice(members, count, replace=False)] = True
    return held


def _stats(errors: np.ndarray) -> tuple[float | None, float | None, float | None]:
    if len(errors) == 0:
        return None, None, None
    return (
        float(np.sqrt(np.mean(errors**2))),
        float(np.median(errors)),
        float(np.percentile(errors, 90)),
    )


def holdout_validation(
    src: np.ndarray,
    dst: np.ndarray,
    rank: np.ndarray,
    final: VerifiedGeometry,
    source_size: tuple[int, int],
    reference_size: tuple[int, int],
    geometry_cfg: GeometryConfig,
    validation_cfg: ValidationConfig,
    guided_cfg: GuidedSearchConfig | None,
    cfg: HoldoutConfig,
    search_area_px: float,
    seed: int = 0,
) -> HoldoutResult:
    """Fit on a spatially stratified share of the candidates, test on the rest, repeatedly.

    The split is made *before* robust estimation, so a held-out point has no influence on
    the model it is tested against, not even on the choice of inliers. Held-out candidates
    include wrong matches, which is why errors are reported within ``gate_px`` together with
    how many candidates fall in it and how many would land there by chance.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    rank = np.asarray(rank, dtype=np.float64).reshape(-1)
    out = HoldoutResult(status="insufficient", model=final.kind, fraction=cfg.fraction, repeats_requested=cfg.repeats, gate_px=cfg.gate_px)
    rng = np.random.default_rng(seed)
    use_strip = guided_cfg if final.strip is not None else None

    train_errors: list[np.ndarray] = []
    test_errors: list[np.ndarray] = []
    candidates = gated = within_three = 0
    for _ in range(cfg.repeats):
        held = stratified_split(src, source_size, cfg.grid_rows, cfg.grid_cols, cfg.fraction, rng)
        if held.sum() < cfg.min_points or (~held).sum() < cfg.min_train_inliers:
            continue
        model, _ = verify_candidates(
            src[~held], dst[~held], rank[~held], source_size, reference_size,
            geometry_cfg, validation_cfg, use_strip, mirrored=final.mirrored,
        )  # fmt: skip
        if model is None or not model.valid or len(model.inlier_idx) < cfg.min_train_inliers:
            continue
        train_src, train_dst = src[~held][model.inlier_idx], dst[~held][model.inlier_idx]
        train_errors.append(model.errors(train_src, train_dst))
        errors = model.errors(src[held], dst[held])
        candidates += len(errors)
        keep = errors <= cfg.gate_px
        gated += int(keep.sum())
        within_three += int((errors <= geometry_cfg.reproj_threshold_px).sum())
        test_errors.append(errors[keep])
        out.repeats_completed += 1

    if out.repeats_completed == 0:
        out.reason = "no split left enough candidates for a valid model"
    else:
        all_test = np.concatenate(test_errors) if test_errors else np.zeros(0)
        all_train = np.concatenate(train_errors)
        out.training_reprojection_rmse_px, out.training_median_px, _ = _stats(all_train)
        out.heldout_candidates = candidates
        out.heldout_within_gate = gated
        out.heldout_within_3px = within_three
        out.heldout_within_gate_fraction = gated / candidates if candidates else None
        if search_area_px > 0:
            out.expected_chance_within_gate = float(candidates * math.pi * cfg.gate_px**2 / search_area_px)
        if len(all_test) >= cfg.min_points:
            out.status = "ok"
            out.heldout_transfer_rmse_px, out.heldout_median_px, out.heldout_p90_px = _stats(all_test)
        else:
            out.reason = f"only {len(all_test)} held-out candidates near the model"

    # Block test on the verified inliers, with the final model family.
    idx = final.inlier_idx
    block = block_holdout(src[idx], dst[idx], final, source_size)
    if block is not None:
        out.block_rmse_px, out.block_median_px, out.block_p90_px, out.block_count = block
    return out


def block_holdout(
    src: np.ndarray, dst: np.ndarray, final: VerifiedGeometry, source_size: tuple[int, int], blocks: int = 6
) -> tuple[float, float, float, int] | None:
    """Leave one contiguous stretch of the image out at a time."""

    def fit(train_src, train_dst):
        if final.strip is not None:
            model = fit_strip(train_src, train_dst, source_size, final.strip.spacing_px, final.strip.smoothness)
            return None if model is None else model.apply
        matrix = fit_least_squares(train_src, train_dst, final.kind)
        return None if matrix is None else (lambda points: apply_transform(matrix, points))

    errors = leave_block_out_errors(src, dst, source_size, fit, blocks)
    if not errors:
        return None
    rmse, median, p90 = _stats(np.concatenate(errors))
    return rmse, median, p90, len(errors)


def paired_refinement_holdout(
    src: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    kind: str,
    source_size: tuple[int, int],
    fit: Callable[[np.ndarray, np.ndarray], Callable[[np.ndarray], np.ndarray] | None],
    cfg: HoldoutConfig,
    seed: int = 0,
) -> dict:
    """Same splits, fitted and tested on initial coordinates and on refined coordinates.

    If refinement removes measurement noise rather than adding it, a model fitted on refined
    training points predicts refined test points better than the initial ones predict theirs.
    Only points that refinement accepted are compared, so both sides see the same points.
    """
    rng = np.random.default_rng(seed)
    e_before: list[np.ndarray] = []
    e_after: list[np.ndarray] = []
    for _ in range(cfg.repeats):
        held = stratified_split(src, source_size, cfg.grid_rows, cfg.grid_cols, cfg.fraction, rng)
        if held.sum() < cfg.min_points or (~held).sum() < 8:
            continue
        model_before = fit(src[~held], before[~held])
        model_after = fit(src[~held], after[~held])
        if model_before is None or model_after is None:
            continue
        e_before.append(np.linalg.norm(model_before(src[held]) - before[held], axis=1))
        e_after.append(np.linalg.norm(model_after(src[held]) - after[held], axis=1))
    if not e_before:
        return {"status": "insufficient", "label": "same held-out splits, initial vs refined coordinates"}
    b, a = np.concatenate(e_before), np.concatenate(e_after)
    rmse_b, median_b, p90_b = _stats(b)
    rmse_a, median_a, p90_a = _stats(a)
    return {
        "status": "ok",
        "label": "same held-out splits, initial vs refined coordinates",
        "points": int(len(b)),
        "kind": kind,
        "initial_rmse_px": rmse_b,
        "initial_median_px": median_b,
        "initial_p90_px": p90_b,
        "refined_rmse_px": rmse_a,
        "refined_median_px": median_a,
        "refined_p90_px": p90_a,
        "improved": bool(median_a < median_b and rmse_a < rmse_b),
        "local": local_refinement_check(src, before, after, source_size, seed),
    }


def local_refinement_check(
    src: np.ndarray, before: np.ndarray, after: np.ndarray, source_size: tuple[int, int], seed: int = 0, block_px: float = 500.0, min_points: int = 16
) -> dict:
    """The same comparison with a small affine fitted inside each along-track block.

    A long strip can deviate from any model of the whole image by more than the point noise, which
    then hides whether refinement helped. Within a short stretch a local affine fits the ground, so
    what remains is mostly the error of the points themselves. Still automatic matches, and
    still not ground truth: a bias common to neighbouring points would not show here.
    """
    axis = 1 if source_size[1] >= source_size[0] else 0
    rng = np.random.default_rng(seed)
    blocks = max(1, int(round(source_size[axis] / block_px)))
    labels = np.clip((src[:, axis] / source_size[axis] * blocks).astype(np.int64), 0, blocks - 1)
    e_before: list[np.ndarray] = []
    e_after: list[np.ndarray] = []
    used = 0
    for b in range(blocks):
        members = np.flatnonzero(labels == b)
        if len(members) < min_points:
            continue
        order = rng.permutation(members)
        cut = max(4, len(order) // 3)
        test, train = order[:cut], order[cut:]
        fit_before = fit_least_squares(src[train], before[train], "affine")
        fit_after = fit_least_squares(src[train], after[train], "affine")
        if fit_before is None or fit_after is None:
            continue
        e_before.append(np.linalg.norm(apply_transform(fit_before, src[test]) - before[test], axis=1))
        e_after.append(np.linalg.norm(apply_transform(fit_after, src[test]) - after[test], axis=1))
        used += 1
    if not e_before:
        return {"status": "insufficient", "blocks": 0}
    b, a = np.concatenate(e_before), np.concatenate(e_after)
    rmse_b, median_b, p90_b = _stats(b)
    rmse_a, median_a, p90_a = _stats(a)
    return {
        "status": "ok",
        "blocks": used,
        "points": int(len(b)),
        "initial_rmse_px": rmse_b,
        "initial_median_px": median_b,
        "initial_p90_px": p90_b,
        "refined_rmse_px": rmse_a,
        "refined_median_px": median_a,
        "refined_p90_px": p90_a,
    }


def parse_control_points(text: str) -> list[ControlPoint]:
    """CSV with columns source_x, source_y, reference_x, reference_y (optional id, note)."""
    reader = csv.DictReader(io.StringIO(text.strip()))
    required = {"source_x", "source_y", "reference_x", "reference_y"}
    if reader.fieldnames is None or not required <= {f.strip().lower() for f in reader.fieldnames}:
        raise ValueError("control-point CSV needs the columns source_x, source_y, reference_x, reference_y")
    points: list[ControlPoint] = []
    for number, row in enumerate(reader, start=2):
        row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
        try:
            points.append(
                ControlPoint(
                    source_x=float(row["source_x"]),
                    source_y=float(row["source_y"]),
                    reference_x=float(row["reference_x"]),
                    reference_y=float(row["reference_y"]),
                    id=row.get("id") or None,
                    note=row.get("note") or None,
                )
            )
        except ValueError as exc:
            raise ValueError(f"control-point CSV line {number}: {exc}") from exc
    return points


def evaluate_control_points(
    points: list[ControlPoint],
    apply_model: Callable[[np.ndarray], np.ndarray],
    provenance: str,
    source_size: tuple[int, int] | None = None,
    reference_size: tuple[int, int] | None = None,
) -> ControlPointEvaluation:
    """Measure a finished transformation against independently identified points."""
    warnings: list[str] = []
    if not points:
        raise ValueError("no control points were given")
    src = np.array([[p.source_x, p.source_y] for p in points], dtype=np.float64)
    ref = np.array([[p.reference_x, p.reference_y] for p in points], dtype=np.float64)
    if source_size is not None:
        outside = (src[:, 0] < 0) | (src[:, 1] < 0) | (src[:, 0] >= source_size[0]) | (src[:, 1] >= source_size[1])
        if outside.any():
            warnings.append(f"{int(outside.sum())} control point(s) lie outside the source image")
    if reference_size is not None:
        outside = (ref[:, 0] < 0) | (ref[:, 1] < 0) | (ref[:, 0] >= reference_size[0]) | (ref[:, 1] >= reference_size[1])
        if outside.any():
            warnings.append(f"{int(outside.sum())} control point(s) lie outside the reference image")
    if len(points) < 6:
        warnings.append(f"Only {len(points)} control points: these statistics carry little weight.")
    predicted = apply_model(src)
    errors = np.linalg.norm(predicted - ref, axis=1)
    rmse, median, p90 = _stats(errors)
    return ControlPointEvaluation(
        provenance=provenance,
        count=len(points),
        rmse_px=rmse,
        median_px=median,
        p90_px=p90,
        max_px=float(errors.max()),
        per_point=[
            {
                "id": p.id or str(i + 1),
                "source": [p.source_x, p.source_y],
                "reference": [p.reference_x, p.reference_y],
                "predicted": predicted[i].tolist(),
                "error_px": float(errors[i]),
                "note": p.note,
            }
            for i, p in enumerate(points)
        ],
        warnings=warnings,
    )
