"""AUTO: choosing among verified matcher results by transparent rules.

There is no combined score, because a score would have to be calibrated against ground truth
that the official sample does not contain. Instead the candidates are compared one measured
property at a time, in a fixed order, and a difference too small to mean anything does not
decide. The number of matches is deliberately not a criterion: a matcher can return thousands
of matches that say nothing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel

_RANK = {"SUCCESS": 2, "LOW_CONFIDENCE": 1, "FAILED": 0}

# (name, higher is better, smallest difference that counts)
CRITERIA = [
    ("coverage_within_overlap", True, 0.05),
    ("heldout_median_px", False, 0.25),
    ("inlier_ratio", True, 0.02),
    ("reprojection_rmse_px", False, 0.25),
    ("seconds", False, 30.0),
]

RULE = (
    "Accepted results only, then highest status, then in order: spatial coverage within the overlap "
    "(differences under 5 percentage points are ties), held-out median error (under 0.25 px are ties), "
    "inlier ratio (under 2 points), reprojection RMSE (under 0.25 px), processing time (under 30 s). "
    "The number of matches is not a criterion."
)


class SelectionSummary(BaseModel):
    mode: str  # AUTO or the requested matcher
    selected: str | None
    rule: str = RULE
    reasons: list[str] = []
    candidates: list[dict] = []
    agreement: list[dict] = []
    warnings: list[str] = []


@dataclass
class Candidate:
    matcher: str
    status: str
    values: dict  # every name in CRITERIA


def _prefer(a: Candidate, b: Candidate) -> tuple[Candidate, str | None]:
    """The better of two candidates and why, or (a, None) if no measured property separates them."""
    if _RANK[a.status] != _RANK[b.status]:
        better, worse = (a, b) if _RANK[a.status] > _RANK[b.status] else (b, a)
        return better, f"{better.matcher} over {worse.matcher}: status {better.status} vs {worse.status}"
    for name, higher, tie in CRITERIA:
        va, vb = a.values.get(name), b.values.get(name)
        if va is None or vb is None:
            continue  # only properties measured on both sides can decide
        if abs(va - vb) < tie:
            continue
        a_wins = (va > vb) == higher
        better, worse = (a, b) if a_wins else (b, a)
        vbetter, vworse = (va, vb) if a_wins else (vb, va)
        return better, f"{better.matcher} over {worse.matcher}: {name} {vbetter:.3f} vs {vworse:.3f}"
    return a, None


def select(candidates: list[Candidate], mode: str = "AUTO") -> SelectionSummary:
    """Compare accepted candidates pairwise; returns who won and every comparison that decided it."""
    table = [{"matcher": c.matcher, "status": c.status, **c.values} for c in candidates]
    accepted = [c for c in candidates if c.status != "FAILED"]
    if not accepted:
        return SelectionSummary(mode=mode, selected=None, candidates=table, reasons=["No matcher produced an accepted result."])
    best = accepted[0]
    reasons: list[str] = []
    for other in accepted[1:]:
        winner, why = _prefer(best, other)
        if why is None:
            why = f"{best.matcher} and {other.matcher} are tied on every criterion; {best.matcher} was evaluated first"
        reasons.append(why)
        best = winner
    if len(accepted) == 1:
        reasons.append(f"{best.matcher} is the only matcher with an accepted result.")
    return SelectionSummary(mode=mode, selected=best.matcher, reasons=reasons, candidates=table)


def model_disagreement(apply_a, apply_b, points: np.ndarray) -> dict | None:
    """How far two models place the same source points apart, in reference pixels."""
    if len(points) == 0:
        return None
    distance = np.linalg.norm(apply_a(points) - apply_b(points), axis=1)
    return {
        "points": int(len(points)),
        "median_px": float(np.median(distance)),
        "p90_px": float(np.percentile(distance, 90)),
        "max_px": float(distance.max()),
    }
