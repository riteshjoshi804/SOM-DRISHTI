"""Registration-ready deliverables: correspondence tables, transformation, metadata, metrics, report.

Everything here is derived from a finished run. Geospatial values are copied only from what
the run actually established; nothing is invented to make a file look complete.
"""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from app.registration.schemas import RegistrationResult

if TYPE_CHECKING:
    from app.registration.engine import BranchOutcome


def _table(path: Path, header: list[str], rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def _f(value: float) -> str:
    return f"{float(value):.4f}"


def write_correspondence_tables(out: Path, o: "BranchOutcome", prefix: str = "") -> dict[str, str]:
    """Raw, verified, selected and refined correspondences as CSV, all in original pixels."""
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    key = prefix.replace("__", "_")
    src, ref = o.result.source_points, o.result.reference_points
    conf, rank = o.result.confidence_scores, o.result.rank
    if len(src):
        name = f"{prefix}correspondences_raw.csv"
        _table(
            out / name,
            ["source_x", "source_y", "reference_x", "reference_y", "confidence", "rank"],
            ([_f(s[0]), _f(s[1]), _f(r[0]), _f(r[1]), _f(c), _f(k)] for s, r, c, k in zip(src, ref, conf, rank)),
        )
        files[f"{key}correspondences_raw"] = name
    g = o.geometry
    if g is None or not o.accepted:
        return files

    idx = g.inlier_idx
    in_src, in_ref, in_conf = src[idx], ref[idx], conf[idx]
    residual = g.errors(in_src, in_ref)
    name = f"{prefix}correspondences_inliers.csv"
    _table(
        out / name,
        ["source_x", "source_y", "reference_x", "reference_y", "confidence", "residual_px"],
        ([_f(s[0]), _f(s[1]), _f(r[0]), _f(r[1]), _f(c), _f(e)] for s, r, c, e in zip(in_src, in_ref, in_conf, residual)),
    )
    files[f"{key}correspondences_inliers"] = name

    if o.selected_idx is not None:
        sel = o.selected_idx
        name = f"{prefix}tie_points_selected.csv"
        _table(
            out / name,
            ["source_x", "source_y", "reference_x", "reference_y", "confidence", "residual_px"],
            ([_f(in_src[i, 0]), _f(in_src[i, 1]), _f(in_ref[i, 0]), _f(in_ref[i, 1]), _f(in_conf[i]), _f(residual[i])] for i in sel),
        )
        files[f"{key}tie_points_selected"] = name

    if o.refined is not None:
        r = o.refined
        selected = np.zeros(len(r.source), bool)
        if o.selected_idx is not None:
            selected[o.selected_idx] = True
        name = f"{prefix}tie_points_refined.csv"
        _table(
            out / name,
            [
                "source_x", "source_y", "reference_x_initial", "reference_y_initial", "reference_x_refined", "reference_y_refined",
                "displacement_x", "displacement_y", "refinement_score", "refined", "selected_tie_point", "outcome",
            ],  # fmt: skip
            (
                [
                    _f(r.source[i, 0]), _f(r.source[i, 1]), _f(r.reference_initial[i, 0]), _f(r.reference_initial[i, 1]),
                    _f(r.reference_refined[i, 0]), _f(r.reference_refined[i, 1]), _f(r.displacement[i, 0]), _f(r.displacement[i, 1]),
                    "" if not np.isfinite(r.score[i]) else _f(r.score[i]), int(r.refined[i]), int(selected[i]), r.reason[i],
                ]  # fmt: skip
                for i in range(len(r.source))
            ),
        )
        files[f"{key}tie_points_refined"] = name
    return files


def _dumps(value) -> str:
    return json.dumps(value, indent=2, default=str)


BENCHMARK_COLUMNS = [
    "scene_id", "sensor", "reference_sensor", "matcher", "registration_status", "device", "representation", "transformation_model",
    "source_keypoints", "reference_keypoints", "candidate_correspondences", "accepted_correspondences", "geometric_inliers",
    "inlier_ratio", "reprojection_rmse_px", "reprojection_median_px", "heldout_rmse_px", "heldout_median_px", "heldout_p90_px",
    "block_heldout_rmse_px", "source_spatial_coverage", "overlap_spatial_coverage", "along_track_support", "estimated_overlap_fraction",
    "selected_tie_points", "distribution_score", "refined_points", "median_refinement_px", "refinement_improved_heldout",
    "decoy_max_inliers", "appearance_median_zncc", "processing_seconds", "reasons",
]  # fmt: skip


def benchmark_rows(result: RegistrationResult) -> list[dict]:
    """One row per matcher branch of a run, measured under the same protocol."""
    rows = []
    for m in result.matcher_results:
        h, t, r, v = m.holdout, m.tie_points, m.refinement, m.refinement_validation or {}
        measured = [d.inliers for d in m.decoys if d.tiles_used > 0]
        rows.append(
            {
                "scene_id": result.scene_id,
                "sensor": result.sensor,
                "reference_sensor": result.reference_sensor,
                "matcher": m.matcher,
                "registration_status": "UNAVAILABLE" if m.availability == "UNAVAILABLE" else m.registration_status,
                "device": m.device,
                "representation": m.representation,
                "transformation_model": m.transformation_type,
                "source_keypoints": m.source_keypoints,
                "reference_keypoints": m.reference_keypoints,
                "candidate_correspondences": m.candidate_count,
                "accepted_correspondences": m.accepted_count,
                "geometric_inliers": m.inliers,
                "inlier_ratio": m.inlier_ratio,
                "reprojection_rmse_px": m.reprojection_rmse_px,
                "reprojection_median_px": m.median_reprojection_error_px,
                "heldout_rmse_px": h.heldout_transfer_rmse_px if h else None,
                "heldout_median_px": h.heldout_median_px if h else None,
                "heldout_p90_px": h.heldout_p90_px if h else None,
                "block_heldout_rmse_px": h.block_rmse_px if h else None,
                "source_spatial_coverage": m.image_space_spatial_coverage,
                "overlap_spatial_coverage": m.coverage_within_overlap,
                "along_track_support": m.along_track_support,
                "estimated_overlap_fraction": m.estimated_overlap_fraction,
                "selected_tie_points": t.selected if t else None,
                "distribution_score": t.distribution_score if t else None,
                "refined_points": r.refined if r else None,
                "median_refinement_px": r.median_displacement_px if r else None,
                "refinement_improved_heldout": v.get("improved"),
                "decoy_max_inliers": max(measured) if measured else None,
                "appearance_median_zncc": m.appearance.median_zncc if m.appearance else None,
                "processing_seconds": m.processing_time_seconds,
                "reasons": "; ".join(m.reasons or m.warnings),
            }
        )
    return rows


def _fmt(value, digits: int = 3, unit: str = "") -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}{unit}"
    return f"{value}{unit}"


def _pct(value, digits: int = 1) -> str:
    return "N/A" if value is None else f"{value * 100:.{digits}f}%"


def report_sections(result: RegistrationResult) -> list[tuple[str, list]]:
    """The report as (heading, rows) sections; a row is (label, value) or a plain sentence."""
    m = result.metrics
    meta = result.metadata
    overlap = result.overlap_estimate
    h = result.holdout
    t = result.tie_points
    r = result.refinement
    v = result.refinement_validation or {}
    sections: list[tuple[str, list]] = []

    sections.append(
        (
            "Result",
            [
                ("Registration status", result.status),
                ("Selected matcher", result.selected_matcher or "none accepted"),
                ("Matcher requested", result.requested_matcher),
                ("Device", result.device),
                ("Processing time", _fmt(m.processing_time_seconds, 1, " s")),
                *[("Why it failed", x) for x in result.failure_reasons],
                *[("Why not higher", x) for x in result.warnings],
            ],
        )
    )
    sections.append(
        (
            "Scene",
            [
                ("Scene ID", result.scene_id or "uploaded pair"),
                ("Sensor", result.sensor),
                ("Reference sensor", result.reference_sensor),
                ("Source image", f"{result.source.path} ({result.source.width} x {result.source.height} px, {result.source.dtype})"),
                ("Reference image", f"{result.reference.path} ({result.reference.width} x {result.reference.height} px, {result.reference.dtype})"),
            ],
        )
    )
    if meta is not None:
        sections.append(
            (
                "XML metadata summary (only values present in the file)",
                [
                    ("Native resolution", _fmt(meta.resolution_m, 2, " m")),
                    ("Sun azimuth / elevation", f"{_fmt(meta.sun_azimuth_deg, 2)} / {_fmt(meta.sun_elevation_deg, 2)} deg"),
                    ("Projection / area", f"{meta.projection} / {meta.area}"),
                    ("Date of pass", meta.date_of_pass),
                    ("Reference used by the product", meta.reference_used),
                ],
            )
        )
    if overlap is not None:
        rows = [("Search mode", overlap.mode), ("Basis", overlap.reason)]
        if overlap.coordinate_system is not None:
            c = overlap.coordinate_system
            rows.extend(
                [
                    ("Reference body (from its CRS)", c.body),
                    ("Reference CRS kind / projection / units", f"{c.crs_kind} / {c.projection} / {c.units}"),
                    ("CRS label agrees with geotransform", _fmt(c.label_consistent)),
                ]
            )
        rows.extend(
            [
                ("Prior search radius", _fmt(overlap.search_radius_px, 0, " px")),
                ("Footprint inside the reference (prior)", _pct(overlap.footprint_fraction_in_reference)),
                ("Footprint on valid reference pixels (prior)", _pct(overlap.footprint_fraction_on_valid_reference)),
                ("Estimated overlap (from the image result)", _pct(m.estimated_overlap_fraction)),
            ]
        )
        anchor = (result.search_trace or {}).get("anchor")
        if anchor:
            rows.append(("Location found by", anchor.get("method")))
            rows.append(("Orientation found", anchor.get("hypothesis")))
            rows.append(("Offset of the result from the prior", _fmt(anchor.get("offset_from_prior_m"), 0, " m")))
        rows.extend(("Note", w) for w in overlap.warnings)
        sections.append(("Footprint and overlap", rows))
    if result.illumination is not None:
        i = result.illumination
        sections.append(
            (
                "Illumination (descriptive only; no physical correction is applied)",
                [
                    ("Regime from the XML Sun elevation", i.regime),
                    ("Shadow fraction, source / reference (image-based)", f"{_pct(i.source_shadow_fraction)} / {_pct(i.reference_shadow_fraction)}"),
                    ("Angle between shading axes in the reference frame (image-based)", _fmt(i.shading_axis_difference_deg, 1, " deg")),
                    *[("Note", n) for n in i.notes],
                ],
            )
        )

    sections.append(
        (
            "A. Geometric consistency (fitted points; not accuracy)",
            [
                ("Transformation model", m.transformation_type),
                ("Candidates / accepted", f"{_fmt(m.candidate_count)} / {_fmt(m.accepted_count)}"),
                ("Inliers", _fmt(m.inliers)),
                ("Inlier ratio", _pct(m.inlier_ratio, 2)),
                ("Reprojection RMSE / median", f"{_fmt(m.reprojection_rmse_px, 3)} / {_fmt(m.median_reprojection_error_px, 3)} px"),
                ("Decoy inliers on known-wrong ground (highest)", _fmt(m.decoy_max_inliers)),
                ("Patch similarity at matches (median ZNCC)", _fmt(m.appearance_median_zncc, 3)),
                ("Tie-point support along the strip", _pct(m.supported_along_track_fraction, 0)),
            ],
        )
    )
    rows = [
        "Automatic matches left out of the fit. Real out-of-sample evidence, but from the same matcher, so not independent ground truth."
    ]
    if h is not None:
        rows.extend(
            [
                ("Status", h.status if h.status == "ok" else f"{h.status}: {h.reason}"),
                ("Splits completed", f"{h.repeats_completed} of {h.repeats_requested}, {_pct(h.fraction, 0)} held out"),
                ("Training reprojection RMSE", _fmt(h.training_reprojection_rmse_px, 3, " px")),
                ("Held-out transfer RMSE (within the gate)", _fmt(h.heldout_transfer_rmse_px, 3, " px")),
                ("Held-out median / 90th percentile", f"{_fmt(h.heldout_median_px, 3)} / {_fmt(h.heldout_p90_px, 3)} px"),
                ("Held-out candidates within the gate", f"{h.heldout_within_gate} of {h.heldout_candidates} (gate {_fmt(h.gate_px, 0)} px; expected by chance {_fmt(h.expected_chance_within_gate, 1)})"),
                ("Unseen along-track blocks: RMSE / median", f"{_fmt(h.block_rmse_px, 3)} / {_fmt(h.block_median_px, 3)} px over {h.block_count} blocks"),
            ]
        )
    else:
        rows.append(("Status", "not measured"))
    sections.append(("B. Held-out validation", rows))
    rows = ["Independently identified points, never used in a fit. This is the only figure that may be called independent accuracy."]
    if result.control_points is not None:
        cp = result.control_points
        rows.extend(
            [
                ("Provenance", cp.provenance),
                ("Control points", cp.count),
                ("RMSE / median / 90th percentile", f"{_fmt(cp.rmse_px, 3)} / {_fmt(cp.median_px, 3)} / {_fmt(cp.p90_px, 3)} px"),
                *[("Warning", w) for w in cp.warnings],
            ]
        )
    else:
        rows.append(("Status", "not available: no independent control points have been supplied"))
    sections.append(("C. Independent control-point accuracy", rows))

    rows = []
    if m.image_space_spatial_coverage is not None:
        rows.extend(
            [
                ("Inliers: source-image coverage (8 x 8)", f"{_pct(m.image_space_spatial_coverage)} ({_fmt(m.occupied_cells)} of {_fmt(m.valid_cells)} cells)"),
                ("Inliers: overlap coverage", _pct(m.coverage_within_overlap)),
            ]
        )
    if t is not None:
        rows.extend(
            [
                ("Selected tie points", f"{t.selected} of {t.verified_inliers} verified inliers"),
                ("Selected: source / overlap coverage", f"{_pct(t.source_coverage)} / {_pct(t.overlap_coverage)}"),
                ("Tie-point density", f"{_fmt(t.tie_points_per_km2, 2)} per km2"),
                ("Distribution score", f"{_fmt(t.distribution_score, 3)} ({t.distribution_definition})"),
                ("Convex hull of the tie points, share of overlap", _pct(t.hull_fraction_of_overlap)),
            ]
        )
        if t.below_minimum:
            rows.append(("Warning", f"fewer than the requested minimum of {t.minimum_requested}; no points were invented"))
    sections.append(("Spatial distribution", rows or ["Not measured: no accepted transformation."]))

    rows = ["Decimal coordinates alone prove nothing; the comparison below is what bears on sub-pixel claims."]
    if r is not None:
        rows.extend(
            [
                ("Strategy / representation", f"{r.strategy} / {r.representation}"),
                ("Refined / rejected", f"{r.refined} / {r.rejected}"),
                ("Displacement median / 90th percentile", f"{_fmt(r.median_displacement_px, 3)} / {_fmt(r.p90_displacement_px, 3)} px"),
                ("Rejections", ", ".join(f"{k}: {n}" for k, n in r.rejected_by_reason.items()) or "none"),
            ]
        )
        if v.get("status") == "ok":
            rows.extend(
                [
                    ("Held-out median, initial vs refined", f"{_fmt(v.get('initial_median_px'), 3)} vs {_fmt(v.get('refined_median_px'), 3)} px"),
                    ("Held-out RMSE, initial vs refined", f"{_fmt(v.get('initial_rmse_px'), 3)} vs {_fmt(v.get('refined_rmse_px'), 3)} px"),
                    ("Refinement improved held-out agreement", _fmt(v.get("improved"))),
                    *(
                        [("Local affine per along-track block, held-out median, initial vs refined", f"{_fmt(v['local'].get('initial_median_px'), 3)} vs {_fmt(v['local'].get('refined_median_px'), 3)} px over {v['local'].get('blocks')} blocks")]
                        if (v.get("local") or {}).get("status") == "ok"
                        else []
                    ),
                ]
            )
        if r.systematic_shift:
            rows.append(("Warning", f"points moved coherently ({_fmt(r.median_shift_x_px, 2)}, {_fmt(r.median_shift_y_px, 2)} px): the correlation may be following shading, so refined coordinates should not be trusted"))
        rows.append(("Model refitted on refined points", _fmt(r.model_refit)))
    else:
        rows.append(("Status", "not run"))
    sections.append(("Sub-pixel refinement", rows))

    if result.transformation is not None:
        tr = result.transformation
        rows = [
            ("Model", tr.model),
            ("Estimator", tr.method),
            ("Direction", tr.direction),
            ("Mirrored source", _fmt(tr.mirrored_source)),
            ("Matrix" if tr.strip_model is None else "Matrix (best affine approximation only)", " | ".join(" ".join(f"{x:.6f}" for x in row) for row in tr.matrix)),
        ]
        if tr.strip_check is not None:
            rows.append(("Largest deviation from the affine approximation", _fmt(tr.strip_check.max_deviation_from_affine_px, 1, " px")))
        sections.append(("Transformation", rows))
    if result.matcher_results:
        rows = []
        for a in result.matcher_results:
            held = a.holdout.heldout_median_px if a.holdout else None
            status = "UNAVAILABLE" if a.availability == "UNAVAILABLE" else a.registration_status
            why = "; ".join(a.reasons or a.warnings) or "-"
            rows.append(
                (
                    a.matcher,
                    f"{status}, {_fmt(a.inliers)} inliers, ratio {_pct(a.inlier_ratio, 1)}, coverage {_pct(a.coverage_within_overlap)}, "
                    f"held-out median {_fmt(held, 3)} px, {_fmt(a.processing_time_seconds, 1)} s, {why}",
                )
            )
        if result.selection is not None:
            rows.append(("Selection rule", result.selection.rule))
            rows.extend(("Decision", x) for x in result.selection.reasons)
            for g in result.selection.agreement:
                rows.append(
                    (
                        f"{g['other']} tie points under the {g['selected']} model",
                        f"median {g['median_px']:.2f} px, 90th percentile {g['p90_px']:.2f} px, {g['within_threshold']} of {g['points']} within the threshold",
                    )
                )
            rows.extend(("Warning", x) for x in result.selection.warnings)
        sections.append(("Matchers", rows))
    sections.append(("Notes", list(result.notes)))
    return sections


def _markdown(result: RegistrationResult, sections: list[tuple[str, list]]) -> str:
    lines = [f"# SOM DRISHTI registration report: {result.scene_id or 'uploaded pair'}", ""]
    lines.append("Every value is measured on this run; N/A means not available. Nothing here is a ground-truth accuracy unless section C says so.")
    for heading, rows in sections:
        lines.extend(["", f"## {heading}", ""])
        for row in rows:
            if isinstance(row, tuple):
                lines.append(f"- **{row[0]}:** {row[1]}")
            else:
                lines.append(str(row))
    return "\n".join(lines) + "\n"


def _html(result: RegistrationResult, sections: list[tuple[str, list]]) -> str:
    title = html.escape(f"SOM DRISHTI registration report: {result.scene_id or 'uploaded pair'}")
    colour = {"SUCCESS": "#0ca30c", "LOW_CONFIDENCE": "#b07b00", "FAILED": "#d03b3b"}.get(result.status, "#555")
    parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        f"<title>{title}</title>",
        "<style>body{font:15px/1.5 system-ui,sans-serif;max-width:60rem;margin:2rem auto;padding:0 1rem;color:#1a1a19}"
        "h1{font-size:1.5rem;word-break:break-all}h2{font-size:1.1rem;margin-top:1.8rem;border-bottom:1px solid #ddd;padding-bottom:.2rem}"
        "table{border-collapse:collapse;width:100%}td{padding:.2rem .6rem;vertical-align:top;border-bottom:1px solid #eee}"
        "td:first-child{color:#555;width:38%}.status{font-weight:700}p.small{color:#555;font-size:.9rem}</style></head><body>",
        f"<h1>{title}</h1>",
        f"<p class='status' style='color:{colour}'>{html.escape(result.status)}</p>",
        "<p class='small'>Every value is measured on this run; N/A means not available. Nothing here is a ground-truth accuracy unless section C says so.</p>",
    ]
    for heading, rows in sections:
        parts.append(f"<h2>{html.escape(heading)}</h2>")
        pairs = [row for row in rows if isinstance(row, tuple)]
        for row in rows:
            if not isinstance(row, tuple):
                parts.append(f"<p>{html.escape(str(row))}</p>")
        if pairs:
            parts.append("<table>")
            parts.extend(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>" for k, v in pairs)
            parts.append("</table>")
    parts.append("</body></html>")
    return "\n".join(parts)


def write_result_documents(out: Path, result: RegistrationResult) -> dict[str, str]:
    """Transformation, metadata and metrics JSON, the benchmark row and the scientific report."""
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    if result.transformation is not None:
        (out / "transformation.json").write_text(result.transformation.model_dump_json(indent=2), encoding="utf-8")
        files["transformation"] = "transformation.json"
    summary = {
        "scene_id": result.scene_id,
        "sensor": result.sensor,
        "reference_sensor": result.reference_sensor,
        "source": result.source.model_dump(exclude={"stats"}),
        "reference": result.reference.model_dump(exclude={"stats"}),
        "xml": result.metadata.model_dump(exclude={"raw_fields"}) if result.metadata else None,
        "predicted_footprint": result.predicted_footprint.model_dump() if result.predicted_footprint else None,
        "overlap_estimate": result.overlap_estimate.model_dump() if result.overlap_estimate else None,
        "illumination": result.illumination.model_dump() if result.illumination else None,
    }
    (out / "metadata_summary.json").write_text(_dumps(summary), encoding="utf-8")
    files["metadata_summary"] = "metadata_summary.json"
    metrics = {
        "status": result.status,
        "failure_reasons": result.failure_reasons,
        "warnings": result.warnings,
        "metrics": result.metrics.model_dump(),
        "holdout": result.holdout.model_dump() if result.holdout else None,
        "control_points": result.control_points.model_dump() if result.control_points else None,
        "tie_points": result.tie_points.model_dump() if result.tie_points else None,
        "refinement": result.refinement.model_dump() if result.refinement else None,
        "refinement_validation": result.refinement_validation,
        "decoys": [d.model_dump() for d in result.decoys],
        "selection": result.selection.model_dump() if result.selection else None,
        "evidence": result.evidence.model_dump() if result.evidence else None,
    }
    (out / "metrics.json").write_text(_dumps(metrics), encoding="utf-8")
    files["metrics"] = "metrics.json"
    rows = benchmark_rows(result)
    with (out / "benchmark_row.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=BENCHMARK_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    files["benchmark_row"] = "benchmark_row.csv"
    sections = report_sections(result)
    (out / "report.md").write_text(_markdown(result, sections), encoding="utf-8")
    (out / "report.html").write_text(_html(result, sections), encoding="utf-8")
    files["report_markdown"] = "report.md"
    files["report_html"] = "report.html"
    return files
