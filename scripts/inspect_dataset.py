"""Phase 1/2: inspect the official mentor dataset (read-only).

    .venv\\Scripts\\python.exe scripts\\inspect_dataset.py
    .venv\\Scripts\\python.exe scripts\\inspect_dataset.py --json results\\dataset_inspection.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from app.config import DATA_ROOT
from app.ingestion.indexer import discover_scenes
from app.ingestion.raster import RasterData, compute_stats, load_raster
from app.metadata.georef import predict_footprint
from app.metadata.xml_parser import parse_scene_xml


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "None"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _describe_raster(label: str, raster: RasterData) -> dict:
    info = raster.info
    stats = compute_stats(raster.data, raster.valid_mask)
    crs = "None" if info.crs is None else info.crs.split("[", 1)[1].split(",", 1)[0].strip('"')
    print(f"  {label}: {Path(info.path).name}")
    print(f"      size (W x H)  : {info.width} x {info.height}   bands={info.band_count}   dtype={info.dtype}")
    print(f"      NoData        : {info.nodata}")
    print(f"      CRS           : {crs}" + ("" if info.crs_label_consistent is not False else "   [LABEL INCONSISTENT WITH GEOTRANSFORM]"))
    print(f"      geotransform  : {[round(v, 6) for v in info.transform] if info.has_geotransform else 'identity (not georeferenced)'}")
    print(f"      embedded GCPs : {len(info.gcps)}")
    print(f"      valid pixels  : {stats.valid_fraction:.2%}   NaN={stats.nan_count} Inf={stats.inf_count} zeros={stats.zero_count} negatives={stats.negative_count}")
    print(f"      values (valid): min={_fmt(stats.minimum)} max={_fmt(stats.maximum)} mean={_fmt(stats.mean)} std={_fmt(stats.std)}")
    print(f"      percentiles   : " + "  ".join(f"{k}={v:.3f}" for k, v in stats.percentiles.items()))
    return {"info": info.model_dump(), "stats": stats.model_dump()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(DATA_ROOT))
    parser.add_argument("--json", help="Also write the full inspection to this JSON file")
    args = parser.parse_args()

    index = discover_scenes(args.data_dir)
    print(f"Dataset root : {index.data_root}")
    if not index.exists:
        raise SystemExit("Dataset directory not found.")
    print(f"Scenes found : {len(index.scenes)}   by sensor: {index.sensor_counts}")
    print(f"TMC-2 data   : {'present' if index.sensor_counts.get('TMC-2') else 'NONE in this dataset'}")
    for item in index.incomplete:
        print(f"INCOMPLETE   : {item.scene_id} is missing {item.missing}")
    for path in index.unpaired_rasters:
        print(f"UNPAIRED TIFF: {path}")

    report = {"index": index.model_dump(), "scenes": []}
    for scene in index.scenes:
        print("\n" + "=" * 100)
        print(f"[{scene.sensor}] {scene.scene_id}   (reference: {scene.reference_sensor})")
        meta = parse_scene_xml(scene.xml_path)
        source = load_raster(scene.source_path)
        reference = load_raster(scene.reference_path)
        entry = {"scene": scene.model_dump(), "metadata": meta.model_dump()}
        entry["source"] = _describe_raster("SOURCE   ", source)
        entry["reference"] = _describe_raster("REFERENCE", reference)

        print(f"  XML      : {Path(scene.xml_path).name}")
        print(f"      sensor            : {meta.sensor}  (inferred from {meta.sensor_basis})")
        print(f"      Resolution_in_meter: {meta.resolution_m}")
        print(f"      Sun azimuth / elev : {meta.sun_azimuth_deg} / {meta.sun_elevation_deg} deg   incidence={meta.solar_incidence_angle_deg}")
        print(f"      roll/pitch/yaw     : {meta.roll_deg} / {meta.pitch_deg} / {meta.yaw_deg} deg   altitude={meta.spacecraft_altitude_km} km")
        print(f"      projection / area  : {meta.projection} / {meta.area}   yaw_direction={meta.spacecraft_yaw_direction}")
        print(f"      product size       : {meta.product_image_width} x {meta.product_image_height}   data_type={meta.data_type}")
        for name, c in (meta.corners or {}).items():
            en = (meta.corners_en or {}).get(name)
            en_txt = f"   E={en.easting:.3f} N={en.northing:.3f}" if en else "   (no easting/northing)"
            print(f"      {name:13s}: lat={c.latitude:.6f} lon={c.longitude:.6f}{en_txt}")
        print(f"      refined corners    : {'present' if meta.corners_refined else 'None'}"
              f"   refined easting/northing: {'present' if meta.corners_refined_en else 'None'}")
        print(f"      duplicated tags    : {meta.duplicated_tag_count}   conflicting: {meta.duplicate_conflicts or 'none'}")

        footprint = predict_footprint(source.info, reference.info, meta)
        entry["predicted_footprint"] = footprint.model_dump() if footprint else None
        if footprint:
            print("  METADATA-PREDICTED FOOTPRINT (plausibility check only, not ground truth)")
            print(f"      corner basis       : {footprint.corner_basis}")
            print(f"      map coordinates    : {footprint.map_coordinate_basis}")
            for s, r, n in zip(footprint.source_corners_px, footprint.reference_corners_px, footprint.xml_corner_names):
                print(f"      source px ({s[0]:6.0f},{s[1]:6.0f}) = XML {str(n):12s} -> reference px ({r[0]:8.1f},{r[1]:8.1f})")
            print(f"      scale x/y={footprint.scale_x:.3f}/{footprint.scale_y:.3f}  rotation={footprint.rotation_deg:.1f} deg"
                  f"  mirrored={footprint.mirrored}  corners inside reference={footprint.corners_inside_reference}/4")
            for note in footprint.notes:
                print(f"      NOTE: {note}")
        report["scenes"].append(entry)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
