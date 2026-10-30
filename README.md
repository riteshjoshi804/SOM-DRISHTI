# SOM DRISHTI

**Lunar Image Correspondence & Registration System**
Smart India Hackathon 2026 · Problem statement **SIH26166** · ISRO / Department of Space

> Multi-modal, Sun-angle and scale invariant image correspondence using Chandrayaan-2
> optical images (OHRC, TMC and IIRS).

This repository is the **classical MVP**: SIFT features, ratio-test matching and robust
geometry, run on the official sample dataset supplied by the problem-statement mentor.
Every number it reports is measured. When registration fails, it says so.

---

## 1. Objective

Register a Chandrayaan-2 source image onto its LRO reference image without assuming
that the source lies entirely inside the reference, then report how well-supported
that registration is:

1. discover the official source / reference / XML triplets,
2. parse the XML metadata and load the rasters with their original values,
3. find SIFT correspondences and reject wrong ones with robust geometry,
4. estimate and validate a source → reference transformation (affine or homography),
5. warp the source into the reference grid and render overlays,
6. compute real metrics and a SUCCESS / LOW_CONFIDENCE / FAILED status with reasons,
7. expose all of it through a FastAPI backend and a React interface.

## 2. Official sample dataset

Place the extracted mentor ZIP so that it looks like this (the files are treated as
**read-only**; nothing is ever written next to them):

```
data/official_sih_2026/
├── ohrc/
│   ├── <scene>.xml
│   ├── <scene>_source_correctly_aligned_at_5m.tif     # OHRC, uint8, resampled to 5 m, no CRS, 4 GCPs
│   └── <scene>_reference_at_5m.tif                    # LRO-NAC, uint8, 5 m polar stereographic, NoData 0
└── IIRS/
    ├── <scene>.xml
    ├── <scene>_source.tif                             # IIRS, float32
    └── <scene>_reference.tif                          # LRO-WAC, float32
```

| Sensor | Scenes | Reference | Notes |
|---|---|---|---|
| OHRC | 4 | LRO-NAC | XML resolution 0.23–0.27 m, delivered at 5 m; Sun elevation −0.3° to 5.1° |
| IIRS | 2 | LRO-WAC | XML resolution 93.74 m and 83.14 m; float32 radiance; strips 104–250 px wide |
| TMC-2 | **0** | – | No official sample provided. **TMC-2 is not validated.** |

Scenes are discovered automatically: each XML defines a scene ID and is paired with the
TIFFs named `<scene_id>_source*` and `<scene_id>_reference*`. Several inconsistencies in
the sample data (XML corner names that do not match the TIFF array corners, a mirrored
source, a mislabelled CRS, resampled IIRS strips, km-level metadata geolocation offsets)
are documented in [docs/engineering_findings.md](docs/engineering_findings.md).

## 3. Pipeline

### OHRC → LRO-NAC

1. **Load** with Rasterio; build a valid-pixel mask (NoData, NaN/Inf, zero fill touching the border).
2. **Represent** each image three ways: `raw_normalized` (1–99 % percentile stretch),
   `clahe`, `gradient`. None of these removes real shadow differences; they are tested, not trusted.
3. **SIFT** per 768 px tile (3000 keypoints per tile, contrastThreshold 0.02) inside an
   eroded valid mask; RootSIFT descriptors. Images are never resized to each other.
4. **Match** source → reference with brute-force 2-NN, Lowe ratio 0.90, one-to-one filter.
5. **Verify**: affine RANSAC seed on the most distinctive matches, then expansion over all
   ratio-filtered matches near the seed (MAGSAC, 3 px). Homography only when clearly better.
6. **Mirrored source**: if the delivered source gives no accepted geometry, retry it flipped left-right.
7. **Pool** the candidates of all representations if none alone reaches SUCCESS.
8. **Validate** the transform (determinant, scale, anisotropy, shear, perspective, footprint inside reference).
9. **Warp** with `cv2.warpAffine` / `cv2.warpPerspective`; write PNG previews, a colour overlay
   and a GeoTIFF with the original source values on the reference grid.

### IIRS → LRO-WAC

Same engine. The float32 radiance is percentile-normalised on valid pixels only (never
cast to uint8), and **all three representations are always evaluated** so the benchmark
shows which one gives the strongest valid geometry – or that none does.

## 4. Technology stack

Python 3.11+ (developed on 3.13) · FastAPI · Uvicorn · Pydantic · OpenCV (SIFT, RANSAC,
USAC_MAGSAC) · NumPy · SciPy · Rasterio (bundled GDAL) · PyProj · React 19 · Vite 7 ·
Tailwind CSS 4. No API keys and no external services.

## 5. Installation (Windows, PowerShell, VS Code)

```powershell
# from the project root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
```

If PowerShell blocks `Activate.ps1`, skip activation and call the interpreter directly –
every command below works that way:

```powershell
.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

Frontend (use `npm.cmd` if `npm.ps1` is blocked):

```powershell
cd frontend
npm.cmd install
cd ..
```

npm 11 may print `allow-scripts ... esbuild (postinstall)`. It can be ignored – esbuild's
Windows binary is installed as a normal dependency and the build works without that script.

Optional: copy `.env.example` to `.env` to change the data or results folder.

## 6. Running

All commands are run from the project root.

**Inspect the dataset** (dimensions, dtypes, NoData, CRS, geotransform, GCPs, XML fields,
metadata-predicted footprint):

```powershell
.venv\Scripts\python.exe scripts\inspect_dataset.py --json results\dataset_inspection.json
```

**Register one scene** (default: OHRC 2359602, the demonstration case; any unique part of a scene ID works):

```powershell
.venv\Scripts\python.exe scripts\run_registration.py
.venv\Scripts\python.exe scripts\run_registration.py --scene 2359602
.venv\Scripts\python.exe scripts\run_registration.py --scene 2736702 --modes clahe --ratio 0.85 --model homography
```

Artifacts go to `results/jobs/<scene_id>__<timestamp>/`: `result.json`, `source.jpg`,
`reference.jpg`, `keypoints_*.jpg`, `raw_matches.jpg`, `inlier_matches.jpg`,
`spatial_coverage.jpg`, `reference_footprint.jpg`, and – only for an accepted
transformation – `registered_source.png`, `overlay.png`, `registered_source.tif`.

**Benchmarks**:

```powershell
.venv\Scripts\python.exe scripts\benchmark_ohrc.py     # -> results\ohrc_benchmark.csv
.venv\Scripts\python.exe scripts\benchmark_iirs.py     # -> results\iirs_benchmark.csv
```

Each also writes `*_attempts.csv` (every representation / orientation tried) and a `.json`
with the exact configuration used.

**Backend** (http://127.0.0.1:8000, interactive docs at `/docs`):

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --port 8000
```

**Frontend** – development server with hot reload (proxies `/api` to the backend):

```powershell
cd frontend
npm.cmd run dev          # http://localhost:5173
```

or build once and let FastAPI serve it at http://127.0.0.1:8000:

```powershell
cd frontend
npm.cmd run build
```

**Tests**:

```powershell
.venv\Scripts\python.exe -m pytest
```

### API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status, library versions, SIFT / MAGSAC availability |
| GET | `/api/dataset` | scene counts per sensor, TMC-2 presence, incomplete triplets |
| GET | `/api/scenes` | discovered scenes with key metadata |
| GET | `/api/scenes/{scene_id}` | full XML metadata, raster info, statistics, histograms, predicted footprint |
| GET | `/api/scenes/{scene_id}/preview/{source\|reference}` | percentile-stretched preview JPEG |
| POST | `/api/register/{scene_id}` | start a registration job (optional body `{"config": {...}}` with partial overrides) |
| POST | `/api/register/upload` | register an uploaded source/reference (+ optional XML); stored under `results/uploads/` |
| GET | `/api/results` | all jobs, including those produced by the scripts |
| GET | `/api/results/{job_id}` | job state/progress, and the full result when done |
| GET | `/api/results/{job_id}/metrics` | status, reasons and metrics only |
| GET | `/api/results/{job_id}/artifacts/{name}` | an artifact image / GeoTIFF / result.json |
| GET | `/api/config/default` | the default pipeline configuration |

## 7. Results on the official sample data

Produced by `scripts/benchmark_ohrc.py` and `scripts/benchmark_iirs.py` with the default
configuration (full tables: `results/ohrc_benchmark.csv`, `results/iirs_benchmark.csv`,
per-representation attempts in `*_attempts.csv`; discussion in
[docs/benchmark_results.md](docs/benchmark_results.md)).

| Sensor | Scene | Status | Candidates | Inliers | Inlier ratio | Reproj. RMSE (px) | Median (px) | CV RMSE (px) | Coverage (8×8) | Time (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OHRC | 2359602 V1_0_03 | **SUCCESS** (source mirrored) | 800 | 229 | 28.6 % | 1.54 | 1.29 | 1.50 | 29.7 % | 19.1 |
| OHRC | 2436502 V2_1_01 | **FAILED** – no consensus | 2 526 | 0 | 0 % | N/A | N/A | N/A | N/A | 49.0 |
| OHRC | 2470502 V2_1_02 | **LOW_CONFIDENCE** | 1 940 | 19 | 0.98 % | 0.51 | 0.34 | 0.70 | 6.2 % | 43.7 |
| OHRC | 2736702 V1_0_00 | **LOW_CONFIDENCE** | 1 850 | 32 | 1.73 % | 1.45 | 1.32 | 1.70 | 14.1 % | 29.0 |
| IIRS | 2686502 V2_1 | **FAILED** – no consensus | 3 465 | 0 | 0 % | N/A | N/A | N/A | N/A | 44.4 |
| IIRS | 1519402 V1_1_01 | **LOW_CONFIDENCE** | 1 278 | 43 | 3.36 % | 1.13 | 0.52 | 1.34 | 17.2 % | 18.7 |

Reading the table honestly:

* **One OHRC scene registers well** (2359602): 229 tie points, and only after the engine
  mirrors the source – that sample is delivered mirrored relative to its reference.
* **Two OHRC scenes and one IIRS scene register with weak support.** Their tie points are
  far above chance (expected chance inliers ≤ 0.004) and their rotation agrees with the
  XML/GCP metadata to within 1.7°, but they cover only 6–17 % of the source grid, so the
  transformation is poorly constrained away from those points. They are reported as
  LOW_CONFIDENCE, not SUCCESS.
* **One OHRC scene and one IIRS scene fail** under every representation, orientation
  and setting tried. They are the clearest evidence that classical SIFT does not solve
  the Sun-angle and multi-modal problem and that learned matchers are needed.
* Accepted registrations sit 2.3–6 km (OHRC) and ~19 km (IIRS) from the footprint
  predicted by the XML/GCP metadata. The metadata is system-level pointing, not ground truth.
* The OHRC a-priori first case (2436502, highest Sun elevation) is one of the failures,
  so `run_registration.py` defaults to 2359602 for demonstrations.
* For IIRS, `raw_normalized` (23 tie points) and `gradient` (20) outperform `clahe` (9) on
  1519402; pooling all three gives 43.

## 8. Metrics

| Metric | Definition |
|---|---|
| `source_keypoints`, `reference_keypoints` | SIFT keypoints inside the valid masks |
| `raw_matches` | source keypoints that received a 2-NN answer |
| `ratio_filtered_matches` | survivors of the Lowe ratio test after one-to-one filtering ("candidate matches") |
| `inliers` | candidates consistent with the final transformation within 3 px, counted once per location |
| `inlier_ratio` | inliers / ratio-filtered matches |
| `expected_chance_inliers` | wrong matches expected to agree with the model by coincidence (`N·π·t²/A`); compare with `inliers` |
| **Reprojection RMSE (px)**, median, max | residuals of the inliers under the fitted model – **internal consistency, not ground-truth accuracy** |
| `cross_validated_rmse_px` | 5-fold: each inlier's error when held out of the fit; less optimistic, still not ground truth |
| **Image-space Spatial Coverage** | occupied / valid cells of an 8×8 grid over the source image. It is coverage of the image grid, not a geographic area |
| `coverage_within_overlap` | the same, counting only cells that overlap valid reference pixels (partial overlap) |
| `estimated_overlap_fraction` | valid source pixels that land on valid reference pixels |
| `processing_time_seconds` | wall-clock time of the run |
| `registration_status` | SUCCESS / LOW_CONFIDENCE / FAILED, with reasons |

**Status rules** (all thresholds are in `backend/app/config.py`):

* **FAILED** – fewer than 50 keypoints in either image, fewer than 12 candidates, no
  consensus, fewer than 12 inliers, inlier ratio below 0.5 %, an invalid transformation,
  or coverage below 5 %. No registered image is produced.
* **LOW_CONFIDENCE** – accepted, but below a soft target: 30 inliers, 5 % inlier ratio,
  25 % coverage (within overlap), 2.5 px RMSE.
* **SUCCESS** – all targets met.

No accuracy percentage is reported: there are no independent control points in the
sample data, so no ground-truth accuracy can be computed. The XML/GCP-predicted
footprint is reported alongside each result as a plausibility check only.

## 9. Known limitations

The MVP uses classical SIFT correspondence. It struggles with – and on the official
data demonstrably fails on some of – the following:

* **strong Sun-angle differences and shadow reversal** (SIFT orientation locks onto the
  shading direction, so craters match other craters);
* **IIRS-to-LRO modality differences** (float32 radiance vs. WAC reflectance);
* **low-texture and deeply shadowed terrain** (no features at all – coverage stays low);
* **very small overlap** and very narrow strips;
* **perspective / parallax and large geometric distortion** – one global affine or
  homography cannot model terrain relief or projection scale change along a long strip;
* **no TMC-2 sample** is available, so TMC-2 is untested.

## 10. Future work

These limitations motivate, in order: learned detectors/matchers (SuperPoint + LightGlue,
LoFTR), multi-modal descriptors for IIRS ↔ WAC, illumination-aware matching, local /
piecewise transformation models, DEM-assisted geometry and orthorectification, TMC-2
validation once data is provided, and independent check points for true accuracy
assessment. None of these is implemented in this MVP.

## 11. Project layout

```
backend/app/
  config.py          all thresholds and parameters (Pydantic)
  ingestion/         dataset indexer, raster loading, valid masks
  metadata/          XML parser, metadata-predicted footprint
  preprocessing/     normalisation, CLAHE, gradient
  matching/          tiled SIFT / RootSIFT, KNN + ratio test
  geometry/          robust estimation, two-stage consensus, validation
  registration/      engine, warping, result schema, job storage
  evaluation/        reprojection, cross-validation, coverage, chance inliers
  visualization/     match, coverage, footprint and overlay rendering
  api/               FastAPI routes and background job manager
backend/tests/       pytest suite (synthetic data + official dataset when present)
frontend/            React + Vite + Tailwind interface
scripts/             inspect_dataset, run_registration, benchmark_ohrc, benchmark_iirs
docs/                engineering findings and benchmark results
results/             all generated output (benchmarks, jobs, caches)
```
