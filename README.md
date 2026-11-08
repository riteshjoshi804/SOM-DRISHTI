# SOM DRISHTI

**Lunar Image Correspondence & Registration System**
Smart India Hackathon 2026 · Problem statement **SIH26166** · ISRO / Department of Space

> Multi-modal, Sun-angle and scale invariant image correspondence using Chandrayaan-2
> optical images (OHRC, TMC and IIRS).

Three independent matchers (classical SIFT, SuperPoint + LightGlue, LoFTR) feed one verification
and one report, on the official sample dataset supplied by the problem-statement mentor. Every
number it reports is measured. When registration fails, it says so, and the classical pipeline
stays available as a fixed baseline.

---

## 1. Objective

Register a Chandrayaan-2 source image onto its LRO reference image without assuming
that the source lies entirely inside the reference, then report how well-supported
that registration is:

1. discover the official source / reference / XML triplets,
2. parse the XML metadata and load the rasters with their original values,
3. find correspondences with the chosen matcher, or compare all of them (AUTO),
4. estimate and validate a source → reference mapping with robust geometry,
5. select spatially balanced tie points and refine them to sub-pixel positions,
6. warp the source into the reference grid and render overlays,
7. measure geometric consistency, held-out error and, if supplied, independent control-point
   error, keeping the three apart, and assign SUCCESS / LOW_CONFIDENCE / FAILED with reasons,
8. expose all of it through a FastAPI backend and a React interface, with downloadable outputs.

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

```
source + reference + XML
   |
   |-- XML/GCP search prior: orientation, scale, rough position (lunar CRS checked, whole-image fallback)
   |
   |-- SIFT / RootSIFT ---------- KNN + Lowe ratio test ------------------+
   |-- SuperPoint + LightGlue --- LightGlue confidence + appearance check |-- one verification:
   |-- LoFTR (detector-free) ---- LoFTR confidence + appearance check ----+   MAGSAC/RANSAC, plausibility,
   |-- RIFT (experimental) ------ descriptor similarity + appearance -----+   decoy controls, strip model
   |
   |-- AUTO compares accepted results by coverage, held-out error, inlier ratio, RMSE, time (never match count)
   |
   |-- tie points: duplicates, confidence ranking, ANMS, per-cell cap (cells are never filled)
   |-- sub-pixel refinement: gradient NCC with ambiguity and validity checks; initial coordinates kept
   |-- validation: held-out split and unseen blocks; control points if supplied; synthetic suite
   |
   +-- registered GeoTIFF, previews, overlays, CSV tables, JSON, scientific report
```

### Classical branch (the fixed baseline)

SIFT per 768 px tile inside an eroded valid mask, RootSIFT, brute-force 2-NN, Lowe ratio 0.90, one-to-one
filter, affine seed plus expansion (MAGSAC, 3 px), a mirrored retry, and pooling across representations.
Its results are checked against the stored baseline with `scripts/check_sift_regression.py`.

### Learned branches

Learned matchers are not rotation invariant and do not expect one image to be a small part of the other,
and the official scenes are rotated up to 118° and sit kilometres from their metadata position. So:

1. **Search prior.** The XML/GCP footprint gives orientation, scale and rough position, plus a 180° alternative
   because one scene is reversed relative to its GCPs. The CRS is read from the reference's own WKT; a non-lunar
   body, no CRS or an implausible scale disables the prior, and the whole reference is searched in every orientation.
2. **Anchor sweep.** A source tile, rotated into the reference frame, slides over the search region at reduced
   resolution. A translation vote finds the right window, and another independent tile must confirm it.
3. **Tile passes.** All tiles are matched at full resolution, in three passes that re-centre on the improving model.
   Window positions are jittered so that position bias cannot look like agreement. Tile coordinates map back
   to original pixels exactly, including LoFTR's half-pixel convention.
4. **Appearance check.** Gradient-magnitude patches at the two points must correlate. This is independent of the
   matcher and removes look-alike craters and matches inside black shadow before any geometry is estimated.
5. **Strip model.** A smooth along-track model is adopted only if it predicts stretches left out of its fit better
   than a global transformation; a very stiff one is exactly an affine.
6. **Decoy controls.** The final pass is repeated on known-wrong ground, and the result must exceed that by a factor of 3.

The reasoning and measurements behind each step are in [docs/engineering_findings.md](docs/engineering_findings.md).

## 4. Technology stack

Python 3.11+ (developed on 3.13) · FastAPI · Uvicorn · Pydantic · OpenCV (SIFT, RANSAC, USAC_MAGSAC) · NumPy ·
SciPy · Rasterio (bundled GDAL) · PyProj · React 19 · Vite 7 · Tailwind CSS 4. No API keys and no external services.

Learned matchers (optional): PyTorch · Kornia (LightGlue, LoFTR) · Hugging Face Transformers (SuperPoint). They use a
CUDA GPU if there is one and otherwise the CPU; only the CPU path has actually been run so far.

| Weights | Source | Licence |
|---|---|---|
| SuperPoint | Hugging Face `magic-leap-community/superpoint` (Magic Leap) | **Academic or non-profit noncommercial research use only** |
| LightGlue for SuperPoint | Hugging Face `kornia/lightglue`, falling back to the `cvg/LightGlue` release | Apache-2.0 |
| LoFTR outdoor | Hugging Face `kornia/loftr`, falling back to the authors' mirror | Apache-2.0 |

Weights are not in Git. They download once to `models/` on first use.

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

Learned matchers (optional; everything else runs without them):

```powershell
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r backend\requirements-learned.txt
```

For a CUDA GPU, take the matching PyTorch build from https://pytorch.org/get-started/locally/ instead.

Optional: copy `.env.example` to `.env` to change the data, results or models folder.

## 6. Running

All commands are run from the project root.

**Inspect the dataset** (dimensions, dtypes, NoData, CRS, geotransform, GCPs, XML fields,
metadata-predicted footprint):

```powershell
.venv\Scripts\python.exe scripts\inspect_dataset.py --json results\dataset_inspection.json
```

**Register one scene** (default: OHRC 2359602 with SIFT; any unique part of a scene ID works):

```powershell
.venv\Scripts\python.exe scripts\run_registration.py
.venv\Scripts\python.exe scripts\run_registration.py --scene 2470502 --matcher LOFTR
.venv\Scripts\python.exe scripts\run_registration.py --scene 2470502 --matcher AUTO
.venv\Scripts\python.exe scripts\run_registration.py --scene 2736702 --modes clahe --ratio 0.85 --model homography
```

Artifacts go to `results/jobs/<scene_id>__<timestamp>/`:

| Group | Files |
|---|---|
| Images | `source.jpg`, `reference.jpg`, `reference_footprint.jpg`, `raw_matches.jpg`, `inlier_matches.jpg`, `spatial_coverage.jpg`, `tie_points.jpg`, `keypoints_*.jpg` (SIFT) |
| Accepted result only | `registered_source.tif` (original values, reference grid), `registered_source.png`, `overlay.png`, `checkerboard.jpg`, `reference_crop.jpg` + `registered_crop.png` (alpha overlay) |
| Tables | `correspondences_raw.csv`, `correspondences_inliers.csv`, `tie_points_selected.csv`, `tie_points_refined.csv` |
| Documents | `transformation.json`, `metadata_summary.json`, `metrics.json`, `benchmark_row.csv`, `report.md`, `report.html`, `result.json` |
| Other matchers (AUTO) | `<matcher>__overlay.png`, `<matcher>__inlier_matches.jpg`, `<matcher>__*.csv` |

**Benchmarks** (each scene also becomes a normal job, visible in the web interface):

```powershell
.venv\Scripts\python.exe scripts\benchmark_matchers.py          # -> results\matcher_benchmark.csv/.json, results\benchmark_charts\
.venv\Scripts\python.exe scripts\benchmark_representations.py   # -> results\representation_benchmark.json
.venv\Scripts\python.exe scripts\synthetic_validation.py        # -> results\synthetic_validation.csv/.json (not lunar data)
.venv\Scripts\python.exe scripts\refinement_study.py            # -> results\refinement_study.json (not lunar data)
.venv\Scripts\python.exe scripts\write_benchmark_report.py      # -> docs\benchmark_matchers.md

# The classical baseline, and a check that it has not changed
.venv\Scripts\python.exe scripts\benchmark_ohrc.py --out results\regression\ohrc_benchmark.csv --no-geotiff
.venv\Scripts\python.exe scripts\benchmark_iirs.py --out results\regression\iirs_benchmark.csv --no-geotiff
.venv\Scripts\python.exe scripts\check_sift_regression.py
```

A complete matcher benchmark takes hours on a CPU. It saves after every scene.

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
| POST | `/api/register/{scene_id}` | start a registration job (optional body `{"config": {...}}` with partial overrides, e.g. `{"config": {"matcher": "AUTO"}}`). A matcher that cannot run gives 409, never a silent substitute |
| POST | `/api/register/upload` | register an uploaded source/reference (+ optional XML); stored under `results/uploads/` |
| GET | `/api/results` | all jobs, including those produced by the scripts |
| GET | `/api/results/{job_id}` | job state/progress, and the full result when done |
| GET | `/api/results/{job_id}/metrics` | status, reasons and metrics only |
| GET | `/api/results/{job_id}/artifacts/{name}` | an artifact image / GeoTIFF / result.json |
| GET | `/api/results/{job_id}/download` | every deliverable of a job as one ZIP |
| GET | `/api/results/{job_id}/chip` | a magnified piece of an original raster, for picking control points |
| POST | `/api/results/{job_id}/control-points` | evaluate a finished result against independent points; never used in a fit |
| GET | `/api/matchers` | what can really run here: packages, weights, device, licences |
| GET | `/api/benchmarks` | stored benchmark results; empty until a script has run |
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
