# SOM DRISHTI – Development Progress

Advanced feature upgrade of the classical MVP (SIH26166).
Last updated: 2026-11-01, after the stop at 02:57:21. **State: PAUSED at the user's request. No development process is running and nothing has been committed.**

Status words used here: NOT STARTED, IN PROGRESS, IMPLEMENTED, TESTING, VERIFIED, BLOCKED.
VERIFIED means the complete workflow passed its acceptance tests. Nothing below is claimed beyond what was actually run.

## 1. Current state: PAUSED

Development was paused at the user's request on 2026-11-01. Phases 1, 2 and 3 of the resume brief are finished. Phase 4 is partly done and Phase 5 is not finished.

* **Corrected official benchmark: 6 of 6 scenes completed** for SIFT, SuperPoint + LightGlue and LoFTR (section 3d). 2026-11-01, 01:00:03 to 02:36:00, Python exit code 0, log `runs/benchmark_matchers_20261101_final.log`. The first complete run (2026-10-31, 23:09:47 to 00:46:43) showed that the comparison was not under one protocol (section 3c); it is kept in `runs/benchmark_run1_20261031/`.
* **RIFT: 5 of 6 scenes completed, the last one interrupted intentionally** (section 3d). The pause request described 3 of 6, with 2736702 in progress; RIFT had moved on by the time it was stopped, so 2736702 and 2686502 are also complete, with saved results.

**Exact stopping point.** The RIFT step (`.venv\Scripts\python.exe scripts\benchmark_matchers.py --rift-only`, started 02:36:57, worker PID 27636, launcher PID 8760, log `runs/benchmark_rift_20261101.log`) was stopped at **02:57:21**, during IIRS scene **1519402**. Its last log line, at 02:57:20, is `[ 47%] RIFT: pass 2: tile 15/19: 2 matches, 2 agree`. The exit code of −1 is that deliberate stop.

Ctrl+C was not used, because it reaches every process sharing the console, which could include Claude Code. A polite request cannot end a windowless process, so only the worker was ended directly, after its command line was confirmed, and the launcher then exited by itself. Nothing measured was lost, because results are written only at the end of a scene: the CSV and JSON stayed at their 02:55:59 write, and the unfinished scene has no row and no job folder.

**Next task:** section 9.

## 2. Where things stand

* **Git:** branch `feature/advanced-matching`. The baseline is commit `915e28e` (also on `main`). All upgrade work is **uncommitted** in the working tree, because no commit has been asked for: 68 files, 23 modified and 45 untracked, none staged and none deleted (section 11). Nothing was stashed, reset or discarded. Nothing under `data/` or `models/` shows as changed.
* **Latest backup:** `results/checkpoints/pause_20261101-025721/`, with the source archive, benchmark outputs, logs, probe scripts, job summaries, the Git state and a checksum manifest. The earlier archives `wip_feature-advanced-matching_20261031_phase3.tar.gz`, `..._20261031.tar.gz` and `..._20261030.tar.gz` are kept.
* **Latest tests:** the last complete suite gave **224 passed in 194 s** (`results/dev_probes/runs/full_suite_20261101_support_rule.log`, finished 00:56:36). After it, a strip-support column was added, and `test_engine_branches.py` and `test_benchmark_script.py` passed again (42 passed). **Later edits have not been run:** the `notes` field of `/api/benchmarks` in `routes.py` with the matching edit to its test, the notes section of `write_benchmark_report.py`, and the notes and strip-support column of `Benchmark.jsx`.
* **Frontend:** `frontend/dist` was built on 2026-10-31 at 01:36, and `Methodology.jsx`, `Benchmark.jsx` and others were edited after that, so **the build is stale** and a rebuild is due.
* **Official dataset:** 18 of 18 files matched their SHA-256 baseline (`results/dataset_sha256_baseline.txt`) when checked at the resume on 2026-10-31. The hashes were not re-checked at this pause, as no new checks were to be started. To be checked at the next resume.
* **Official benchmark:** finished for SIFT, SP+LG and LoFTR; RIFT 5 of 6 (section 3d). Figures in sections 3, 3b and 6 come from development drivers or single application runs, and section 3d replaces them where they differ. The generated report `docs/benchmark_matchers.md` does not exist yet.

## 3. Scene 2436502: registered automatically, with partial overlap

**The automatic search now finds this scene on its own reference, with no location supplied.** No acceptance limit was changed, and nothing in the code refers to this scene.

### Configuration change

In `GuidedSearchConfig`, `backend/app/config.py`, the only place these are set:

| Setting | Before | Now | Why |
|---|---:|---:|---|
| `anchor_max_tiles` | 4 | **8** | It sets both the sweep rounds and the along-track segments of `_spread_first`. With 4, the tile that finds the location (source row 1583) and the one that confirms it (row 321) were never tried, because a better-textured tile in the same stretch took the place. With 8, every tile position takes part. |
| `anchor_max_pairs` | 900 | **1700** | About 100 image pairs per tile and orientation on this scene: 8 × 2 × 100 = 1600, plus confirmations. |

Cost: a scene that never confirms now spends up to 1700 pairs instead of 900. `anchor_min_peak` (6), `anchor_min_confirmations` (1), `anchor_confirm_min_ground` (0.5), `appearance_min_zncc` (0.25), `decoy_factor` (3), `learned_min_inliers` (40) and the geometry limits are unchanged. New test: `test_every_tile_position_of_a_strip_takes_part_in_the_sweep`.

### Commands and runs

| Command | Log or job | Took | Result |
|---|---|---:|---|
| `.venv\Scripts\python.exe results\dev_probes\dev_guided.py 2436502 LOFTR 2` | `runs/guided_2436502_LOFTR_v3.log` | 17.6 min | Location found, 540 of 806, decoys 16 and 11 |
| `.venv\Scripts\python.exe scripts\run_registration.py --scene 2436502 --matcher LOFTR` | job `...__20261031-212841-886`, `runs/cli_2436502_LOFTR.log` | 16.4 min | **SUCCESS**, same 540 of 806 |
| `.venv\Scripts\python.exe results\dev_probes\dev_2436502_crosscheck.py` | `runs/crosscheck_2436502.log` | about 4 min | Agreement through another raster, below |

Earlier runs, kept as they were: `runs/guided_2436502_LOFTR.log` (860 pairs, no location) and `runs/guided_2436502_LOFTR_v2.log` (802 pairs, no location).

### What the search did

1. The correlation search found no agreement (13 s), as calibrated for this scene, and handed over to the matcher sweep.
2. Sweep order: rows 2214, 3475, 2844, then 1583. Wrong windows reached 4 to 7 votes; two candidates at wrong offsets (6 and 7 votes) were rejected by their confirmation tiles.
3. Row 1583 gave a candidate with 7 votes (second peak 2) at offset (−1417, 932). Row 321 confirmed it with 19 votes at (−1406, 944).
4. Anchor found after 701 of 1700 image pairs, 668 to 729 s. It is 8.48 km from the metadata prediction.

### Evaluation

| Question | Answer | Evidence |
|---|---|---|
| Correct location? | **Yes, by an independent check** | A model fitted only on the reference of scene 2470502, started from a correlation peak on that raster, predicts the 540 tie points to a median of **1.78 px**, p90 3.27, max 4.22 (5 m pixels) in map coordinates. Both rasters are LROC products, so this is consistency, not ground truth. |
| Genuine correspondences? | Yes | 540 inliers of 806 accepted (67.0 %), 2 821 proposed. Decoys on known-wrong ground: 16 and 11 inliers, so the result is 33 times the null level (3 required). Median patch similarity of the kept matches +0.40. |
| Spatially meaningful? | Yes, within the overlap | 66.7 % of the 8×8 cells inside the overlap are occupied (31.2 % over the whole image). Inliers span source rows 101 to 2351. 98 tie points selected, distribution score 0.857. |
| Plausible geometry? | Yes | Strip model valid. Local scale 0.96 to 1.00, anisotropy 1.04, shear 1.0°, no mirror. Rotation 0.46° from the metadata, scale ratio 0.971. 38.9 % of the valid source lies on the reference, against 38.4 % estimated independently before this run. |
| Held-out error? | Reported, not ground truth | Held-out RMSE / median / p90 2.01 / 1.03 / 2.49 px. Unseen along-track blocks 3.35 px (target 5). Reprojection RMSE 1.29 px. |
| Reproducible? | **Yes** | The application run reproduces the probe point by point: 540 tie points, largest coordinate difference 0.00005 px (CSV rounding), same anchor. Same seed, same CPU. |
| By eye? | Consistent | `runs/inspect_20261031-212841.jpg`: craters line up in source, reference and checkerboard at 0.04, 0.12, 0.22, 0.30 and 0.42 along the strip, despite a 60° difference in shading direction. |
| Refinement? | Improves agreement; no accuracy claim | 355 of 540 points refined, median shift 0.80 px, no coherent shift. Same held-out splits: median 0.95 → 0.84 px, RMSE 1.23 → 1.16 px. Local check: median 0.87 → 0.63 px. These are automatic matches, not truth, and the synthetic study warns about strong Sun change, so **no sub-pixel accuracy is claimed**. |

**Limit of this result:** 61 % of the source has no ground on this reference. Nothing is known about that part from this registration, and the reported SUCCESS applies to the overlap.

## 3b. SuperPoint + LightGlue: works on official imagery, on one IIRS and one OHRC scene

Run on CPU (torch 2.15.0+cpu, 8 threads), with the current limits. No standby during these runs. This is search and verification through `guided_search`; the application pipeline, with held-out errors and refinement, runs in the benchmark. So the status stays TESTING, not VERIFIED.

### The matcher itself, at a known location

`.venv\Scripts\python.exe results\dev_probes\dev_splg_check.py`, log `runs/splg_check_20261031.log`, 62 s. Tiles were placed using the LoFTR model of the same scene, which shares no features or search with this matcher, so this tests the matcher with the search out of the way.

| Check | Result |
|---|---|
| Weights | Present on disk: SuperPoint 5.2 MB (`models/huggingface/...`), LightGlue 47.5 MB (`models/torch/checkpoints/...`). Loaded in 4.4 s, and a second construction reuses the same objects. |
| Device handling | `auto` resolves to CPU. Asking for `cuda` is refused with a clear error, not silently run on CPU. The GPU path itself is **not tested**, because this machine has no CUDA GPU. |
| Coordinates | Keypoints inside the tile, on valid pixels, on integer pixels: all passed in both scenes. |
| Confidence | In [0, 1], and nothing below the configured threshold kept: all passed in both scenes. |

| Scene | Tile pairs | Candidates | Accepted | Near the LoFTR model | Expected by chance | After appearance check, near / kept | Confidence min / median / max | s per pair |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| IIRS 2686502 | 11 | 877 | 398 | **331** | 1 | 142 / 157 | 0.200 / 0.637 / 0.999 | 2.7 |
| OHRC 2359602 | 8 | 804 | 638 | **609** | under 1 | 396 / 406 | 0.204 / 0.977 / 1.000 | 2.8 |

### The automatic search, geometric verification and decoys

`.venv\Scripts\python.exe results\dev_probes\dev_guided.py <scene> SUPERPOINT_LIGHTGLUE 2`, logs `runs/guided_2686502_SPLG.log` and `runs/guided_2359602_SPLG.log`. Both Python exit codes were 0.

| Scene | Location found by | Accepted | Inliers | Ratio | Model | RMSE / median px | Coverage (8×8, image) | Decoy inliers | Search + decoys |
|---|---|---:|---:|---:|---|---:|---:|---|---:|
| IIRS 2686502 | Gradient correlation, 18 of 20 tiles, mean z 11.3, 11.7 km off the metadata | 210 | **114** | 54.3 % | strip | 1.83 / 1.72 | 59.4 % | 0, 0 | 133 + 73 s |
| OHRC 2359602 | Gradient correlation, 3 of 13 tiles, mean z 8.4, rotated 180°, 2.3 km off the metadata, mirrored | 553 | **301** | 54.4 % | strip | 1.88 / 1.77 | 45.3 % | 0, 0 | 130 + 38 s |

* **Independent agreement:** the best affine agrees closely with LoFTR's, which was found by a different matcher. 2686502: linear terms within 0.005, offset within 3.3 px. 2359602: linear terms within 0.002, offset within 1.1 px. Both matchers also find the 180° rotation with mirroring on 2359602, and local anisotropy of about 1.54 on 2686502.
* **What this does not show:** the residuals are on automatic matches, not accuracy. Two scenes are not all scenes; 2436502, 2470502 and 2736702 are the harder cases and come with the benchmark.
* **Watch, on 2359602:** pass 3 fell to 198 inliers. At one tile near the end of the strip, 9 of 14 matches agree on a shift of about (−47, −52) px, far beyond the 7 px by which the LoFTR strip departs from its affine, and the flexible spline bent towards them (deviation up to 51 px). The cluster is probably false, though that has not been confirmed. The run kept pass 2 (301 inliers), which was not affected, but this shows that a small cluster can tempt the strip model.

## 3c. First benchmark run: what inspecting it found

18 rows, 6 scenes, 96.9 min on CPU, exit code 0, no standby, no Windows crash or hang event, no traceback. This run is superseded by the final one (section 3d) because of the first finding below; its figures stay in `runs/benchmark_run1_20261031/`.

### Corrected: the comparison was not under one protocol

Measured from the saved inliers with `results/dev_probes/dev_support_preview.py`, which reproduces every warning the engine gave.

* **The along-track support rule applied only to learned matchers with a strip model.** SIFT was never held to it, not even under AUTO. On 2359602 SIFT's 229 inliers span 30 % of the strip and it reported SUCCESS, while LoFTR's 7 249 span 40 % and it reported LOW_CONFIDENCE.
* **A single tie point could count as support.** SP+LG reached exactly 50 % on 2359602, and so SUCCESS, only through three stretches holding one point each. With any one of them gone it would be 45 %, and LoFTR, with a better held-out median (1.56 against 2.43 px), would have been AUTO's choice. The registration itself barely changes: the two models differ by a median of 0.42 px and at most 1.49 px at SP+LG's inliers.

**Change** (`registration/engine.py`, `config.py`, both stricter, no limit lowered): a twentieth of the strip counts as supported only with at least 3 tie points (`along_track_support_min_points`), the fewest that can constrain an affine locally; and wherever results are compared, every branch faces the rule, whatever its model family. The SIFT-only run stays the unchanged baseline, so its stored SUCCESS on 2359602 is not comparable with the SIFT row under AUTO. Support is now a column in the benchmark.

| Scene | SIFT support | SP+LG support | LoFTR support | Effect, predicted from the saved inliers |
|---|---:|---:|---:|---|
| 2359602 | 30 % | 35 % | 40 % | All three LOW_CONFIDENCE; AUTO should move from SP+LG to LoFTR |
| 2436502 | – | – | 50 % | LoFTR stays SUCCESS, **exactly at the limit**, with one stretch holding exactly 3 points. A requirement of 5 points would make it LOW_CONFIDENCE |
| 2470502 | 5 % | 45 % | 35 % | LoFTR gains the warning; no status changes |
| 2736702 | 20 % | 35 % | 40 % | No status changes |
| 2686502 | – | 60 % | 90 % | No status changes |
| 1519402 | 10 % | 85 % | 100 % | No status changes |

### Closed

* **LoFTR on 2359602 fell from 13 041 to 7 249 inliers, as expected.** The earlier development run used 22 tile pairs from two near-duplicate columns (18 089 accepted), this one 11 pairs (8 802 accepted). Coverage went from 50 % to 45.3 %.
* **Anisotropy on 2686502 is not a fitting artefact** (`runs/anisotropy_check.log`, `runs/anisotropy_latitude.log`). The metadata prior, with no matcher involved, already has anisotropy 1.418 and the fit 1.395. Along the strip the cross-track scale varies from 1.078 to 1.293 (spread 1.199), but scale × cos(latitude) has a spread of only 1.041, and the along-track scale is constant at 0.850. So the variation along the strip follows the geographic reference grid. A constant anisotropy of about 1.16 remains, which the metadata footprint also predicts; its physical cause has not been established.
* **The visual check of all six selected results** (`runs/inspect_*__2026*.jpg`): craters and shadow edges line up wherever tie points exist. Stretches with no tie points are mostly dark or ambiguous, so the images neither confirm nor refute alignment there.
* **Provisional limit `learned_min_inliers = 40` is kept.** The highest decoy result in the run is 16 inliers and the weakest accepted learned result 96. It is still calibrated on these six scenes only.

### RIFT trial

`.venv\Scripts\python.exe results\dev_probes\dev_guided.py 1519402 RIFT 2`, log `runs/guided_1519402_RIFT.log`, 2 min 47 s, limited to 25 min. It runs on official imagery: 66 of 122 accepted matches are inliers (54.1 %), affine model, RMSE 0.92 px, median 0.61 px, decoys 0 and 0, but coverage is only 9.4 %. The Python exit code was not captured by that launch; stderr is empty and the log is complete.

## 3d. Final benchmark: SIFT, SP+LG and LoFTR on all six official scenes

`.venv\Scripts\python.exe scripts\benchmark_matchers.py`, 2026-11-01, 01:00:03 to 02:36:00 (95.9 min), Python exit code 0, on CPU (8 threads), with torch 2.15.0+cpu, Kornia 0.9.0, transformers 5.22.1, OpenCV 5.0.0. No traceback, no crash or hang event, no standby. Outputs: `results/matcher_benchmark.csv`, `.json`, `results/benchmark_charts/`, and one job per scene under `results/jobs/`.

**Errors are measured on automatic matches, not ground truth.** `*` marks AUTO's choice. Support is the share of 20 stretches along the strip holding at least 3 inliers. Coverage is within the overlap.

| Scene | Sensor | Matcher | Status | Inliers | Ratio | RMSE px | Held-out median px | Unseen blocks px | Coverage | Support | Decoy max | Time s |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2359602 | OHRC | SIFT | LOW_CONFIDENCE | 229 | 28.6 % | 1.54 | 1.61 | 2.39 | 29.7 % | 30 % | 0 | 15 |
| | | SP+LG | LOW_CONFIDENCE | 301 | 54.4 % | 1.88 | 2.43 | 2.13 | 45.3 % | 35 % | 0 | 194 |
| | | LoFTR * | LOW_CONFIDENCE | 7 249 | 82.4 % | 1.59 | 1.56 | 2.16 | 45.3 % | 40 % | 0 | 289 |
| 2436502 | OHRC | SIFT | FAILED | 0 | 0 % | – | – | – | – | – | – | 37 |
| | | SP+LG | FAILED | 0 | – | – | – | – | – | – | – | 850 |
| | | LoFTR * | SUCCESS | 540 | 67.0 % | 1.29 | 1.03 | 3.35 | 66.7 % | 50 % | 16 | 1 071 |
| 2470502 | OHRC | SIFT | LOW_CONFIDENCE | 19 | 0.98 % | 0.51 | 0.46 | – | 6.2 % | 5 % | 0 | 33 |
| | | SP+LG * | LOW_CONFIDENCE | 101 | 43.0 % | 1.84 | 3.03 | 7.04 | 45.3 % | 45 % | 5 | 279 |
| | | LoFTR | LOW_CONFIDENCE | 732 | 67.2 % | 1.64 | 1.82 | 9.16 | 26.6 % | 35 % | 6 | 396 |
| 2736702 | OHRC | SIFT | LOW_CONFIDENCE | 32 | 1.73 % | 1.45 | 1.41 | 1.96 | 14.1 % | 20 % | 0 | 22 |
| | | SP+LG | LOW_CONFIDENCE | 96 | 42.9 % | 1.82 | 3.10 | 5.24 | 31.3 % | 35 % | 0 | 182 |
| | | LoFTR * | LOW_CONFIDENCE | 720 | 55.9 % | 1.67 | 2.19 | 2.67 | 29.7 % | 40 % | 6 | 362 |
| 2686502 | IIRS | SIFT | FAILED | 0 | 0 % | – | – | – | – | – | – | 33 |
| | | SP+LG | SUCCESS | 114 | 54.3 % | 1.83 | 2.37 | 4.36 | 59.4 % | 60 % | 0 | 240 |
| | | LoFTR * | SUCCESS | 4 984 | 84.7 % | 1.40 | 1.22 | 2.02 | 90.6 % | 90 % | 10 | 434 |
| 1519402 | IIRS | SIFT | LOW_CONFIDENCE | 43 | 3.36 % | 1.13 | 0.36 | 10.42 | 17.2 % | 10 % | 0 | 13 |
| | | SP+LG | SUCCESS | 347 | 58.1 % | 1.60 | 1.59 | 3.10 | 87.5 % | 85 % | 0 | 452 |
| | | LoFTR * | SUCCESS | 12 763 | 87.1 % | 1.08 | 0.81 | 2.91 | 100 % | 100 % | 16 | 740 |

* **Reproducible:** every measured value of all 18 rows is identical at full precision to the first complete run (`results/dev_probes/dev_compare_benchmarks.py`, `runs/benchmark_compare_run1_final.log`). Same CPU and seed; this says nothing about another machine.
* **The corrected rule did what the saved inliers predicted:** only 2359602 changed. All three of its matchers are now LOW_CONFIDENCE and AUTO chose LoFTR. The IIRS scenes and 2436502 stay SUCCESS.
* **What the status means here:** three scenes reach SUCCESS. The three LOW_CONFIDENCE OHRC scenes register visibly well where they have tie points, but less than half of each strip has enough of them. 2436502 sits exactly at the support limit, and its SUCCESS applies to the 38.9 % of the source with ground on its reference.
* **Learned matchers versus SIFT, on these six scenes:** LoFTR is accepted on all six, SP+LG on five (no location on 2436502), SIFT on four, three of those with under 45 inliers. AUTO does not rank by match count.
* **Timing:** see the power entries in section 7. The LoFTR branch of 2436502 ran mostly on battery.
* **Kept copies:** the outputs before RIFT are in `runs/benchmark_final_no_rift_20261101/`. RIFT left those 18 rows identical.

### RIFT (experimental): 5 of 6 scenes, the last one interrupted intentionally

`.venv\Scripts\python.exe scripts\benchmark_matchers.py --rift-only`, same protocol and machine, log `runs/benchmark_rift_20261101.log`, stderr empty, no power event. The implementation is in-house and experimental, runs on about one CPU core, and AUTO never selects it. Its rows are in the same CSV and JSON.

| Scene | Sensor | Status | Accepted | Inliers | Ratio | Model | RMSE px | Held-out median px | Unseen blocks px | Coverage | Support | Decoy max | Time s | What happened |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2359602 | OHRC | LOW_CONFIDENCE | 750 | 526 | 70.1 % | affine | 1.76 | 1.84 | 2.63 | 31.3 % | 30 % | 0 | 91 | Location from gradient correlation |
| 2436502 | OHRC | FAILED | 0 | 0 | – | – | – | – | – | – | – | – | 372 | No location after 1 504 image pairs |
| 2470502 | OHRC | FAILED | 0 | 0 | – | – | – | – | – | – | – | – | 246 | No location after 1 434 image pairs |
| 2736702 | OHRC | FAILED | 0 | 0 | – | – | – | – | – | – | – | – | 294 | Correlation found a location, RIFT reached 21, 16 and 19 inliers there (40 required), and the sweep found none |
| 2686502 | IIRS | FAILED | 0 | 0 | – | – | – | – | – | – | – | – | 119 | Correlation found the location, RIFT reached 0 inliers there, and the sweep found none |
| 1519402 | IIRS | **NOT COMPLETED** | – | – | – | – | – | – | – | – | – | – | – | Stopped at pass 2, tile 15 of 19. **No result exists.** The development trial in section 3c is a separate run, not a benchmark row |

Whether RIFT's result on 2359602 agrees with the other matchers has not been measured, because `--rift-only` does not run AUTO's cross-check.

## 4. Status by feature

**Only Phase 0 is VERIFIED.** The official benchmark has now run, but the acceptance tests of every other feature also include the final gate: complete tests after the latest edits, the frontend workflows in a browser, and the remaining studies.

| # | Feature (master prompt section) | Status | What is missing |
|---|---|---|---|
| – | Phase 0: audit, Git checkpoint, feature report | **VERIFIED** | – |
| – | Existing SIFT pipeline preserved | TESTING | Its six benchmark rows repeat the stored baseline figures, but `check_sift_regression.py` has not been re-run since `engine.py` was edited |
| 1 | SuperPoint + LightGlue | TESTING | Benchmark through the application: accepted on 5 of 6, FAILED on 2436502 (no location). Final gate; GPU path untested |
| 2 | LoFTR, independent branch | TESTING | Benchmark through the application: accepted on 6 of 6. Final gate; GPU path untested |
| 3 | Unified matcher interface and AUTO | TESTING | AUTO ran on all six scenes. Not yet used from the frontend |
| 4 | XML-guided footprint and overlap localisation | TESTING | Used on all six scenes in the benchmark. Final gate |
| 4b | Location search without a matcher (added) | TESTING | In the benchmark it supplied a location on several scenes and handed over to the sweep on the others. Its false-location rate rests on the calibration in section 5 |
| 5 | Illumination-aware representations | IMPLEMENTED | `benchmark_representations.py` not run |
| 5b | Optional RIFT | IN PROGRESS | Experimental. 5 of 6 scenes, 1 accepted (LOW_CONFIDENCE), 4 FAILED, **1519402 not completed**. Report not generated |
| 6 | ANMS and spatially balanced tie points | TESTING | Generated for every accepted benchmark result. Final gate |
| 7 | Robust geometric verification | TESTING | `learned_min_inliers = 40` kept after the benchmark. Final gate |
| 7b | Decoy null control (added) | TESTING | Measured on every accepted benchmark result, highest 16. Final gate |
| 7c | Along-track strip model (added) | TESTING | Adopted by SP+LG and LoFTR on all their accepted results. Final gate |
| 8 | Sub-pixel refinement | TESTING | Held-out agreement improved for LoFTR on four of six scenes and for SP+LG on all five; not for LoFTR on 2470502 and 2736702. Agreement between automatic matches only, so **no sub-pixel claim**. Synthetic validation not run in full |
| 9 | Held-out, control-point and synthetic validation | TESTING | `synthetic_validation.py` not run in full; picker not clicked in a browser |
| 10 | Matcher benchmarking | TESTING | Complete for SIFT, SP+LG and LoFTR, with CSV, JSON and charts. RIFT 5 of 6. **Report not generated, charts not inspected.** A run stopped by an exception is recorded as FAILED with no figures, and the next scene goes on |
| 11 | Scientific confidence report | TESTING | Generated for every benchmark job and in tests; not all read |
| 12 | Registration-ready exports | TESTING | Generated for every benchmark job and in tests; downloads to re-check |
| 13 | Frontend upgrade | TESTING | All pages rendered earlier; **build stale**; no click-through; real benchmark data not yet seen on the page |
| – | Requirements files, README, docs | IN PROGRESS | README sections 7 to 11, `benchmark_results.md`, `engineering_findings.md` |

## 5. Completed work, with its evidence

### False-match rejection tests: both passed

Run with the eight-tile sweep, as part of the complete suite (`backend/tests/test_engine_branches.py`, 216 passed), and again in the later suite with the corrected support rule (224 passed). Times below are from the first of those. Both use LoFTR on rendered terrain, with a source and a reference of different terrain and a prior claiming that they overlap.

| Test | What it checks | Result | Time |
|---|---|---|---:|
| `test_unrelated_imagery_is_rejected_by_the_full_pipeline` | The whole pipeline on unrelated imagery ends FAILED, with no transformation, no selected matcher, and reasons given | **Passed** | 55 s |
| `test_a_fooled_anchor_sweep_does_not_become_a_registration` | Even when the location search is forced to accept a false location and the tile passes run, the later gates still end FAILED with no transformation | **Passed** | 14 s |

### Known-bad pairs through the full pipeline: 4 of 4 FAILED, with the longer sweep

A longer sweep gives false locations more chances, so this was measured again at the current settings (appearance threshold 0.25, 8 tiles, 1700 pairs). `.venv\Scripts\python.exe results\dev_probes\dev_unrelated.py AB`, log `runs/unrelated_v2_eight_tiles.log`, 21:53:23 to 22:57:00. FAILED is the correct outcome for all of them.

| Pair | Status | Inliers | Image pairs swept | Highest peak in the logged lines |
|---|---|---:|---:|---:|
| Rendered baseline on a reference of different terrain | FAILED | 0 | 80 | 10 |
| Rendered Sun azimuth 180° on a reference of different terrain | FAILED | 0 | 90 | 8 |
| OHRC 2470502 on the reference of 2736702 | FAILED | 0 | 1421 | 5 |
| OHRC 2736702 on the reference of 2470502 | FAILED | 0 | 1418 | 7 |

The reason in all four is that no location was confirmed by two independent source tiles, and no tie point or transformation was produced. The peak column shows only the last six sweep lines the probe prints for each pair, so it is a lower bound. Peaks of 7 on real wrong ground and 10 on rendered terrain show that the vote count alone (6) does not protect against false locations; the confirmation by an independent tile does. The Python exit code was not captured (the recorded `0` is the PowerShell wrapper's). The last report of the script is in the log and Windows recorded no Python crash or hang. Elapsed time includes two periods of Modern Standby (section 7).

### Investigations closed

* **The 125-inlier decoy was my bug, not false matches.** Clipped windows put decoys back on true ground: reproduced at 124 inliers, all within 10 px of the real model. Strict windows give 7, none within 150 px.
* **What does get through on wrong ground:** crater look-alikes and black shadow patches (6 to 15 inliers). Neither the threshold (1 to 12 px) nor the model family is the cause. A gradient patch check independent of the matcher was added: real inliers median ZNCC +0.66, decoy inliers +0.07.
* **Anisotropy 3.09 was not physical.** Clamped spline knots forced a steep slope at the unsupported ends. With uniform knots it is 1.13 on the same scene.
* **SIFT regression:** 6 scenes, 0 differences against the stored baseline (`results/regression/comparison.log`). Run at 22:45, before the last `engine.py` edit at 23:30, so it must be repeated.

### Location search and partial overlap (added tonight, tests pass)

* `backend/app/matching/localisation.py`: masked FFT correlation of gradient magnitude, tried before any matcher. Calibrated on all six scenes and on 28 scans of pairs with no shared ground and of wrong orientations (`runs/locate_calibration.log`): true locations reach mean z 8.4 to 11.9, the others at most 7.1, limit 8.0. Scenes 2470502 (6.6) and 2436502 on its own reference (below that) fall through to the matcher sweep. A wrong location costs one pass, then the sweep runs.
* In `guided.py`: a tile only confirms a location where it has ground at that offset; sweep tiles are spread along the strip; tiles with no reference ground are skipped instead of pushed onto other terrain; near-duplicate tile columns are gone.
* New tests: `backend/tests/test_localisation.py`, 25 tests.

## 6. Earlier measurements

**These are development runs, kept for the record. Section 3d replaces them.** LoFTR rows come from `results/dev_probes/dev_guided.py` (search and verification only). All except 2436502 were made **before the search changes**. The reported errors are residuals on automatic matches, not accuracy. Logs: `results/dev_probes/runs/guided_*_LOFTR*.log`.

| Scene | Sensor | Orientation found | Off metadata by | Accepted | Inliers | Ratio | Model | RMSE / median px | Coverage (8×8, image) | Decoy inliers |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---|
| 2359602 | OHRC | Rotated 180° | 2.3 km | 18 089 | 13 041 | 72.1 % | strip | 1.61 / 1.37 | 50.0 % | 0, 0 |
| 2436502 (current code, partial overlap) | OHRC | Metadata | 8.5 km | 806 | 540 | 67.0 % | strip | 1.29 / 1.03 | 31.2 % (66.7 % within the overlap) | 16, 11 |
| 2470502 (threshold 0.30) | OHRC | Metadata | 5.2 km | 945 | 603 | 63.8 % | strip | 1.70 / 1.50 | 34.4 % | 0, 0 |
| 2736702 | OHRC | Metadata | 6.0 km | 1 362 | 803 | 59.0 % | strip | 1.64 / 1.41 | 32.8 % | 5, 0 |
| 1519402 | IIRS | Metadata | 20.0 km | 14 666 | 12 788 | 87.2 % | strip | 1.08 / 0.76 | 100 % | 17, 0 |
| 2686502 | IIRS | Metadata | 11.9 km | 5 408 | 4 576 | 84.6 % | strip | 1.39 / 1.08 | 93.8 % | 10, 0 |

* **2470502 through the application** (LoFTR, appearance threshold 0.30, job `...__20261030-231638-222`): held-out RMSE / median / p90 2.82 / 1.96 / 4.40 px; unseen along-track blocks 9.0 px; 78 tie points selected; distribution score 0.687; 513 points refined, median shift 0.54 px. Paired held-out: median 1.42 → 1.28 px, RMSE 1.72 → 1.75 px, so refinement is **not** shown to improve it. By eye: aligned at 0.60 and 0.78 along the strip, about 20 px off at 0.42.
* 2686502 has local anisotropy up to 1.54. This was checked against the metadata and the reference grid (section 3c): it is not a fitting artefact.

**SIFT baseline, stored run:**

| Scene | Status | Candidates | Inliers | Ratio | RMSE px | Coverage |
|---|---|---:|---:|---:|---:|---:|
| 2359602 | SUCCESS (mirrored) | 800 | 229 | 28.6 % | 1.54 | 29.7 % |
| 2436502 | FAILED | – | – | – | – | – |
| 2470502 | LOW_CONFIDENCE | 1 940 | 19 | 0.98 % | 0.51 | 6.2 % |
| 2736702 | LOW_CONFIDENCE | 1 850 | 32 | 1.73 % | 1.45 | 14.1 % |
| 2686502 (IIRS) | FAILED | – | – | – | – | – |
| 1519402 (IIRS) | LOW_CONFIDENCE | 1 278 | 43 | 3.36 % | 1.13 | 17.2 % |

**Sub-pixel refinement, SYNTHETIC rendered terrain with known truth, median px, initial → refined** (`scripts/refinement_study.py`): same Sun, plain Lucas–Kanade 0.24 → 0.016, gradient NCC 0.24 → 0.10. Noise, gradient NCC 0.37 → 0.17. Blur, 0.53 → 0.30. Sun lowered to 5°, 0.60 → 0.34. **Sun azimuth moved 90°: plain NCC 1.78 → 4.56 and ECC 1.40 → 4.82, which is worse**, while gradient NCC mostly refuses to refine. So refinement works where the appearance is comparable and must not be trusted under strong Sun change. No sub-pixel accuracy is claimed on real scenes.

**Provisional limits.** `learned_min_inliers = 40` sits between the highest null in the benchmark (16; 17 in an earlier development run) and the weakest accepted learned result (96), and rejects the 29-inlier look-alike seen on small synthetic scenes. It was kept after the benchmark. It is still calibrated on these six scenes only.

## 7. Known bugs, limitations and open investigations

**Known bug, not fixed:**

* **A misleading failure reason.** On RIFT's 2686502 and 2736702 rows, the reason says that no location was confirmed. In fact gradient correlation found a location, the matcher got too few inliers there, and only then did the sweep find nothing. The saved job has the true story in `search_trace.abandoned_anchor` and `abandoned_passes`, but the row and the report do not say it. This could affect any learned matcher. `results/dev_probes/dev_rift_trace.py` prints it.

**Open investigations:**

* The cluster that tempted the strip model on 2359602 (pass 3 of SP+LG, section 3b) is probably false, but that has not been confirmed. The final result is not affected.
* The cause of the constant anisotropy of about 1.16 on 2686502 is not established. The metadata predicts it (section 3c).
* Whether RIFT's result on 2359602 agrees with the other matchers has not been measured.
* The benchmark charts and the Benchmark page with real data have not been looked at.
* Windows logged no standby during the final benchmark, with `keep_awake` active. That does not prove `keep_awake` was the reason.

**Limitations:**

* 2436502 registers only on the 38.9 % of its valid source that lies on its reference; nothing is known about the rest from this result (section 3). Its SUCCESS is **borderline**: strip support is exactly 50 %, the limit, and one stretch holds exactly 3 points. Requiring 5 points would make it LOW_CONFIDENCE.
* SP+LG and RIFT find no location on 2436502. Only LoFTR registers that scene.
* Statuses of SIFT under AUTO differ from the SIFT-only baseline by design: SIFT on 2359602 is LOW_CONFIDENCE in the benchmark and SUCCESS in the stored baseline, with identical figures (section 3c).
* The sweep accepts the first confirmed location. If that location were false, the search would not continue past it; the later gates would have to reject the result, and the scene would then report FAILED even if a true location existed further along the sweep.
* **Standby can interrupt long runs.** During the known-bad run the machine entered Modern Standby on its idle timeout, 22:06:55 to 22:11:21 and 22:26:49 to 22:27:56. It is not known whether the process was fully frozen. Such a run's elapsed time is not a clean runtime. Results are unaffected. The long scripts now ask Windows to stay awake for their own lifetime (`keep_awake` in `scripts/_bootstrap.py`, no power setting changed); the screen stays on meanwhile, and the power button or lid still suspend the machine, so the event log must be checked for each long run.
* **Power interruption during the final benchmark, for timing analysis.** Windows event log (Kernel-Power 105): AC lost at 01:26:21 on 2026-11-01 and restored at 01:40:57, so the machine ran on battery for 14 min 36 s, while capacity fell from 64 190 to 55 210 mWh of 69 890. No standby occurred, and the process was neither stopped nor restarted (PID 13072). AC was verified at 01:45:14 (`PowerLineStatus: Online`). Whether Windows lowered the CPU clock on battery was not measured. Only timings can be affected, not results.
  * **Affected timing:** the LoFTR branch of 2436502 (about 01:23:42 to 01:41:33), so about 876 of its 1 071 s ran on battery. The final-run row for that branch must be read with this in mind. For a timing on AC, use the first run (975 s), which logged no power change. No other branch overlaps the window.
  * **Effect not attributable:** that branch took 9.8 % longer than in the first run (1 070.8 s against 975.5 s), with the same 540 inliers. But SP+LG on the same scene, on AC, was 9.9 % faster than in the first run (850 s against 944 s), and SP+LG on 2359602, also on AC, 29 % slower (194 s against 151 s). Run-to-run differences are therefore at least as large as this one, and it cannot be put down to the battery.
  * **A second, shorter drop:** AC was lost for 3 s at 01:54:05, at the start of the SP+LG branch of 2736702. The same notes are in `results/matcher_benchmark_notes.json`, which the generated report and the Benchmark page show next to the timings.
* Refinement can follow shading instead of terrain under a strong Sun change. A coherent-shift detector flags this.
* The cause of the along-track deviation is not established. Pushbroom attitude variation is a hypothesis; terrain relief and the product's resampling are not ruled out.
* No independent control points exist in the official sample, so no absolute accuracy is known.
* **Licence, before any public release:** SuperPoint weights are for academic or non-profit noncommercial research only (Magic Leap). LightGlue, LoFTR, Kornia and transformers are Apache-2.0. Checked against upstream on 2026-10-30.
* **GPU path unverified:** this machine has no CUDA GPU, so everything ran on CPU (8 threads).
* **RAM:** 13.3 GB in total, about 3 to 5 GB free. Run one heavy job at a time.
* No TMC-2 sample exists in the official dataset, so nothing about TMC-2 is validated.
* Do not quote the timing of one earlier pass logged at 6193 s; the machine may have been suspended.

## 8. Completed and pending tasks

**Completed in this resume:**

1. Phase 1: existing work recovered and backed up.
2. Phase 2: 2436502 located automatically by LoFTR, evaluated, reproduced and cross-checked, with all rejection checks repeated at the longer sweep (sections 3 and 5).
3. Phase 3: SP+LG verified on official imagery (section 3b).
4. Phase 4, partly: the benchmark script records failures and stays awake; the first complete run; the protocol correction; the final run for three matchers, reproducible at full precision; 5 of 6 RIFT scenes; the visual check of the first run's six selected results; the 2686502 anisotropy check.

**Pending, in order:**

1. Complete test suite, because the latest edits have not been run (section 2).
2. Finish RIFT on the one remaining scene: `.venv\Scripts\python.exe scripts\benchmark_matchers.py --rift-only --scene 1519402`. It keeps all other rows. A development trial on that scene took about 3 minutes.
3. Decide whether to fix the misleading failure reason (section 7).
4. `scripts\benchmark_representations.py`, then `scripts\synthetic_validation.py`.
5. Re-run `scripts\refinement_study.py` if refinement code changed, and `scripts\check_sift_regression.py` with the two baseline benchmarks that feed it.
6. `scripts\write_benchmark_report.py`, which writes `docs/benchmark_matchers.md`.
7. Inspect the outputs of the scene whose selection changed (2359602, now LoFTR).
8. Rebuild the frontend. With the backend running: RUN with a matcher and with AUTO, alpha overlay, downloads, control-point picker, Benchmark charts with real data. Look at the screenshots, including the charts against the dataviz anti-patterns.
9. Docs: README sections 7 to 11, `docs/benchmark_results.md`, `docs/engineering_findings.md` (partial overlap, location search, threshold 0.25, rejection results, support rule), then the rewrite of this file with final statuses.
10. **Final gate:** complete tests, backend and frontend together, major workflows, dataset hashes, no leftover processes, new backup, and the 14-item final report with the implemented / partial / missing list.

## 9. How to resume

**Exact next task: step 1 of the pending list, then step 2.** Do not re-run the corrected benchmark; it is complete and reproducible.

The master briefs are saved in `results/checkpoints/pause_20261101-025721/briefs/`. Paste them again if the conversation is lost.

Keep the laptop on AC with the lid open during long runs, and run one heavy job at a time.

```powershell
cd C:\Users\RITESH\Desktop\MAIN\SOM-DRISHTI
git status                                      # expect branch feature/advanced-matching, 68 files changed, nothing staged
Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' }   # expect none
.venv\Scripts\python.exe -m pytest backend/tests -q   # expect 224 passed, about 4 minutes
.venv\Scripts\python.exe scripts\benchmark_matchers.py --rift-only --scene 1519402
```

Check that the dataset is untouched (Git Bash, from the project root):

```bash
sha256sum -c results/dataset_sha256_baseline.txt
```

Compare the current benchmark with the saved one, if anything looks wrong (no process needed): `results/checkpoints/pause_20261101-025721/MANIFEST.txt` lists the SHA-256 of every saved file.

Start the application, each in its own terminal:

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --port 8000   # http://127.0.0.1:8000
cd frontend; npm.cmd run dev                                                     # http://localhost:5173
```

Recreate the environment only if `.venv` is lost:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r backend\requirements-learned.txt
```

The full package list is in `results/dev_probes/pip_freeze_20261030.txt`. Weights download again on first use if `models/` is lost (95 MB, ignored by Git).

## 10. Decisions

* **SuperPoint from `transformers`, LightGlue and LoFTR from Kornia, all via PyPI.** Installing `cvg/LightGlue` from its Git URL was denied by the session's permission classifier and was not retried.
* **`matcher` defaults to `SIFT`,** so existing API calls keep their behaviour.
* **The SIFT-only run stays the unchanged classical baseline.** The soft rule on unseen ground and refitting on refined points do not apply to it. Under AUTO the same standard applies to every branch.
* **AUTO uses transparent rules with no combined score.** The number of matches is not a criterion.
* **The location search supplies a starting point only.** No reported figure comes from it.
* **A location proven only through another scene's reference is never reported as that scene's result.**
* **Experiment files live in `results/dev_probes/`** (ignored by Git). They are not project code.

## 11. Files changed

Uncommitted against `915e28e`: **68 files, 23 modified and 45 untracked, none staged, none deleted.** The exact output of `git status` at the pause is in `results/checkpoints/pause_20261101-025721/git_status.txt`, and every file is listed below by group. `results/matcher_benchmark.csv` shows as untracked because `.gitignore` allows `results/*.csv`; the JSON, charts, jobs, logs and checkpoints are ignored. Probe scripts in `results/dev_probes/` are ignored too, and are saved in the checkpoint.

**Backend, new:** `matching/` `base.py`, `models.py`, `tiling.py`, `superpoint_lightglue.py`, `loftr.py`, `rift.py`, `registry.py`, `appearance.py`, `guided.py`, `localisation.py`, `selection.py`; `geometry/` `strip_model.py`, `verification.py`, `tiepoints.py`; `metadata/overlap.py`; `preprocessing/` `phase_congruency.py`, `illumination.py`; `refinement/`; `evaluation/` `validation.py`, `synthetic.py`; `registration/` `exports.py`, `control.py`; `requirements-learned.txt`.

**Backend, modified:** `config.py`, `main.py`, `api/routes.py`, `preprocessing/pipeline.py`, `registration/engine.py`, `registration/schemas.py`, `registration/warp.py`, `visualization/render.py`.

**Tests, new:** `test_strip_model.py`, `test_tiling_appearance.py`, `test_tiepoints_refinement_validation.py`, `test_representations.py`, `test_matchers.py`, `test_engine_branches.py`, `test_localisation.py`, `test_benchmark_script.py`. No existing test was removed or weakened. One exact-shape check of `/api/benchmarks` in the uncommitted `test_engine_branches.py` was extended with the new `notes` field.

**Scripts, new:** `benchmark_matchers.py`, `benchmark_representations.py`, `synthetic_validation.py`, `refinement_study.py`, `check_sift_regression.py`, `write_benchmark_report.py`. **Modified:** `run_registration.py`, `_bootstrap.py`.

**Frontend, new:** `components/RunOptions.jsx`, `GroupedBars.jsx`, `ControlPoints.jsx`, `pages/Benchmark.jsx`. **Modified:** `App.jsx`, `api.js`, `index.css`, `components/ui.jsx`, `pages/Dashboard.jsx`, `Methodology.jsx`, `ResultDetail.jsx`, `SceneDetail.jsx`, `Scenes.jsx`.

**Other:** `.env.example`, `.gitignore`, `.gitattributes`, `README.md`, `docs/engineering_findings.md`, `results/matcher_benchmark.csv`, this file.
