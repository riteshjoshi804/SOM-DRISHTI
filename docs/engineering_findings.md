# Engineering findings – classical MVP on the official SIH26166 sample data

This note records what was measured while building the MVP, including the approaches
that did **not** work. It explains the design of the registration engine and gives the
next phase (learned matchers) a factual starting point.

Numbers in sections 2–4 come from exploratory runs made during development (earlier
settings than the final defaults). The final, reproducible results are in
`results/ohrc_benchmark.csv` and `results/iirs_benchmark.csv`
(see [benchmark_results.md](benchmark_results.md)).

## 1. Dataset facts that shaped the design

Verified with `scripts/inspect_dataset.py` (full output: `results/dataset_inspection.json`).

| Finding | Consequence |
|---|---|
| 6 complete scenes: 4 OHRC (+ LRO-NAC reference), 2 IIRS (+ LRO-WAC reference). No TMC-2. | TMC-2 is shown as "not provided / not tested" everywhere. |
| OHRC sources 552–648 × 4649–5059 px, uint8, **no CRS, identity geotransform**, but **4 embedded GCPs** (lon/lat at the array corners). References 5 m polar stereographic, NoData = 0. | Source georeferencing is only known through GCPs/XML; the image-based transform is the product. |
| XML `Resolution_in_meter` 0.23 / 0.25 / 0.26 / 0.27 (OHRC), 93.74 / 83.14 (IIRS). OHRC width 12000 px × res / 5 m equals the delivered width (e.g. 12000 × 0.26 / 5 = 624). | Confirms the OHRC sources were resampled to 5 m. |
| **XML corner names ≠ TIFF array corners.** Per the GCPs, array pixel (0,0) is the XML *bottom-right* corner (3 scenes, yaw flag True) or *top-right* corner (1 scene, yaw flag False). | XML corners are shown as-is but never assumed to label the array corners; GCPs are used when present. |
| XML `*_latitude_en` / `*_longitude_en` are metres: `longitude_en` = easting, `latitude_en` = northing (the IIRS polar reference origin equals them exactly). They differ from a plain spherical polar stereographic projection by a constant ~0.19 % scale. | The `_en` values are used directly for footprint prediction. |
| IIRS D32 reference is labelled *geographic* but its geotransform is 200 m pixels in polar-stereographic metres. | Detected (`crs_label_consistent = false`) and treated as a metre grid. |
| IIRS D32 source is 104 × 4851 while its XML says 250 × 11669 (factor ≈ 0.416 → ≈ 200 m/px, matching its reference). IIRS D18 source is native 250 × 7783. | Recorded; the engine does not rely on XML sizes. |
| OHRC 2736702 reference is 29.9 % NoData (top rows + a diagonal edge) and the source footprint crosses into it. IIRS strips are 104–250 px wide inside 960–3175 px references. | Partial overlap is real; overlap-aware coverage is reported. |
| Float32 IIRS values: sources ≈ −70…1712, references ≈ −2.5…244, with negatives. | Percentile normalisation on valid pixels only; never a direct cast. |
| Most XML tags are duplicated (15–42 repeats per file), all identical. | Parser counts duplicates and reports conflicts (none found). |

## 2. Why plain SIFT + Lowe ratio + RANSAC fails on these pairs

On OHRC 2436502 (the planned first case, Sun elevation 5.1°) with default SIFT:
8 189 source / 30 578 reference keypoints, **29 matches passed ratio 0.8**, RANSAC returned
a 3-point model – i.e. nothing.

Diagnosis using the transforms later found for other scenes:

* **Keypoint orientation follows the Sun, not the terrain.** The histogram of
  (reference − source) keypoint orientation peaks near 300° on 2436502, while the
  metadata rotation is ≈ 9°. A small crater lit from another side looks like the same
  crater *rotated*, so SIFT normalises every crater to a sun-aligned frame and they all
  look alike.
* **So descriptors are not discriminative.** With the true transform known (2470502,
  2736702), only ~8–9 % of source keypoints have a reference keypoint at the right
  place (barely above chance for this keypoint density), and the true partner is the
  descriptor nearest neighbour for only **10–11** keypoints per scene (173 for 2359602).
* **The ratio test then removes most of the few correct matches**, because the second
  neighbour is another similar crater.

## 3. Things that were tried and rejected

| Idea | Result | Decision |
|---|---|---|
| Lowe-style rotation/scale pose clustering before RANSAC | The dominant cluster is the *illumination* orientation mode; true matches in the winning cluster dropped to 0 at ratio 0.9 | Removed |
| Downscaling to 10 m / 20 m | True matches fell to 0 – the matchable structure is at the finest scale (small fresh craters) | Rejected |
| Contrast-inverted source (opposite-Sun hypothesis) | 0 true matches on the two scenes with known answers; nothing on 2436502 | Rejected |
| Dense block correlation (NCC on CLAHE, gradient, low-pass) around the metadata prediction | No consistent offset across blocks | Rejected |
| **Locally guided matching** (search each keypoint's predicted position, local ratio test) | A control run with the model deliberately shifted by 150 px still produced 105–163 "inliers" – chance level is very high at this keypoint density | **Rejected: it would inflate inlier counts with coincidences** |
| No ratio test, very dense keypoints, looser 6 px geometry, stronger CLAHE on 2436502 | 0 inliers in every variant | 2436502 is reported as FAILED |

## 4. What was kept

* **More keypoints** (3000 per 768 px tile, contrast threshold 0.02): true matches rose
  from ~10 to ~17–35 per hard scene.
* **Two-stage verification**: affine RANSAC seed on the lowest-ratio matches, pool
  widened only if needed (5 spatially distinct inliers, ≥ 16 px apart), then expansion
  over all ratio-filtered *global* nearest-neighbour matches near the seed model.
  Because these are global matches, the expected number of chance inliers is
  `N · π · t² / A` – about 0.001–0.004 here – and is reported with every result.
* **Mirrored-source hypothesis**: SIFT is not mirror-invariant. OHRC 2359602 has **no**
  geometry as delivered and 229 tie points after a left-right flip; the resulting
  transform has determinant −1.02. The GCPs predict the mirror, but the image content
  also shows the array reversed along-track relative to the GCPs (rotation differs by
  ~180°).
* **Pooling representations**: raw, CLAHE and gradient images recover different true
  matches; verifying their candidates jointly raised 2736702 from 19 (best single mode)
  to 32 tie points.

## 5. Geolocation of the metadata

For every accepted registration the image-based rotation agrees with the XML/GCP
prediction to within 1.7° (after accounting for the 180° array reversal of 2359602),
but the footprint position differs by **2.3–6 km** for OHRC and ~19 km for the IIRS
polar strip. The XMLs state `ReferenceUsed = System`, i.e. system-level pointing. This
is why the metadata is only used as a plausibility report and never to guide matching.
