import { useEffect, useState } from 'react'
import { api } from '../api.js'
import { Card, SectionTitle } from '../components/ui.jsx'

const STEPS = [
  ['Dataset indexing', 'Every XML defines a scene; its source and reference TIFFs are paired by scene ID. Nothing is hard-coded and the dataset is opened read-only.'],
  ['Metadata', 'The product XML is parsed for resolution, Sun azimuth/elevation, attitude and corner coordinates (geographic, refined, easting/northing). Absent fields stay empty; duplicated tags are checked for conflicts.'],
  ['Raster loading', 'Rasterio reads the original values (uint8 OHRC, float32 IIRS). A valid-pixel mask excludes NoData, NaN/Inf and zero fill touching the image border; interior zeros (hard shadows) remain data.'],
  ['Search prior', 'The XML/GCP footprint gives an orientation, a scale and a rough position, used only as a search prior. The reference CRS is read from its own WKT, and a body other than the Moon, a missing CRS or an implausible scale disables the prior. No Earth CRS or EPSG code is ever assumed. Without a usable prior the whole reference is searched, in every orientation.'],
  ['Representations', 'Each image is percentile-stretched (1–99%, valid pixels only) to 8 bit. Available: normalized grayscale, CLAHE, gradient, structural (gradient over its local mean), phase congruency, shadow-aware. None of them removes real shadow differences, and they are only ever used for matching and display: the scientific values stay untouched.'],
  ['SIFT branch', 'Tiled keypoints (768 px tiles, 3000 per tile, contrastThreshold 0.02), RootSIFT descriptors, brute-force 2-NN, Lowe ratio 0.90, one-to-one filtering. If the delivered source yields no geometry, the same source flipped left-right is tried, and if no single representation reaches SUCCESS their candidates are verified together. This branch is the unchanged classical baseline.'],
  ['SuperPoint + LightGlue branch', 'SuperPoint keypoints matched by LightGlue, filtered by LightGlue confidence. The Lowe ratio test does not apply.'],
  ['LoFTR branch', 'Detector-free matching, independent of the other two, filtered by LoFTR coarse confidence. The Lowe ratio test does not apply. Coordinates are corrected for the half-pixel convention of the pretrained models.'],
  ['Location search (learned branches)', 'Before any matcher runs, each source tile, rotated into the reference frame, is correlated against its search region on gradient magnitude, which survives a changed Sun better than intensity. Only valid pixels are compared, so a tile may hang over the edge of the reference. A location counts only if tiles that share no source pixels agree on it. This supplies a starting point and nothing else: if it finds nothing, or its location leads nowhere, the matcher searches by itself.'],
  ['Guided search (learned branches)', 'Learned matchers are not rotation invariant, and one image is a small part of the other. So a source tile, rotated into the reference frame, slides over the search region at reduced resolution, and a translation vote separates right from wrong windows. Every tile position takes part, because a strip can lie partly off its reference and texture does not tell which tiles have ground. A candidate counts only once another, independent tile with ground at that offset confirms it. All tiles with reference ground are then matched at full resolution, in three passes that re-centre on the improving model. Window positions are jittered so that bias tied to the position in a window cannot look like agreement.'],
  ['Appearance check', 'Each learned match must also pass a comparison that is independent of the matcher: the gradient-magnitude patches at the two points must correlate. This removes look-alike craters and matches inside black shadow before any geometry is estimated.'],
  ['Geometric verification', 'All branches share one verification. Seed: affine RANSAC on the most distinctive matches first, widening the pool only if needed; a seed needs 5 inliers at least 16 px apart. Expansion: candidates near the seed model are re-fitted (USAC-MAGSAC, RANSAC fallback, 3 px). A homography replaces the affine only with 30+ tie points, a 20% gain and a sane geometry.'],
  ['Along-track model', 'Long strips can deviate from any single transformation. A smooth along-track model (cubic P-spline for scan-line position, plus a cross-track term) is therefore also fitted, and adopted only when it predicts stretches of the image left out of its fit better than the global model does. A very stiff spline is exactly an affine, so the data decide how flexible it is. Its local scale, shear and orientation are checked along the whole strip.'],
  ['Decoy controls', 'Learned matchers return matches for any input. So the final pass is repeated on ground known to be wrong, and the result must exceed the inliers found there by a factor of three. The SIFT equivalent removes the true ground from the reference before matching.'],
  ['Validation', 'Finite values, determinant, scale 0.2–5, anisotropy ≤ 2, shear ≤ 30°, perspective, convex footprint, at least 2% of the source landing inside the reference, and for guided results agreement with the metadata orientation and scale.'],
  ['Tie-point selection', 'From the verified inliers: duplicate suppression, confidence-aware ranking, Adaptive Non-Maximal Suppression and a per-cell cap. Only verified points can be selected. Cells without a match stay empty, and too few points are reported, never invented.'],
  ['Sub-pixel refinement', 'The source point stays fixed. A source patch is resampled into the reference frame through the local model, and the shift to the reference patch is measured by normalised cross-correlation with a parabolic peak fit (Lucas–Kanade, ECC and phase correlation are also available). Points are rejected when the patch is flat or invalid, or the peak is ambiguous, weak or at the search limit. Initial coordinates are always kept. The model is refitted on refined points only if held-out agreement improves; the SIFT baseline is never refitted.'],
  ['AUTO selection', 'Runs the available matchers and compares accepted results by status, then coverage within the overlap, held-out median error, inlier ratio, reprojection RMSE and time, in that order; a difference too small to mean anything does not decide. The number of matches is deliberately not a criterion. If another matcher’s tie points disagree strongly with the chosen model, the status is lowered.'],
  ['Registration', 'The source is warped into the reference pixel grid. A GeoTIFF with the original source values and the reference geotransform is written alongside the previews. The CRS is copied only if its label agrees with the geotransform; none is ever invented.'],
]

const METRICS = [
  ['Candidate matches', 'Correspondences that passed the matcher’s own filter (and, for learned matchers, the appearance check).'],
  ['Inliers', 'Candidates consistent with the final model within the reprojection threshold, counted once per location (2 px) in both images.'],
  ['Inlier ratio', 'Inliers / candidate matches.'],
  ['A. Reprojection RMSE / median', 'Residuals of the inliers under the fitted model. They show internal consistency. They are NOT independent ground-truth registration accuracy.'],
  ['B. Held-out transfer error', 'Candidates are split by grid cell before robust estimation, a model is fitted without the held-out ones, and they are measured against it. The split is repeated. Errors are reported inside a 10 px gate, with the number of candidates in the gate and the number expected there by chance. Real out-of-sample evidence, but from the same matcher, so not ground truth.'],
  ['B. Unseen blocks', 'Contiguous along-track stretches are left out in turn, which tests prediction of ground the model has not seen. This is the harder test on a long strip.'],
  ['C. Control-point error', 'Independently identified points, applied to the finished transformation and never fitted. Only this may be called independent accuracy. The official sample contains none, so it stays empty until someone supplies them.'],
  ['Synthetic error', 'Error against an exactly known transformation on rendered terrain. Shows the machinery works and whether refinement reduces error. Always labelled synthetic; it does not predict accuracy on real scenes.'],
  ['Decoy inliers', 'The most inliers the same pipeline finds on ground known to be wrong.'],
  ['Expected chance inliers', 'Candidates × π·t² / search area. It ignores the search over models, so it understates what a learned matcher can reach; the decoy figure is the stronger test.'],
  ['Image-space Spatial Coverage', 'Share of valid cells of an 8 × 8 grid over the source image that contain at least one tie point. It describes the image grid, not a geographic area.'],
  ['Coverage within overlap', 'The same, counting only cells that overlap valid reference pixels, since the rest can never be matched.'],
  ['Distribution score', 'Normalised entropy of the tie-point counts over the valid cells of the overlap: 1 means equally many points in every cell, 0 means all points in one. Empty cells lower it.'],
  ['Status', 'FAILED: a hard limit was missed. For SIFT: keypoints, matches, inliers ≥ 12, inlier ratio ≥ 0.5%, valid geometry, coverage ≥ 5%. For learned matchers also inliers ≥ 40, ratio ≥ 2%, agreement with the metadata orientation and scale, and three times the decoy result. LOW_CONFIDENCE: accepted but below a soft target (30 inliers, 5% ratio, 25% coverage, 2.5 px RMSE), or with no null control, or (except in a SIFT-only run, which stays the unchanged baseline) with less than half of the strip length supported by tie points, at least 3 per twentieth of the strip, or predicting left-out along-track blocks worse than 5 px RMSE. SUCCESS otherwise.'],
]

const LIMITS = [
  'Strong Sun-angle differences and shadow reversal: most small craters resemble some other crater. A correspondence inside moved shadows cannot be recovered by any method here.',
  'Learned matchers were trained on Earth photographs, not on lunar orbital imagery. How well they do here is an empirical result of this benchmark, not a property to rely on elsewhere.',
  'IIRS-to-LRO-WAC is a different modality (float32 radiance vs. WAC reflectance) and a very narrow strip.',
  'Low-texture and deeply shadowed terrain gives no matches at all, so coverage stays low there. Parts of a strip with no tie points are extrapolated by the model.',
  'The cause of the along-track deviation has not been established. Pushbroom attitude variation is a hypothesis; terrain relief and the resampling of the delivered products have not been ruled out.',
  'No held-out figure is independent of the matcher, and the official sample has no control points, so no absolute accuracy is known. Sub-pixel accuracy is not claimed on real scenes.',
  'XML/GCP geolocation differs from the image-based result by kilometres on these samples; it is a search prior and a plausibility check, not ground truth.',
  'The acceptance limits were calibrated on these six scenes and may not transfer. The GPU path could not be tested on the development machine.',
  'No TMC-2 sample exists in the official ZIP, so nothing about TMC-2 has been validated.',
]

export default function Methodology() {
  const [cfg, setCfg] = useState(null)
  useEffect(() => {
    api.defaultConfig().then(setCfg).catch(() => setCfg(null))
  }, [])

  return (
    <div className="max-w-4xl">
      <h1 className="text-2xl font-bold tracking-tight">Methodology</h1>
      <p className="mt-2 text-ink-2">
        Three independent matchers feed one verification and one report. Everything reported is measured on the run it belongs to, and failures are
        reported as failures.
      </p>

      <SectionTitle>Pipeline</SectionTitle>
      <ol className="space-y-3">
        {STEPS.map(([title, text], i) => (
          <li key={title} className="flex gap-4">
            <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-white/10 text-xs font-semibold">
              {i + 1}
            </span>
            <div>
              <div className="font-semibold">{title}</div>
              <p className="text-sm leading-relaxed text-ink-2">{text}</p>
            </div>
          </li>
        ))}
      </ol>

      <SectionTitle>Metrics</SectionTitle>
      <Card>
        <dl className="space-y-3">
          {METRICS.map(([k, v]) => (
            <div key={k}>
              <dt className="font-semibold">{k}</dt>
              <dd className="text-sm leading-relaxed text-ink-2">{v}</dd>
            </div>
          ))}
        </dl>
      </Card>

      <SectionTitle>Known limitations</SectionTitle>
      <Card>
        <ul className="list-disc space-y-1.5 pl-5 text-sm text-ink-2">
          {LIMITS.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
      </Card>

      <SectionTitle hint="Served live by the backend">Default configuration</SectionTitle>
      <Card>
        {cfg ? (
          <details>
            <summary className="cursor-pointer text-sm text-accent">Show all parameters</summary>
            <pre className="tabular mt-3 overflow-x-auto text-xs leading-relaxed text-ink-2">{JSON.stringify(cfg, null, 2)}</pre>
          </details>
        ) : (
          <p className="text-sm text-muted">Backend not reachable.</p>
        )}
      </Card>
    </div>
  )
}
