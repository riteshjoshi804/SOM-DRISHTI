import { useEffect, useState } from 'react'
import { api } from '../api.js'
import { Card, SectionTitle } from '../components/ui.jsx'

const STEPS = [
  ['Dataset indexing', 'Every XML defines a scene; its source and reference TIFFs are paired by scene ID. Nothing is hard-coded and the dataset is opened read-only.'],
  ['Metadata', 'The product XML is parsed for resolution, Sun azimuth/elevation, attitude and corner coordinates (geographic, refined, easting/northing). Absent fields stay empty; duplicated tags are checked for conflicts.'],
  ['Raster loading', 'Rasterio reads the original values (uint8 OHRC, float32 IIRS). A valid-pixel mask excludes NoData, NaN/Inf and zero fill touching the image border; interior zeros (hard shadows) remain data.'],
  ['Representations', 'Each image is percentile-stretched (1–99%, valid pixels only) to 8 bit. Three representations are tried: raw_normalized, clahe (local contrast equalisation) and gradient (Scharr magnitude). None of them removes real shadow differences.'],
  ['SIFT', 'Keypoints are detected tile by tile (768 px tiles, 3000 keypoints per tile, contrastThreshold 0.02) inside an eroded valid mask, and described with RootSIFT. Images are never resized to each other and the source is not assumed to lie inside the reference.'],
  ['Matching', 'Each source descriptor gets its two nearest reference descriptors (brute force, L2). Lowe ratio test at 0.90, then one-to-one filtering on the reference side.'],
  ['Geometric verification', 'Seed: affine RANSAC on the most distinctive matches first, widening the pool only if needed; a seed needs 5 inliers at least 16 px apart. Expansion: candidates near the seed model are re-fitted (MAGSAC, 3 px). A homography replaces the affine only with 30+ tie points, a 20% gain and a sane geometry.'],
  ['Mirrored source', 'SIFT is not mirror-invariant. If the delivered source yields no geometry, the same source flipped left-right is tried. One official OHRC scene only registers this way.'],
  ['Pooling', 'If no single representation reaches SUCCESS, the candidates of all representations are verified together against one transformation.'],
  ['Validation', 'Finite values, determinant, scale 0.2–5, anisotropy ≤ 2, shear ≤ 30°, perspective, convex footprint, and at least 2% of the source landing inside the reference.'],
  ['Registration', 'The source is warped into the reference pixel grid (warpAffine / warpPerspective). A GeoTIFF with the original source values and the reference georeferencing is written alongside the PNG previews.'],
]

const METRICS = [
  ['Candidate matches', 'Ratio-test survivors after one-to-one filtering.'],
  ['Inliers', 'Candidate matches consistent with the final transformation within the reprojection threshold, counted once per location (2 px) in both images.'],
  ['Inlier ratio', 'Inliers / candidate matches. On these cross-illumination pairs it is often only a few percent.'],
  ['Expected chance inliers', 'Wrong global matches that would agree with the model by coincidence: candidates × π·t² / valid reference area. Compare it with the inlier count.'],
  ['Reprojection RMSE / median', 'Residuals of the inliers under the fitted model. They show internal consistency. They are NOT independent ground-truth registration accuracy.'],
  ['Cross-validated RMSE', 'Each inlier’s error when it is held out of the fit (5-fold). Less optimistic, still not ground truth.'],
  ['Image-space Spatial Coverage', 'Share of valid cells of an 8 × 8 grid over the source image that contain at least one tie point. It describes the image grid, not a geographic area.'],
  ['Coverage within overlap', 'The same, counting only cells that overlap valid reference pixels, since the rest can never be matched.'],
  ['Status', 'FAILED: a hard limit was missed (keypoints, matches, inliers ≥ 12, inlier ratio ≥ 0.5%, valid geometry, coverage ≥ 5%). LOW_CONFIDENCE: accepted but below a soft target (30 inliers, 5% ratio, 25% coverage, 2.5 px RMSE). SUCCESS otherwise.'],
]

const LIMITS = [
  'Strong Sun-angle differences and shadow reversal: most small craters match some other crater. One OHRC scene and one IIRS scene fail completely.',
  'IIRS-to-LRO-WAC is a different modality (float32 radiance vs. WAC reflectance) and a very narrow strip.',
  'Low-texture and deeply shadowed terrain gives no features at all; coverage stays low where the source is dark.',
  'A single affine/homography cannot absorb terrain parallax, attitude jitter or map-projection scale change over long strips.',
  'XML/GCP geolocation differs from the image-based result by several km on these samples; it is reported as a plausibility check only.',
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
        This MVP is a classical, fully explainable pipeline: SIFT features, ratio-test matching and robust geometry,
        tested on the official SIH sample data. Its failures are reported as failures - they show where learned
        multi-modal matchers are needed.
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
        <p className="mt-4 text-sm text-ink-2">
          Planned next steps: SuperPoint + LightGlue, LoFTR, multi-modal descriptors and DEM-assisted geometry. None
          of them is part of this MVP.
        </p>
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
