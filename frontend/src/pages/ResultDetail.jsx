import { useEffect, useState } from 'react'
import { api } from '../api.js'
import { NA, deg, int, num, pct, withUnit } from '../format.js'
import { Card, ErrorNote, ImagePanel, KeyValue, Loading, MetricCard, SectionTitle, StatusBadge } from '../components/ui.jsx'
import ControlPoints from '../components/ControlPoints.jsx'

const MATCHER_LABELS = {
  SIFT: 'SIFT / RootSIFT',
  SUPERPOINT_LIGHTGLUE: 'SuperPoint + LightGlue',
  LOFTR: 'LoFTR',
  RIFT: 'RIFT (experimental)',
}

function Progress({ job }) {
  return (
    <Card className="mt-6">
      <div className="flex items-center justify-between text-sm">
        <StatusBadge status={job.state} />
        <span className="tabular text-ink-2">{Math.round(job.progress * 100)}%</span>
      </div>
      <div className="mt-3 h-2 overflow-hidden rounded-full bg-white/10">
        <div className="h-full rounded-full bg-accent transition-all" style={{ width: `${Math.max(2, job.progress * 100)}%` }} />
      </div>
      <p className="mt-2 text-sm text-ink-2">{job.message || 'waiting for the worker'}</p>
      <p className="mt-1 text-xs text-muted">
        SIFT takes under a minute per scene. A learned matcher takes several minutes on a CPU, and a scene where nothing matches takes longest, because
        the whole search region is tried before FAILED is reported.
      </p>
    </Card>
  )
}

function Matrix({ m }) {
  return (
    <pre className="tabular overflow-x-auto rounded-lg bg-black/40 p-3 text-xs leading-relaxed">
      {m.map((row) => row.map((v) => v.toFixed(6).padStart(14)).join(' ')).join('\n')}
    </pre>
  )
}

function Attempts({ attempts }) {
  const showMatcher = attempts.some((a) => a.matcher)
  return (
    <div className="overflow-x-auto rounded-xl border border-white/10 bg-surface">
      <table className="w-full text-sm">
        <thead className="border-b border-white/10 text-left text-xs text-ink-2">
          <tr>
            {showMatcher && <th className="px-3 py-2 font-medium">Matcher</th>}
            <th className="px-3 py-2 font-medium">Representation</th>
            <th className="px-3 py-2 font-medium">Source</th>
            <th className="px-3 py-2 text-right font-medium">Keypoints (src / ref)</th>
            <th className="px-3 py-2 text-right font-medium">Candidates</th>
            <th className="px-3 py-2 text-right font-medium">Inliers</th>
            <th className="px-3 py-2 font-medium">Outcome</th>
            <th className="px-3 py-2 font-medium">Reason</th>
          </tr>
        </thead>
        <tbody className="tabular">
          {attempts.map((a, i) => (
            <tr key={i} className="border-b border-white/5 align-top last:border-0">
              {showMatcher && <td className="px-3 py-2">{MATCHER_LABELS[a.matcher] || a.matcher || 'SIFT / RootSIFT'}</td>}
              <td className="px-3 py-2">{a.preprocessing_mode}</td>
              <td className="px-3 py-2 text-ink-2">{a.mirrored_source ? 'mirrored' : 'as delivered'}</td>
              <td className="px-3 py-2 text-right">
                {int(a.source_keypoints)} / {int(a.reference_keypoints)}
              </td>
              <td className="px-3 py-2 text-right">{int(a.ratio_filtered_matches)}</td>
              <td className="px-3 py-2 text-right">{int(a.inliers)}</td>
              <td className="px-3 py-2">
                <StatusBadge status={a.status} />
              </td>
              <td className="px-3 py-2 text-xs text-ink-2">{a.reasons.join('; ') || '–'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// The same crop of both images, blended by the reader. Terrain that lines up does not move when the slider does.
function AlphaOverlay({ reference, registered }) {
  const [alpha, setAlpha] = useState(50)
  return (
    <Card>
      <div className="relative mx-auto max-w-full overflow-hidden rounded-lg bg-black/40" style={{ width: 'fit-content' }}>
        <img src={reference} alt="Reference, same crop as the registered source" className="block max-h-[640px] w-auto max-w-full" />
        <img src={registered} alt="Registered source" className="absolute inset-0 h-full w-full" style={{ opacity: alpha / 100 }} />
      </div>
      <label className="mt-3 flex items-center gap-3 text-xs text-ink-2">
        <span className="whitespace-nowrap">Reference</span>
        <input type="range" min="0" max="100" value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} className="w-full accent-[var(--color-accent)]" aria-label="Opacity of the registered source" />
        <span className="whitespace-nowrap">Registered source</span>
      </label>
      <p className="mt-2 text-xs text-muted">Slide between the two. Aligned terrain stays put, though shading can differ because the Sun does.</p>
    </Card>
  )
}

function Evidence({ r }) {
  const m = r.metrics || {}
  const h = r.holdout
  const cp = r.control_points
  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <Card>
        <div className="text-xs font-semibold uppercase tracking-wider text-ink-2">A. Geometric consistency</div>
        <p className="mt-1 text-xs leading-relaxed text-muted">Residuals of the points used for the fit. Shows internal consistency, not accuracy.</p>
        <div className="mt-3">
          <KeyValue
            rows={[
              ['Reprojection RMSE', withUnit(m.reprojection_rmse_px, 'px')],
              ['Median', withUnit(m.median_reprojection_error_px, 'px')],
              ['After refinement, RMSE', withUnit(m.refined_reprojection_rmse_px, 'px')],
              ['5-fold on inliers', withUnit(m.cross_validated_rmse_px, 'px')],
            ]}
          />
        </div>
      </Card>
      <Card>
        <div className="text-xs font-semibold uppercase tracking-wider text-ink-2">B. Held-out validation</div>
        <p className="mt-1 text-xs leading-relaxed text-muted">Automatic matches left out of the fit. Real out-of-sample evidence, but from the same matcher, so not independent ground truth.</p>
        <div className="mt-3">
          {h ? (
            <KeyValue
              rows={[
                ['Training RMSE', withUnit(h.training_reprojection_rmse_px, 'px')],
                ['Held-out RMSE', withUnit(h.heldout_transfer_rmse_px, 'px')],
                ['Held-out median', withUnit(h.heldout_median_px, 'px')],
                ['Held-out 90th pct.', withUnit(h.heldout_p90_px, 'px')],
                ['Within the gate', `${int(h.heldout_within_gate)} of ${int(h.heldout_candidates)} (${num(h.gate_px, 0)} px)`],
                ['Unseen blocks RMSE', `${withUnit(h.block_rmse_px, 'px')} (${h.block_count} blocks)`],
              ]}
            />
          ) : (
            <p className="text-sm text-muted">Not measured for this run.</p>
          )}
          {h?.reason && <p className="mt-2 text-xs text-warning">! {h.reason}</p>}
        </div>
      </Card>
      <Card>
        <div className="text-xs font-semibold uppercase tracking-wider text-ink-2">C. Independent accuracy</div>
        <p className="mt-1 text-xs leading-relaxed text-muted">Independently identified control points, never used in a fit. Only this may be called accuracy.</p>
        <div className="mt-3">
          {cp ? (
            <>
              <KeyValue
                rows={[
                  ['Control points', cp.count],
                  ['RMSE', withUnit(cp.rmse_px, 'px')],
                  ['Median', withUnit(cp.median_px, 'px')],
                  ['90th pct.', withUnit(cp.p90_px, 'px')],
                  ['Provenance', cp.provenance],
                ]}
              />
              {cp.warnings.map((w) => (
                <p key={w} className="mt-2 text-xs text-warning">
                  ! {w}
                </p>
              ))}
            </>
          ) : (
            <p className="text-sm text-ink-2">Not available. The official sample has no independent control points, and none have been supplied for this run.</p>
          )}
        </div>
      </Card>
    </div>
  )
}

function Matchers({ r, jobId, art }) {
  const rows = r.matcher_results
  return (
    <>
      <div className="overflow-x-auto rounded-xl border border-white/10 bg-surface">
        <table className="w-full text-sm">
          <thead className="border-b border-white/10 text-left text-xs text-ink-2">
            <tr>
              <th className="px-3 py-2 font-medium">Matcher</th>
              <th className="px-3 py-2 font-medium">Outcome</th>
              <th className="px-3 py-2 text-right font-medium">Accepted</th>
              <th className="px-3 py-2 text-right font-medium">Inliers</th>
              <th className="px-3 py-2 text-right font-medium">Ratio</th>
              <th className="px-3 py-2 text-right font-medium">RMSE (px)</th>
              <th className="px-3 py-2 text-right font-medium">Held-out median (px)</th>
              <th className="px-3 py-2 text-right font-medium">Coverage</th>
              <th className="px-3 py-2 text-right font-medium">Decoy max</th>
              <th className="px-3 py-2 text-right font-medium">Time (s)</th>
              <th className="px-3 py-2 font-medium">Views</th>
            </tr>
          </thead>
          <tbody className="tabular">
            {rows.map((m) => {
              const decoys = m.decoys?.filter((d) => d.tiles_used > 0) || []
              const prefix = m.matcher.toLowerCase()
              return (
                <tr key={m.matcher} className="border-b border-white/5 align-top last:border-0">
                  <td className="px-3 py-2 whitespace-nowrap">
                    {MATCHER_LABELS[m.matcher] || m.matcher}
                    {r.selected_matcher === m.matcher && <span className="ml-1 text-xs text-ink-2">(selected)</span>}
                  </td>
                  <td className="px-3 py-2">
                    <StatusBadge status={m.availability === 'UNAVAILABLE' ? 'UNAVAILABLE' : m.registration_status} />
                    {(m.reasons.length > 0 || m.warnings.length > 0) && <div className="mt-1 max-w-sm text-xs text-ink-2">{[...m.reasons, ...m.warnings].join('; ')}</div>}
                  </td>
                  <td className="px-3 py-2 text-right">{int(m.accepted_count)}</td>
                  <td className="px-3 py-2 text-right">{int(m.inliers)}</td>
                  <td className="px-3 py-2 text-right">{pct(m.inlier_ratio)}</td>
                  <td className="px-3 py-2 text-right">{num(m.reprojection_rmse_px)}</td>
                  <td className="px-3 py-2 text-right">{num(m.holdout?.heldout_median_px)}</td>
                  <td className="px-3 py-2 text-right">{pct(m.coverage_within_overlap ?? m.image_space_spatial_coverage)}</td>
                  <td className="px-3 py-2 text-right">{decoys.length ? int(Math.max(...decoys.map((d) => d.inliers))) : NA}</td>
                  <td className="px-3 py-2 text-right">{num(m.processing_time_seconds, 0)}</td>
                  <td className="px-3 py-2 text-xs">
                    {art(`${prefix}_overlay`) && (
                      <a className="mr-2 text-accent hover:underline" href={art(`${prefix}_overlay`)} target="_blank" rel="noreferrer">
                        Overlay
                      </a>
                    )}
                    {art(`${prefix}_inlier_matches`) && (
                      <a className="text-accent hover:underline" href={art(`${prefix}_inlier_matches`)} target="_blank" rel="noreferrer">
                        Inliers
                      </a>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      {r.selection && (
        <Card className="mt-3">
          <p className="text-xs text-muted">{r.selection.rule}</p>
          {r.selection.reasons.map((x) => (
            <p key={x} className="mt-2 text-sm">
              {x}
            </p>
          ))}
          {r.selection.agreement?.map((g) => (
            <p key={g.other} className="mt-2 text-sm text-ink-2">
              {MATCHER_LABELS[g.other] || g.other} tie points under the selected model: median {num(g.median_px)} px, 90th percentile {num(g.p90_px)} px,{' '}
              {int(g.within_threshold)} of {int(g.points)} within the threshold.
            </p>
          ))}
          {r.selection.warnings?.map((x) => (
            <p key={x} className="mt-2 text-sm">
              <span className="font-semibold text-warning">! </span>
              {x}
            </p>
          ))}
        </Card>
      )}
    </>
  )
}

const DOWNLOADS = [
  ['report_html', 'Scientific report (HTML)'],
  ['report_markdown', 'Scientific report (Markdown)'],
  ['registered_geotiff', 'Registered GeoTIFF, original values on the reference grid'],
  ['registered_source', 'Registered preview (PNG)'],
  ['reference', 'Reference preview'],
  ['overlay', 'Overlay'],
  ['correspondences_raw', 'Raw correspondences (CSV)'],
  ['correspondences_inliers', 'Verified inliers (CSV)'],
  ['tie_points_selected', 'Spatially selected tie points (CSV)'],
  ['tie_points_refined', 'Sub-pixel refined tie points (CSV)'],
  ['transformation', 'Transformation (JSON)'],
  ['metadata_summary', 'Metadata summary (JSON)'],
  ['metrics', 'Metrics (JSON)'],
  ['benchmark_row', 'Benchmark row (CSV)'],
]

function Downloads({ jobId, art }) {
  const available = DOWNLOADS.filter(([key]) => art(key))
  return (
    <Card>
      <a className="text-sm font-semibold text-accent hover:underline" href={api.downloadUrl(jobId)}>
        Download everything as a ZIP
      </a>
      <ul className="mt-3 grid gap-x-6 gap-y-1 text-sm sm:grid-cols-2">
        {available.map(([key, label]) => (
          <li key={key}>
            <a className="text-accent hover:underline" href={art(key)}>
              {label}
            </a>
          </li>
        ))}
        <li>
          <a className="text-accent hover:underline" href={api.artifactUrl(jobId, 'result.json')}>
            Full result (JSON)
          </a>
        </li>
      </ul>
    </Card>
  )
}

export default function ResultDetail({ jobId }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [refresh, setRefresh] = useState(0)

  useEffect(() => {
    let timer
    let alive = true
    const load = () =>
      api
        .result(jobId)
        .then((d) => {
          if (!alive) return
          setData(d)
          if (d.job.state === 'queued' || d.job.state === 'running') timer = setTimeout(load, 1000)
        })
        .catch((e) => alive && setError(e))
    load()
    return () => {
      alive = false
      clearTimeout(timer)
    }
  }, [jobId, refresh])

  const job = data?.job
  const r = data?.result
  const m = r?.metrics || {}
  const art = (key) => (r?.artifacts?.[key] ? api.artifactUrl(jobId, r.artifacts[key]) : null)
  const cov = r?.spatial_coverage
  const t = r?.transformation
  const mc = r?.metadata_consistency
  const ov = r?.overlap_estimate
  const tp = r?.tie_points
  const rf = r?.refinement
  const rv = r?.refinement_validation
  const il = r?.illumination
  const isSift = !r?.selected_matcher || r?.selected_matcher === 'SIFT'
  const matcherName = MATCHER_LABELS[m.matcher] || m.matcher

  return (
    <div>
      <a href="#/results" className="text-sm text-accent hover:underline">
        ← All results
      </a>
      <div className="mt-2 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="text-xs uppercase tracking-wider text-ink-2">
            {r?.sensor || job?.label || 'Registration'} {r?.reference_sensor ? `→ ${r.reference_sensor}` : ''}
            {r?.requested_matcher ? ` · requested ${MATCHER_LABELS[r.requested_matcher] || r.requested_matcher}` : ''}
            {r?.device ? ` · ${r.device}` : ''}
          </div>
          <h1 className="break-all font-mono text-xl font-semibold">{job?.scene_id || job?.label || jobId}</h1>
          <div className="mt-1 font-mono text-xs text-muted">{jobId}</div>
        </div>
        {r && <StatusBadge status={r.status} className="px-4 py-1 text-sm" />}
      </div>
      <ErrorNote error={error} />
      {!data && !error && <Loading />}
      {job && (job.state === 'queued' || job.state === 'running') && <Progress job={job} />}
      {job?.state === 'error' && (
        <Card className="mt-6 border-critical/50">
          <StatusBadge status="FAILED" />
          <p className="mt-2 text-sm">{job.error}</p>
        </Card>
      )}

      {r && (
        <>
          <Card className={`mt-6 ${r.status === 'FAILED' ? 'border-critical/50' : r.status === 'LOW_CONFIDENCE' ? 'border-warning/40' : 'border-good/40'}`}>
            <div className="text-xs font-semibold uppercase tracking-wider text-ink-2">Why this status</div>
            {r.status === 'SUCCESS' && <p className="mt-1 text-sm">Every hard limit and soft target was met. That says the geometry holds together, not how accurate it is: see the evidence below.</p>}
            {r.failure_reasons.map((x) => (
              <p key={x} className="mt-1 text-sm">
                <span className="font-semibold text-critical">✕ </span>
                {x}
              </p>
            ))}
            {r.warnings.map((x) => (
              <p key={x} className="mt-1 text-sm">
                <span className="font-semibold text-warning">! </span>
                {x}
              </p>
            ))}
          </Card>

          <SectionTitle hint="All values measured in this run">Metrics{matcherName ? `: ${matcherName}` : ''}</SectionTitle>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <MetricCard
              label="Candidate Matches"
              value={int(m.ratio_filtered_matches)}
              hint={isSift ? `Ratio-test survivors of ${int(m.raw_matches)} nearest-neighbour candidates` : `Passed confidence and appearance checks, out of ${int(m.raw_matches)} proposed`}
            />
            <MetricCard label="Inliers" value={int(m.inliers)} hint="Spatially distinct tie points consistent with the transformation" />
            <MetricCard label="Inlier Ratio" value={pct(m.inlier_ratio, 2)} hint="Inliers / candidate matches" />
            <MetricCard label="Reprojection RMSE" value={withUnit(m.reprojection_rmse_px, 'px')} hint="On the fitted matches - not ground-truth accuracy" />
            <MetricCard label="Held-out error" value={withUnit(m.heldout_median_px, 'px')} hint="Median, on automatic matches left out of the fit - not ground truth" />
            <MetricCard
              label="Spatial Coverage"
              value={pct(m.image_space_spatial_coverage)}
              hint={`Image-space: ${int(m.occupied_cells)} of ${int(m.valid_cells)} valid source grid cells hold a tie point`}
            />
            <MetricCard label="Processing Time" value={withUnit(m.processing_time_seconds, 's', 1)} hint={r.device || undefined} />
            <div className="flex flex-col justify-center rounded-xl border border-white/10 bg-surface p-4">
              <div className="text-xs text-ink-2">Status</div>
              <div className="mt-2">
                <StatusBadge status={r.status} />
              </div>
            </div>
          </div>
          <div className="mt-3 grid grid-cols-2 gap-3 md:grid-cols-4">
            <MetricCard
              label="Keypoints (source / reference)"
              value={m.source_keypoints == null ? 'Detector-free' : `${int(m.source_keypoints)} / ${int(m.reference_keypoints)}`}
              hint={r.selected_preprocessing?.startsWith('pooled') ? 'Summed over the pooled representations' : `In the ${r.selected_preprocessing} representation`}
            />
            <MetricCard
              label="Decoy inliers"
              value={m.decoy_max_inliers == null ? NA : int(m.decoy_max_inliers)}
              hint="Most inliers the same pipeline found on ground known to be wrong; the result must stand clearly above it"
            />
            <MetricCard label="Expected chance inliers" value={num(m.expected_chance_inliers, 4)} hint="Analytic estimate; it ignores model search, so the decoy figure is the stronger test" />
            <MetricCard
              label="Coverage within overlap"
              value={pct(m.coverage_within_overlap)}
              hint={`Estimated overlap: ${pct(m.estimated_overlap_fraction)} of valid source pixels`}
            />
          </div>

          {r.evidence && (
            <>
              <SectionTitle hint="Three different questions, never to be read as one another">Evidence</SectionTitle>
              <Evidence r={r} />
            </>
          )}

          <SectionTitle>Images</SectionTitle>
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            <ImagePanel title="Source" src={art('source')} caption="Percentile-stretched source (as delivered)" />
            <ImagePanel title="Reference" src={art('reference')} caption="Percentile-stretched reference" />
            <ImagePanel title="Estimated overlap" src={art('reference_footprint')} caption="Green: registered source footprint. Yellow: footprint predicted from XML/GCP metadata." />
            {art('keypoints_source') && <ImagePanel title="Keypoints - source" src={art('keypoints_source')} caption={`${int(m.source_keypoints)} keypoints (display subsampled)`} />}
            {art('keypoints_reference') && <ImagePanel title="Keypoints - reference" src={art('keypoints_reference')} caption={`${int(m.reference_keypoints)} keypoints (display subsampled)`} />}
            <ImagePanel title="Raw correspondences" src={art('raw_matches')} caption="Candidates before geometric verification" />
            <ImagePanel title="Verified inliers" src={art('inlier_matches')} caption="Tie points accepted by geometric verification" />
            <ImagePanel title="Spatial coverage" src={art('spatial_coverage')} caption="Green: occupied cell · red: empty cell · grey: invalid or outside the overlap" />
            <ImagePanel title="Selected tie points" src={art('tie_points')} caption="Green: selected · red: other verified inliers · grey: grid · cyan: valid overlap boundary" missing="Not produced: no accepted transformation" />
            <ImagePanel title="Registered image" src={art('registered_source')} caption="Source warped into the reference pixel grid" missing="Not produced: no accepted transformation" />
            <ImagePanel title="Overlay" src={art('overlay')} caption="Reference = red, registered source = cyan; aligned terrain looks grey. Colour fringes on crater walls also come from different Sun angles." missing="Not produced: no accepted transformation" />
            <ImagePanel title="Checkerboard" src={art('checkerboard')} caption="Alternating squares of reference and registered source; terrain should continue across the edges" missing="Not produced: no accepted transformation" />
          </div>
          {art('reference_crop') && art('registered_crop') && (
            <>
              <SectionTitle>Alpha overlay</SectionTitle>
              <AlphaOverlay reference={art('reference_crop')} registered={art('registered_crop')} />
            </>
          )}

          {r.matcher_results?.length > 0 && (
            <>
              <SectionTitle hint="Measured under the same protocol">Matchers</SectionTitle>
              <Matchers r={r} jobId={jobId} art={art} />
            </>
          )}

          <SectionTitle>Transformation</SectionTitle>
          {t ? (
            <div className="grid gap-4 lg:grid-cols-2">
              <Card>
                <div className="mb-2 text-sm text-ink-2">
                  {t.model} · {t.method} · maps source pixel → reference pixel
                  {t.mirrored_source ? ' · includes a left-right mirror of the source' : ''}
                  {t.strip_model ? ' · the matrix is only the best affine approximation of a smooth along-track model' : ''}
                </div>
                <Matrix m={t.matrix} />
                {art('registered_geotiff') && (
                  <a className="mt-3 inline-block text-sm text-accent hover:underline" href={art('registered_geotiff')}>
                    Download registered_source.tif (original values, reference grid)
                  </a>
                )}
              </Card>
              <Card>
                <KeyValue
                  rows={[
                    ['Scale (min / max)', `${num(t.properties.scale_minor, 4)} / ${num(t.properties.scale_major, 4)}`],
                    ['Rotation', deg(t.properties.rotation_deg)],
                    ['Shear', deg(t.properties.shear_deg)],
                    ['Anisotropy', num(t.properties.anisotropy, 3)],
                    ['Perspective ratio', num(t.properties.perspective_ratio, 4)],
                    ['Footprint inside reference', pct(t.properties.fraction_inside_reference)],
                    ['Seed', `${t.seed_inliers} distinct inliers from the ${t.seed_pool} best-ranked matches`],
                    ['Selected representation', r.selected_preprocessing],
                    ...(t.strip_check
                      ? [
                          ['Local scale along the strip', `${num(t.strip_check.scale_min, 3)} – ${num(t.strip_check.scale_max, 3)}`],
                          ['Largest local anisotropy', num(t.strip_check.max_anisotropy, 3)],
                          ['Largest deviation from the affine', withUnit(t.strip_check.max_deviation_from_affine_px, 'px', 1)],
                          ['Tie-point support along the strip', pct(m.supported_along_track_fraction, 0)],
                        ]
                      : []),
                    ...(t.refit_on_refined_points ? [['Refitted on refined points', 'yes, the held-out comparison supported it']] : []),
                  ]}
                />
              </Card>
            </div>
          ) : (
            <Card>
              <p className="text-sm text-ink-2">No transformation was accepted, so none is reported.</p>
            </Card>
          )}

          {(tp || rf) && (
            <>
              <SectionTitle hint="Selection only removes verified points; empty cells are never filled">Tie points and sub-pixel refinement</SectionTitle>
              <div className="grid gap-4 lg:grid-cols-2">
                {tp && (
                  <Card>
                    <KeyValue
                      rows={[
                        ['Selected / verified', `${int(tp.selected)} of ${int(tp.verified_inliers)}`],
                        ['Coverage, inliers', `source ${pct(tp.inlier_source_coverage)} · overlap ${pct(tp.inlier_overlap_coverage)}`],
                        ['Coverage, selected', `source ${pct(tp.source_coverage)} · overlap ${pct(tp.overlap_coverage)}`],
                        ['Occupied grid cells', `${int(tp.occupied_cells)} of ${int(tp.valid_cells)} valid`],
                        ['Tie-point density', tp.tie_points_per_km2 == null ? NA : `${num(tp.tie_points_per_km2, 2)} per km²`],
                        ['Distribution score', num(tp.distribution_score, 3)],
                        ['Hull share of overlap', pct(tp.hull_fraction_of_overlap)],
                      ]}
                    />
                    <p className="mt-2 text-xs text-muted">{tp.distribution_definition}</p>
                    {tp.below_minimum && <p className="mt-2 text-xs text-warning">! Fewer than the requested minimum of {tp.minimum_requested}. No points were invented.</p>}
                  </Card>
                )}
                {rf && (
                  <Card>
                    <KeyValue
                      rows={[
                        ['Strategy', `${rf.strategy} on ${rf.representation}`],
                        ['Refined / rejected', `${int(rf.refined)} / ${int(rf.rejected)}`],
                        ['Displacement median / p90', `${num(rf.median_displacement_px, 3)} / ${num(rf.p90_displacement_px, 3)} px`],
                        ['Median correlation', num(rf.median_score, 3)],
                        ...(rv?.status === 'ok'
                          ? [
                              ['Held-out median, initial → refined', `${num(rv.initial_median_px, 3)} → ${num(rv.refined_median_px, 3)} px`],
                              ['Held-out RMSE, initial → refined', `${num(rv.initial_rmse_px, 3)} → ${num(rv.refined_rmse_px, 3)} px`],
                              ['Improved held-out agreement', rv.improved ? 'yes' : 'no'],
                              ...(rv.local?.status === 'ok'
                                ? [['Local affine per block, held-out median', `${num(rv.local.initial_median_px, 3)} → ${num(rv.local.refined_median_px, 3)} px (${rv.local.blocks} blocks)`]]
                                : []),
                            ]
                          : []),
                      ]}
                    />
                    {rf.systematic_shift && (
                      <p className="mt-2 text-xs">
                        <span className="font-semibold text-warning">! </span>
                        The points moved together ({num(rf.median_shift_x_px)}, {num(rf.median_shift_y_px)} px), so the correlation may be following shading rather than terrain. These refined coordinates should not be trusted.
                      </p>
                    )}
                    {Object.keys(rf.rejected_by_reason || {}).length > 0 && (
                      <ul className="mt-2 list-disc pl-5 text-xs text-muted">
                        {Object.entries(rf.rejected_by_reason).map(([k, v]) => (
                          <li key={k}>
                            {k}: {v}
                          </li>
                        ))}
                      </ul>
                    )}
                    <p className="mt-2 text-xs text-muted">{rf.note}</p>
                  </Card>
                )}
              </div>
            </>
          )}

          {r.decoys?.length > 0 && (
            <>
              <SectionTitle hint="The same pipeline on ground known to be wrong">Null control</SectionTitle>
              <div className="overflow-x-auto rounded-xl border border-white/10 bg-surface">
                <table className="w-full text-sm">
                  <thead className="border-b border-white/10 text-left text-xs text-ink-2">
                    <tr>
                      <th className="px-3 py-2 font-medium">Kind</th>
                      <th className="px-3 py-2 text-right font-medium">Tiles used</th>
                      <th className="px-3 py-2 text-right font-medium">Proposed</th>
                      <th className="px-3 py-2 text-right font-medium">Accepted</th>
                      <th className="px-3 py-2 text-right font-medium">Inliers</th>
                      <th className="px-3 py-2 font-medium">Geometry</th>
                    </tr>
                  </thead>
                  <tbody className="tabular">
                    {r.decoys.map((d, i) => (
                      <tr key={i} className="border-b border-white/5 last:border-0">
                        <td className="px-3 py-2">{d.kind.replace('_', ' ')}</td>
                        <td className="px-3 py-2 text-right">{int(d.tiles_used)}</td>
                        <td className="px-3 py-2 text-right">{int(d.candidates_before_appearance)}</td>
                        <td className="px-3 py-2 text-right">{int(d.candidates)}</td>
                        <td className="px-3 py-2 text-right">{int(d.inliers)}</td>
                        <td className="px-3 py-2 text-ink-2">{d.model ? `${d.model}${d.valid ? ', passes plausibility' : ', rejected as implausible'}` : 'no consensus'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}

          {mc && r.status !== 'FAILED' && (
            <>
              <SectionTitle hint="Metadata is system-level pointing knowledge, not ground truth">Consistency with XML/GCP metadata</SectionTitle>
              <Card>
                <KeyValue
                  rows={[
                    ['Footprint centre offset', `${num(mc.centre_offset_px, 0)} px${mc.centre_offset_m != null ? ` (${num(mc.centre_offset_m / 1000, 2)} km)` : ''}`],
                    ['Rotation difference', deg(mc.rotation_difference_deg)],
                    ['Scale ratio', num(mc.scale_ratio, 3)],
                    ['Mirror agrees with metadata', mc.mirror_agrees ? 'yes' : 'no'],
                    ['Array orientation agrees', mc.orientation_agrees ? 'yes' : 'no'],
                  ]}
                />
              </Card>
            </>
          )}

          <SectionTitle hint="Every representation and orientation that was tried">Attempts</SectionTitle>
          <Attempts attempts={r.attempts} />

          {cov && (
            <>
              <SectionTitle>Spatial coverage grid</SectionTitle>
              <Card>
                <KeyValue
                  rows={[
                    ['Grid', `${cov.grid_rows} × ${cov.grid_cols}`],
                    ['Valid cells', cov.valid_cells],
                    ['Cells inside overlap', cov.overlap_cells ?? NA],
                    ['Occupied cells', cov.occupied_cells],
                    ['Image-space Spatial Coverage', pct(cov.coverage)],
                    ['Coverage within overlap', pct(cov.coverage_within_overlap)],
                  ]}
                />
              </Card>
            </>
          )}

          {r.transformation && (
            <>
              <SectionTitle hint="Used for evaluation only, never in a fit">Independent control points</SectionTitle>
              <ControlPoints jobId={jobId} result={r} overviewSource={art('source')} overviewReference={art('reference')} onDone={() => setRefresh((n) => n + 1)} />
            </>
          )}

          {il && (
            <>
              <SectionTitle hint="Describes the scene; no physical correction is applied">Illumination</SectionTitle>
              <Card>
                <KeyValue
                  rows={[
                    ['Sun azimuth / elevation (XML, source only)', `${deg(il.sun_azimuth_deg)} / ${deg(il.sun_elevation_deg)}`],
                    ['Regime', il.regime ?? NA],
                    ['Shadow fraction, source / reference (image-based)', `${pct(il.source_shadow_fraction)} / ${pct(il.reference_shadow_fraction)}`],
                    ['Shading axis, source / reference (image-based)', `${deg(il.source_shading_axis_deg, 1)} / ${deg(il.reference_shading_axis_deg, 1)}`],
                    ['Angle between shading axes, reference frame', deg(il.shading_axis_difference_deg, 1)],
                  ]}
                />
                <ul className="mt-3 list-disc space-y-1 pl-5 text-xs text-muted">
                  {il.notes.map((n) => (
                    <li key={n}>{n}</li>
                  ))}
                </ul>
              </Card>
            </>
          )}

          {ov && (
            <>
              <SectionTitle hint="A prior for the search, not a result">Metadata-guided search</SectionTitle>
              <Card>
                <KeyValue
                  rows={[
                    ['Search mode', ov.mode === 'metadata' ? 'around the XML/GCP footprint' : 'whole reference'],
                    ['Why', ov.reason],
                    ['Reference body (from its CRS)', ov.coordinate_system?.body ?? NA],
                    ['CRS kind / projection / units', ov.coordinate_system ? `${ov.coordinate_system.crs_kind} / ${ov.coordinate_system.projection ?? NA} / ${ov.coordinate_system.units ?? NA}` : NA],
                    ['Search radius', withUnit(ov.search_radius_px, 'px', 0)],
                    ['Prior footprint inside reference', pct(ov.footprint_fraction_in_reference)],
                    ['Prior footprint on valid pixels', pct(ov.footprint_fraction_on_valid_reference)],
                    ['Location found by', r.search_trace?.anchor?.method ?? NA],
                    ['Orientation found', r.search_trace?.anchor?.hypothesis ?? NA],
                    [
                      'Offset of the result from the prior',
                      r.search_trace?.anchor
                        ? `${r.search_trace.anchor.offset_from_prior_px.map((v) => num(v, 0)).join(', ')} px${r.search_trace.anchor.offset_from_prior_m != null ? ` (${num(r.search_trace.anchor.offset_from_prior_m / 1000, 2)} km)` : ''}`
                        : NA,
                    ],
                    ...ov.warnings.map((w, i) => [`Note ${i + 1}`, w]),
                  ]}
                />
              </Card>
            </>
          )}

          <SectionTitle>Downloads</SectionTitle>
          <Downloads jobId={jobId} art={art} />

          <SectionTitle>Notes</SectionTitle>
          <Card>
            <ul className="list-disc space-y-1 pl-5 text-sm text-ink-2">
              {r.notes.map((n) => (
                <li key={n}>{n}</li>
              ))}
            </ul>
          </Card>

          {r.metadata && (
            <>
              <SectionTitle>Scene metadata</SectionTitle>
              <Card>
                <KeyValue
                  rows={[
                    ['Native resolution (XML)', withUnit(r.metadata.resolution_m, 'm')],
                    ['Sun azimuth / elevation', `${deg(r.metadata.sun_azimuth_deg)} / ${deg(r.metadata.sun_elevation_deg)}`],
                    ['Projection / area', `${r.metadata.projection ?? NA} / ${r.metadata.area ?? NA}`],
                    ['Source', `${r.source.width} × ${r.source.height} px · ${r.source.dtype}`],
                    ['Reference', `${r.reference.width} × ${r.reference.height} px · ${r.reference.dtype}`],
                  ]}
                />
              </Card>
            </>
          )}
        </>
      )}
    </div>
  )
}
