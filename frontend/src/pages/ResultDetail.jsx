import { useEffect, useState } from 'react'
import { api } from '../api.js'
import { NA, deg, int, num, pct, withUnit } from '../format.js'
import { Card, ErrorNote, ImagePanel, KeyValue, Loading, MetricCard, SectionTitle, StatusBadge } from '../components/ui.jsx'

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
        A scene takes roughly 20–60 s. Scenes where nothing matches take longest, because every representation and
        the mirrored source are tried before FAILED is reported.
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
  return (
    <div className="overflow-x-auto rounded-xl border border-white/10 bg-surface">
      <table className="w-full text-sm">
        <thead className="border-b border-white/10 text-left text-xs text-ink-2">
          <tr>
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

export default function ResultDetail({ jobId }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

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
  }, [jobId])

  const job = data?.job
  const r = data?.result
  const m = r?.metrics || {}
  const art = (key) => (r?.artifacts?.[key] ? api.artifactUrl(jobId, r.artifacts[key]) : null)
  const cov = r?.spatial_coverage
  const t = r?.transformation
  const mc = r?.metadata_consistency

  return (
    <div>
      <a href="#/results" className="text-sm text-accent hover:underline">
        ← All results
      </a>
      <div className="mt-2 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="text-xs uppercase tracking-wider text-ink-2">
            {r?.sensor || job?.label || 'Registration'} {r?.reference_sensor ? `→ ${r.reference_sensor}` : ''}
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
          {(r.failure_reasons.length > 0 || r.warnings.length > 0) && (
            <Card className={`mt-6 ${r.status === 'FAILED' ? 'border-critical/50' : 'border-warning/40'}`}>
              {r.failure_reasons.map((x) => (
                <p key={x} className="text-sm">
                  <span className="font-semibold text-critical">✕ </span>
                  {x}
                </p>
              ))}
              {r.warnings.map((x) => (
                <p key={x} className="text-sm">
                  <span className="font-semibold text-warning">! </span>
                  {x}
                </p>
              ))}
            </Card>
          )}

          <SectionTitle hint="All values measured in this run">Metrics</SectionTitle>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <MetricCard label="Candidate Matches" value={int(m.ratio_filtered_matches)} hint={`Ratio-test survivors of ${int(m.raw_matches)} nearest-neighbour candidates`} />
            <MetricCard label="Inliers" value={int(m.inliers)} hint="Spatially distinct tie points consistent with the transformation" />
            <MetricCard label="Inlier Ratio" value={pct(m.inlier_ratio, 2)} hint="Inliers / candidate matches" />
            <MetricCard label="Reprojection RMSE" value={withUnit(m.reprojection_rmse_px, 'px')} hint="On the fitted matches - not ground-truth accuracy" />
            <MetricCard label="Median Error" value={withUnit(m.median_reprojection_error_px, 'px')} hint="Median inlier reprojection error" />
            <MetricCard
              label="Spatial Coverage"
              value={pct(m.image_space_spatial_coverage)}
              hint={`Image-space: ${int(m.occupied_cells)} of ${int(m.valid_cells)} valid source grid cells hold a tie point`}
            />
            <MetricCard label="Processing Time" value={withUnit(m.processing_time_seconds, 's', 1)} />
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
              value={`${int(m.source_keypoints)} / ${int(m.reference_keypoints)}`}
              hint={r.selected_preprocessing?.startsWith('pooled') ? 'Summed over the pooled representations' : `In the ${r.selected_preprocessing} representation`}
            />
            <MetricCard label="Cross-validated RMSE" value={withUnit(m.cross_validated_rmse_px, 'px')} hint="5-fold: errors on inliers held out of the fit" />
            <MetricCard label="Expected chance inliers" value={num(m.expected_chance_inliers, 4)} hint="Wrong matches expected to agree with this model by coincidence" />
            <MetricCard
              label="Coverage within overlap"
              value={pct(m.coverage_within_overlap)}
              hint={`Estimated overlap: ${pct(m.estimated_overlap_fraction)} of valid source pixels`}
            />
          </div>

          <SectionTitle>Images</SectionTitle>
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            <ImagePanel title="Source" src={art('source')} caption="Percentile-stretched source (as delivered)" />
            <ImagePanel title="Reference" src={art('reference')} caption="Percentile-stretched reference" />
            <ImagePanel title="Reference footprint" src={art('reference_footprint')} caption="Green: registered source footprint. Yellow: footprint predicted from XML/GCP metadata." />
            <ImagePanel title="Keypoints - source" src={art('keypoints_source')} caption={`${int(m.source_keypoints)} SIFT keypoints (display subsampled)`} />
            <ImagePanel title="Keypoints - reference" src={art('keypoints_reference')} caption={`${int(m.reference_keypoints)} SIFT keypoints (display subsampled)`} />
            <ImagePanel title="Matches" src={art('raw_matches')} caption="Ratio-filtered candidates (most are wrong - that is expected)" />
            <ImagePanel title="Inliers" src={art('inlier_matches')} caption="Tie points accepted by geometric verification" />
            <ImagePanel title="Spatial coverage" src={art('spatial_coverage')} caption="Green: occupied cell · red: empty cell · grey: invalid or outside the overlap" />
            <ImagePanel title="Registered image" src={art('registered_source')} caption="Source warped into the reference pixel grid" missing="Not produced: no accepted transformation" />
            <ImagePanel title="Overlay" src={art('overlay')} caption="Reference = red, registered source = cyan; aligned terrain looks grey. Colour fringes on crater walls also come from different Sun angles." missing="Not produced: no accepted transformation" />
          </div>

          <SectionTitle>Transformation</SectionTitle>
          {t ? (
            <div className="grid gap-4 lg:grid-cols-2">
              <Card>
                <div className="mb-2 text-sm text-ink-2">
                  {t.model} · {t.method} · maps source pixel → reference pixel
                  {t.mirrored_source ? ' · includes a left-right mirror of the source' : ''}
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
                    ['Seed', `${t.seed_inliers} distinct inliers from the ${t.seed_pool} best-ratio matches`],
                    ['Selected representation', r.selected_preprocessing],
                  ]}
                />
              </Card>
            </div>
          ) : (
            <Card>
              <p className="text-sm text-ink-2">No transformation was accepted, so none is reported.</p>
            </Card>
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

          <SectionTitle>Notes</SectionTitle>
          <Card>
            <ul className="list-disc space-y-1 pl-5 text-sm text-ink-2">
              {r.notes.map((n) => (
                <li key={n}>{n}</li>
              ))}
            </ul>
            <a className="mt-3 inline-block text-sm text-accent hover:underline" href={api.artifactUrl(jobId, 'result.json')}>
              Download full result.json
            </a>
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
