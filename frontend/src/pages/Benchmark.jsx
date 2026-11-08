import { useEffect, useMemo, useState } from 'react'
import { api } from '../api.js'
import { NA, int, num, pct, shortId, when } from '../format.js'
import GroupedBars from '../components/GroupedBars.jsx'
import { Card, ErrorNote, Loading, SectionTitle, StatusBadge } from '../components/ui.jsx'

// One colour per matcher, whichever matchers a run contains. Validated together for this surface.
const MATCHER_COLOURS = {
  SIFT: 'var(--color-series-1)',
  SUPERPOINT_LIGHTGLUE: 'var(--color-series-2)',
  LOFTR: 'var(--color-series-3)',
  RIFT: 'var(--color-series-4)',
}
const MATCHER_LABELS = {
  SIFT: 'SIFT / RootSIFT',
  SUPERPOINT_LIGHTGLUE: 'SuperPoint + LightGlue',
  LOFTR: 'LoFTR',
  RIFT: 'RIFT (experimental)',
}

function why(row) {
  if (!row) return 'not run'
  if (row.registration_status === 'UNAVAILABLE') return 'unavailable'
  if (row.registration_status === 'FAILED') return 'failed'
  return 'not measured'
}

function RowsTable({ rows }) {
  const head = 'px-3 py-2 font-medium'
  const right = `${head} text-right`
  return (
    <div className="overflow-x-auto rounded-xl border border-white/10 bg-surface">
      <table className="w-full text-sm">
        <thead className="border-b border-white/10 text-left text-xs text-ink-2">
          <tr>
            <th className={head}>Scene</th>
            <th className={head}>Matcher</th>
            <th className={head}>Status</th>
            <th className={head}>Model</th>
            <th className={right}>Candidates</th>
            <th className={right}>Accepted</th>
            <th className={right}>Inliers</th>
            <th className={right}>Inlier ratio</th>
            <th className={right}>RMSE (px)</th>
            <th className={right}>Held-out RMSE / median / p90 (px)</th>
            <th className={right}>Unseen blocks RMSE (px)</th>
            <th className={right}>Coverage, source / overlap</th>
            <th className={right}>Strip support</th>
            <th className={right}>Decoy max</th>
            <th className={right}>Time (s)</th>
            <th className={head}>Device</th>
          </tr>
        </thead>
        <tbody className="tabular">
          {rows.map((r) => (
            <tr key={`${r.scene_id}-${r.matcher}`} className="border-b border-white/5 align-top last:border-0">
              <td className="px-3 py-2 whitespace-nowrap">
                {r.job_id ? (
                  <a className="text-accent hover:underline" href={`#/results/${encodeURIComponent(r.job_id)}`}>
                    {shortId(r.scene_id)}
                  </a>
                ) : (
                  shortId(r.scene_id)
                )}
              </td>
              <td className="px-3 py-2 whitespace-nowrap">
                {MATCHER_LABELS[r.matcher] || r.matcher}
                {r.selected_by_auto ? <span className="ml-1 text-xs text-ink-2">(AUTO choice)</span> : null}
              </td>
              <td className="px-3 py-2">
                <StatusBadge status={r.registration_status} />
                {r.reasons && <div className="mt-1 max-w-[20rem] text-xs text-ink-2">{r.reasons}</div>}
              </td>
              <td className="px-3 py-2">{r.transformation_model || NA}</td>
              <td className="px-3 py-2 text-right">{int(r.candidate_correspondences)}</td>
              <td className="px-3 py-2 text-right">{int(r.accepted_correspondences)}</td>
              <td className="px-3 py-2 text-right">{int(r.geometric_inliers)}</td>
              <td className="px-3 py-2 text-right">{pct(r.inlier_ratio, 1)}</td>
              <td className="px-3 py-2 text-right">{num(r.reprojection_rmse_px)}</td>
              <td className="px-3 py-2 text-right whitespace-nowrap">
                {num(r.heldout_rmse_px)} / {num(r.heldout_median_px)} / {num(r.heldout_p90_px)}
              </td>
              <td className="px-3 py-2 text-right">{num(r.block_heldout_rmse_px)}</td>
              <td className="px-3 py-2 text-right whitespace-nowrap">
                {pct(r.source_spatial_coverage)} / {pct(r.overlap_spatial_coverage)}
              </td>
              <td className="px-3 py-2 text-right">{pct(r.along_track_support, 0)}</td>
              <td className="px-3 py-2 text-right">{int(r.decoy_max_inliers)}</td>
              <td className="px-3 py-2 text-right">{num(r.processing_seconds, 0)}</td>
              <td className="px-3 py-2 text-ink-2">{r.device || NA}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function SyntheticTable({ data }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-white/10 bg-surface">
      <table className="w-full text-sm">
        <thead className="border-b border-white/10 text-left text-xs text-ink-2">
          <tr>
            <th className="px-3 py-2 font-medium">Case</th>
            <th className="px-3 py-2 font-medium">Matcher</th>
            <th className="px-3 py-2 font-medium">Status</th>
            <th className="px-3 py-2 text-right font-medium">Inliers</th>
            <th className="px-3 py-2 text-right font-medium">Error vs known truth, RMSE / max (px)</th>
            <th className="px-3 py-2 text-right font-medium">Tie-point error, initial → refined, median (px)</th>
            <th className="px-3 py-2 text-right font-medium">Refined points</th>
          </tr>
        </thead>
        <tbody className="tabular">
          {data.rows.map((r) => (
            <tr key={`${r.case}-${r.matcher}`} className="border-b border-white/5 last:border-0">
              <td className="px-3 py-2">{r.case}</td>
              <td className="px-3 py-2">{MATCHER_LABELS[r.matcher] || r.matcher}</td>
              <td className="px-3 py-2">
                <StatusBadge status={r.status} />
              </td>
              <td className="px-3 py-2 text-right">{int(r.inliers)}</td>
              <td className="px-3 py-2 text-right">
                {num(r.truth_rmse_px)} / {num(r.truth_max_px)}
              </td>
              <td className="px-3 py-2 text-right">
                {num(r.tie_error_initial_median_px, 3)} → {num(r.tie_error_refined_median_px, 3)}
              </td>
              <td className="px-3 py-2 text-right">{int(r.refined_points)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function RepresentationTable({ data }) {
  const modes = data.modes || []
  return (
    <div className="overflow-x-auto rounded-xl border border-white/10 bg-surface">
      <table className="w-full text-sm">
        <thead className="border-b border-white/10 text-left text-xs text-ink-2">
          <tr>
            <th className="px-3 py-2 font-medium">Scene</th>
            <th className="px-3 py-2 font-medium">Matcher</th>
            {modes.map((m) => (
              <th key={m} className="px-3 py-2 text-right font-medium">
                {m}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="tabular">
          {data.table.map((row) => (
            <tr key={`${row.scene_id}-${row.matcher}`} className="border-b border-white/5 last:border-0">
              <td className="px-3 py-2 whitespace-nowrap">{shortId(row.scene_id)}</td>
              <td className="px-3 py-2 whitespace-nowrap">{MATCHER_LABELS[row.matcher] || row.matcher}</td>
              {modes.map((m) => (
                <td key={m} className="px-3 py-2 text-right">
                  {row.values[m] === null || row.values[m] === undefined ? NA : int(row.values[m])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function Benchmark() {
  const [data, setData] = useState(null)
  const [status, setStatus] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.benchmarks().then(setData).catch(setError)
    api.matchers().then(setStatus).catch(() => setStatus([]))
  }, [])

  const bench = data?.matchers
  const rows = useMemo(() => bench?.rows || [], [bench])
  const scenes = useMemo(() => [...new Set(rows.map((r) => r.scene_id))].map((id) => ({ key: id, label: shortId(id).replace(/_V.*/, '') })), [rows])
  const series = useMemo(
    () => [...new Set(rows.map((r) => r.matcher))].map((m) => ({ key: m, label: MATCHER_LABELS[m] || m, colour: MATCHER_COLOURS[m] || 'var(--color-muted)' })),
    [rows],
  )
  const row = (scene, matcher) => rows.find((r) => r.scene_id === scene && r.matcher === matcher)
  // Only accepted results are charted; a failed branch gets its status in words instead of a number.
  const measured = (field) => (scene, matcher) => {
    const r = row(scene, matcher)
    return r && r.registration_status !== 'FAILED' && r.registration_status !== 'UNAVAILABLE' ? r[field] : null
  }
  const reason = (scene, matcher) => why(row(scene, matcher))

  return (
    <div>
      <h1 className="text-2xl font-bold tracking-tight">Matcher Benchmark</h1>
      <p className="mt-1 max-w-3xl text-sm text-ink-2">
        Every value comes from a stored run on the official sample, using the same protocol for every matcher. Reprojection and held-out errors are
        measured on automatic matches. They are not independent ground-truth accuracy.
      </p>
      <ErrorNote error={error} />
      {!data && !error && <Loading />}

      {status && (
        <>
          <SectionTitle hint="Checked now, on this machine">Matcher availability</SectionTitle>
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
            {status.map((m) => {
              const ok = m.available || m.needs_download
              return (
                <Card key={m.name}>
                  <div className="flex items-start justify-between gap-2">
                    <div className="font-semibold">{m.label}</div>
                    <span className={`whitespace-nowrap rounded-full border px-2 py-0.5 text-xs ${ok ? 'border-white/15 text-ink-2' : 'border-warning/50 text-warning'}`}>
                      {ok ? (m.needs_download ? 'Weights on first use' : 'Available') : 'Unavailable'}
                    </span>
                  </div>
                  <div className="mt-2 text-xs text-muted">Device: {m.device || NA}</div>
                  {m.reason && <div className="mt-1 text-xs text-warning">{m.reason}</div>}
                  {m.weights?.map((w) => (
                    <div key={w.name} className="mt-1 text-xs text-muted">
                      {w.name}: {w.license}
                    </div>
                  ))}
                </Card>
              )
            })}
          </div>
        </>
      )}

      {data && !bench && (
        <Card className="mt-6">
          <p className="text-sm text-ink-2">
            No benchmark has been run yet, so there is nothing to show. Run <code className="text-ink">scripts\benchmark_matchers.py</code>.
          </p>
        </Card>
      )}

      {bench && (
        <>
          <SectionTitle hint={`Generated ${when(bench.generated_at)} · ${bench.environment?.device || ''}`}>Comparison on the official scenes</SectionTitle>
          <div className="grid gap-4 xl:grid-cols-2">
            <GroupedBars title="Inlier ratio" subtitle="Geometrically verified share of accepted matches" categories={scenes} series={series} value={measured('inlier_ratio')} format={(v) => `${(v * 100).toFixed(v < 0.1 && v > 0 ? 1 : 0)}%`} reason={reason} />
            <GroupedBars title="Held-out error" subtitle="Median, px; matches left out of the fit. Lower is better" categories={scenes} series={series} value={measured('heldout_median_px')} format={(v) => `${v.toFixed(2)} px`} reason={reason} />
            <GroupedBars title="Spatial coverage" subtitle="Share of valid 8 × 8 grid cells over the overlap that hold an inlier" categories={scenes} series={series} value={measured('overlap_spatial_coverage')} format={(v) => `${(v * 100).toFixed(0)}%`} reason={reason} />
            <GroupedBars
              title="Processing time"
              subtitle={`Seconds per scene on ${bench.environment?.device || 'this machine'}, decoy controls included`}
              categories={scenes}
              series={series}
              value={(scene, matcher) => row(scene, matcher)?.processing_seconds ?? null}
              format={(v) => `${v.toFixed(0)} s`}
              reason={reason}
            />
          </div>

          <SectionTitle hint="All measured values">Results table</SectionTitle>
          <RowsTable rows={rows} />
          <p className="mt-2 text-xs text-muted">
            Decoy max: the most inliers the same pipeline found on ground known to be wrong. A result has to stand clearly above it. Unseen blocks: stretches of
            the image left out of the fit in turn, the harder test for a long strip. Strip support: share of the 20 stretches along the strip holding at least 3
            inliers; under 50% a result cannot be SUCCESS.
          </p>
          {data.notes?.notes?.map((n) => (
            <p key={n} className="mt-2 text-xs text-muted">
              {n}
            </p>
          ))}
        </>
      )}

      {data?.representations && (
        <>
          <SectionTitle hint={data.representations.label}>Image representations</SectionTitle>
          <RepresentationTable data={data.representations} />
          {data.representations.notes?.map((n) => (
            <p key={n} className="mt-2 text-xs text-muted">
              {n}
            </p>
          ))}
        </>
      )}

      {data?.synthetic && (
        <>
          <SectionTitle hint="Rendered terrain with a known transformation: not lunar data">Synthetic ground truth</SectionTitle>
          <div className="mb-3 rounded-lg border border-warning/40 bg-warning/5 px-4 py-2 text-xs text-ink-2">
            <span className="font-semibold text-warning">! SYNTHETIC. </span>
            {data.synthetic.label}. These figures show that the machinery recovers known geometry. They do not predict accuracy on real lunar scenes.
          </div>
          <SyntheticTable data={data.synthetic} />
        </>
      )}
    </div>
  )
}
