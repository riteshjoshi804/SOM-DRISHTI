import { useEffect, useState } from 'react'
import { api } from '../api.js'
import { NA, int, num, pct, shortId, when } from '../format.js'
import { Card, ErrorNote, Loading, SectionTitle, StatusBadge } from '../components/ui.jsx'

function SensorCard({ name, reference, count, tested, note }) {
  return (
    <Card>
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-xs uppercase tracking-wider text-ink-2">Chandrayaan-2</div>
          <div className="text-2xl font-bold tracking-wide">{name}</div>
        </div>
        {tested ? (
          <span className="rounded-full border border-white/15 px-2.5 py-0.5 text-xs text-ink-2">vs {reference}</span>
        ) : (
          <span className="rounded-full border border-white/15 px-2.5 py-0.5 text-xs text-muted">Not tested</span>
        )}
      </div>
      <div className="mt-4 text-sm text-ink-2">{note}</div>
      {count !== null && <div className="mt-1 text-3xl font-semibold">{count}</div>}
    </Card>
  )
}

export default function Dashboard() {
  const [dataset, setDataset] = useState(null)
  const [health, setHealth] = useState(null)
  const [latest, setLatest] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.dataset().then(setDataset).catch(setError)
    api.health().then(setHealth).catch(() => setHealth(null))
    api
      .results()
      .then(async (jobs) => {
        // Latest finished run per scene, with its measured metrics.
        const seen = new Map()
        for (const job of jobs) if (job.scene_id && job.state === 'done' && !seen.has(job.scene_id)) seen.set(job.scene_id, job)
        const rows = await Promise.all(
          [...seen.values()].map(async (job) => ({ job, m: await api.metrics(job.job_id).catch(() => null) })),
        )
        rows.sort((a, b) => a.job.scene_id.localeCompare(b.job.scene_id))
        setLatest(rows)
      })
      .catch(() => setLatest([]))
  }, [])

  const counts = dataset?.sensor_counts || {}
  return (
    <div>
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">SOM DRISHTI</h1>
          <p className="mt-1 text-ink-2">
            Multi-modal, Sun-angle and scale invariant image correspondence for Chandrayaan-2 optical imagery ·
            SIH26166
          </p>
        </div>
        <div className="text-right text-xs text-muted">
          Backend:{' '}
          {health ? (
            <span className="text-ink-2">
              online · OpenCV {health.versions.opencv} · GDAL {health.versions.gdal}
            </span>
          ) : (
            <span className="text-critical">unreachable</span>
          )}
        </div>
      </div>

      <SectionTitle hint={dataset?.data_root}>Official SIH Sample Dataset</SectionTitle>
      <ErrorNote error={error} />
      {!dataset && !error && <Loading />}
      {dataset && (
        <div className="grid gap-4 md:grid-cols-3">
          <SensorCard
            name="OHRC"
            reference="LRO-NAC"
            tested={Boolean(counts.OHRC)}
            count={counts.OHRC ?? 0}
            note="official sample scenes (source + LRO-NAC reference + XML)"
          />
          <SensorCard
            name="IIRS"
            reference="LRO-WAC"
            tested={Boolean(counts.IIRS)}
            count={counts.IIRS ?? 0}
            note="official sample scenes (source + LRO-WAC reference + XML)"
          />
          <SensorCard
            name="TMC-2"
            tested={dataset.tmc2_present}
            count={dataset.tmc2_present ? counts['TMC-2'] : null}
            note={
              dataset.tmc2_present
                ? 'official sample scenes'
                : 'No official sample currently provided. Nothing about TMC-2 has been validated.'
            }
          />
        </div>
      )}
      {dataset?.incomplete?.length > 0 && (
        <p className="mt-3 text-sm text-warning">
          ! {dataset.incomplete.length} XML file(s) without a complete source/reference pair were skipped.
        </p>
      )}

      <SectionTitle hint="Most recent finished run of each scene; values are measured, never assumed">
        Latest registration per scene
      </SectionTitle>
      {latest === null && <Loading />}
      {latest?.length === 0 && (
        <Card>
          <p className="text-sm text-ink-2">
            No registration has been run yet. Open <a className="text-accent hover:underline" href="#/scenes">Dataset / Scenes</a>{' '}
            and press RUN REGISTRATION. Metrics show N/A until then.
          </p>
        </Card>
      )}
      {latest?.length > 0 && (
        <div className="overflow-x-auto rounded-xl border border-white/10 bg-surface">
          <table className="w-full text-sm">
            <thead className="border-b border-white/10 text-left text-xs text-ink-2">
              <tr>
                <th className="px-4 py-2 font-medium">Scene</th>
                <th className="px-4 py-2 font-medium">Status</th>
                <th className="px-4 py-2 font-medium">Matcher</th>
                <th className="px-4 py-2 text-right font-medium">Candidates</th>
                <th className="px-4 py-2 text-right font-medium">Inliers</th>
                <th className="px-4 py-2 text-right font-medium">Reproj. RMSE (px)</th>
                <th className="px-4 py-2 text-right font-medium">Coverage</th>
                <th className="px-4 py-2 font-medium">Finished</th>
              </tr>
            </thead>
            <tbody className="tabular">
              {latest.map(({ job, m }) => (
                <tr key={job.job_id} className="border-b border-white/5 last:border-0 hover:bg-white/5">
                  <td className="px-4 py-2">
                    <a className="text-accent hover:underline" href={`#/results/${encodeURIComponent(job.job_id)}`}>
                      {shortId(job.scene_id)}
                    </a>
                  </td>
                  <td className="px-4 py-2">
                    <StatusBadge status={m?.registration_status || job.status} />
                  </td>
                  <td className="px-4 py-2 text-ink-2">{m?.metrics?.matcher ? m.metrics.matcher.replace('_', ' + ') : 'SIFT'}</td>
                  <td className="px-4 py-2 text-right">{int(m?.metrics?.ratio_filtered_matches)}</td>
                  <td className="px-4 py-2 text-right">{int(m?.metrics?.inliers)}</td>
                  <td className="px-4 py-2 text-right">{num(m?.metrics?.reprojection_rmse_px)}</td>
                  <td className="px-4 py-2 text-right">{pct(m?.metrics?.image_space_spatial_coverage)}</td>
                  <td className="px-4 py-2 text-ink-2">{when(job.finished_at) || NA}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
