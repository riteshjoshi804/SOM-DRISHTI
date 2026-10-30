import { useEffect, useState } from 'react'
import { api } from '../api.js'
import { shortId, when } from '../format.js'
import { Card, ErrorNote, Loading, StatusBadge } from '../components/ui.jsx'

export default function Results() {
  const [jobs, setJobs] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let timer
    const load = () =>
      api
        .results()
        .then((list) => {
          setJobs(list)
          // Keep polling while anything is still queued or running.
          if (list.some((j) => j.state === 'queued' || j.state === 'running')) timer = setTimeout(load, 2000)
        })
        .catch(setError)
    load()
    return () => clearTimeout(timer)
  }, [])

  return (
    <div>
      <h1 className="text-2xl font-bold tracking-tight">Registration Results</h1>
      <p className="mt-1 text-sm text-ink-2">
        Runs started here or from the command-line scripts. Stored under results/jobs/.
      </p>
      <ErrorNote error={error} />
      {!jobs && !error && <Loading />}
      {jobs?.length === 0 && (
        <Card className="mt-6">
          <p className="text-sm text-ink-2">
            No runs yet. Start one from <a className="text-accent hover:underline" href="#/scenes">Dataset / Scenes</a>.
          </p>
        </Card>
      )}
      {jobs?.length > 0 && (
        <div className="mt-6 overflow-x-auto rounded-xl border border-white/10 bg-surface">
          <table className="w-full text-sm">
            <thead className="border-b border-white/10 text-left text-xs text-ink-2">
              <tr>
                <th className="px-4 py-2 font-medium">Run</th>
                <th className="px-4 py-2 font-medium">Scene</th>
                <th className="px-4 py-2 font-medium">State</th>
                <th className="px-4 py-2 font-medium">Registration status</th>
                <th className="px-4 py-2 font-medium">Finished</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((j) => (
                <tr key={j.job_id} className="border-b border-white/5 last:border-0 hover:bg-white/5">
                  <td className="px-4 py-2">
                    <a className="font-mono text-xs text-accent hover:underline" href={`#/results/${encodeURIComponent(j.job_id)}`}>
                      {j.job_id.split('__').pop()}
                    </a>
                  </td>
                  <td className="px-4 py-2">{j.scene_id ? shortId(j.scene_id) : j.label}</td>
                  <td className="px-4 py-2 text-ink-2">
                    {j.state === 'running' ? `running ${Math.round(j.progress * 100)}% · ${j.message}` : j.state}
                  </td>
                  <td className="px-4 py-2">
                    {j.status ? <StatusBadge status={j.status} /> : <StatusBadge status={j.state} />}
                  </td>
                  <td className="px-4 py-2 text-ink-2">{when(j.finished_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
