import { useEffect, useState } from 'react'
import { api } from '../api.js'
import { go } from '../nav.js'
import { NA, deg, withUnit } from '../format.js'
import { Button, Card, ErrorNote, KeyValue, Loading, SectionTitle } from '../components/ui.jsx'
import { RunOptions, runConfig, useRunOptions } from '../components/RunOptions.jsx'

export function useRunScene(options) {
  const [busy, setBusy] = useState(null)
  const [error, setError] = useState(null)
  const run = async (sceneId) => {
    setBusy(sceneId)
    setError(null)
    try {
      const job = await api.register(sceneId, runConfig(options))
      go(`/results/${encodeURIComponent(job.job_id)}`)
    } catch (e) {
      setError(e)
    } finally {
      setBusy(null)
    }
  }
  return { run, busy, error }
}

function SceneCard({ s, onRun, busy }) {
  return (
    <Card className="flex flex-col">
      <div className="mb-3 flex items-center justify-between">
        <span className="rounded-md bg-white/10 px-2 py-0.5 text-xs font-bold tracking-wider">{s.sensor}</span>
        <span className="text-xs text-muted">reference: {s.reference_sensor || NA}</span>
      </div>
      <div className="mb-4 break-all font-mono text-sm">{s.scene_id}</div>
      <KeyValue
        rows={[
          ['Source', s.source_file],
          ['Reference', s.reference_file],
          ['Source size', `${s.source_size[0]} × ${s.source_size[1]} px · ${s.source_dtype}`],
          ['Reference size', `${s.reference_size[0]} × ${s.reference_size[1]} px · ${s.reference_dtype}`],
          ['Native XML resolution', withUnit(s.native_resolution_m, 'm')],
          ['Sun azimuth', deg(s.sun_azimuth_deg)],
          ['Sun elevation', deg(s.sun_elevation_deg)],
          ['Date of pass (XML)', s.date_of_pass ?? NA],
        ]}
      />
      <div className="mt-5 flex gap-2 pt-1">
        <Button onClick={() => onRun(s.scene_id)} disabled={Boolean(busy)}>
          {busy === s.scene_id ? 'STARTING…' : 'RUN REGISTRATION'}
        </Button>
        <Button variant="secondary" onClick={() => go(`/scenes/${encodeURIComponent(s.scene_id)}`)}>
          Details
        </Button>
      </div>
    </Card>
  )
}

function UploadForm({ options }) {
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const submit = async (e) => {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      const form = new FormData(e.currentTarget)
      form.set('matcher', options.matcher)
      if (options.representation) form.set('representation', options.representation)
      const job = await api.upload(form)
      go(`/results/${encodeURIComponent(job.job_id)}`)
    } catch (err) {
      setError(err)
    } finally {
      setBusy(false)
    }
  }
  const field = 'block w-full text-sm text-ink-2 file:mr-3 file:rounded-md file:border-0 file:bg-white/10 file:px-3 file:py-1.5 file:text-ink'
  return (
    <Card>
      <form onSubmit={submit} className="grid gap-4 md:grid-cols-4 md:items-end">
        <label className="text-xs text-ink-2">
          Source TIFF
          <input required name="source" type="file" accept=".tif,.tiff" className={`mt-1 ${field}`} />
        </label>
        <label className="text-xs text-ink-2">
          Reference TIFF
          <input required name="reference" type="file" accept=".tif,.tiff" className={`mt-1 ${field}`} />
        </label>
        <label className="text-xs text-ink-2">
          XML metadata (optional)
          <input name="xml" type="file" accept=".xml" className={`mt-1 ${field}`} />
        </label>
        <Button type="submit" disabled={busy}>
          {busy ? 'UPLOADING…' : 'UPLOAD & REGISTER'}
        </Button>
      </form>
      <ErrorNote error={error} />
      <p className="mt-3 text-xs text-muted">
        Uploads are stored under results/uploads/. The official dataset is never modified. Without XML metadata a learned matcher has to search every
        orientation over the whole reference, which is slow and may run out of its search budget.
      </p>
    </Card>
  )
}

export default function Scenes() {
  const [scenes, setScenes] = useState(null)
  const [error, setError] = useState(null)
  const { options, update, matchers } = useRunOptions()
  const { run, busy, error: runError } = useRunScene(options)

  useEffect(() => {
    api.scenes().then(setScenes).catch(setError)
  }, [])

  const groups = scenes ? [...new Set(scenes.map((s) => s.sensor))] : []
  return (
    <div>
      <h1 className="text-2xl font-bold tracking-tight">Dataset / Scene Selection</h1>
      <p className="mt-1 text-sm text-ink-2">
        Scenes are discovered automatically by pairing each XML with its source and reference TIFF.
      </p>
      <ErrorNote error={error || runError} />
      <SectionTitle hint="Applies to every RUN button and to uploads">Registration options</SectionTitle>
      <Card>
        <RunOptions options={options} update={update} matchers={matchers} />
      </Card>
      {!scenes && !error && <Loading what="Reading dataset" />}
      {scenes?.length === 0 && <p className="mt-6 text-sm text-muted">No complete scenes were found.</p>}
      {groups.map((sensor) => (
        <div key={sensor}>
          <SectionTitle>{sensor}</SectionTitle>
          <div className="grid gap-4 lg:grid-cols-2">
            {scenes
              .filter((s) => s.sensor === sensor)
              .map((s) => (
                <SceneCard key={s.scene_id} s={s} onRun={run} busy={busy} />
              ))}
          </div>
        </div>
      ))}
      <SectionTitle hint="For your own source/reference pairs">Manual upload</SectionTitle>
      <UploadForm options={options} />
    </div>
  )
}
