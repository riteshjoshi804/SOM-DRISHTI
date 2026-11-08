import { useState } from 'react'
import { api } from '../api.js'
import { Button, Card, ErrorNote } from './ui.jsx'

const CHIP = 320
const ZOOM = 4

function invert2(m) {
  const det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
  return [
    [m[1][1] / det, -m[0][1] / det],
    [-m[1][0] / det, m[0][0] / det],
  ]
}

// One image: click the overview to choose where to look, then click the magnified chip to place the point.
function PickSide({ jobId, which, title, overview, width, height, rectifyWith, point, onPoint }) {
  const [centre, setCentre] = useState(null)
  const [rectify, setRectify] = useState(false)
  const useRectify = rectify && rectifyWith

  const fromOverview = (e) => {
    const box = e.currentTarget.getBoundingClientRect()
    setCentre({ x: ((e.clientX - box.left) / box.width) * width, y: ((e.clientY - box.top) / box.height) * height })
  }
  const fromChip = (e) => {
    const box = e.currentTarget.getBoundingClientRect()
    const scale = CHIP / box.width
    const u = (e.clientX - box.left) * scale - (CHIP - 1) / 2
    const v = (e.clientY - box.top) * scale - (CHIP - 1) / 2
    const m = useRectify ? invert2(rectifyWith.map((row) => row.slice(0, 2).map((value) => value * ZOOM))) : invert2([[ZOOM, 0], [0, ZOOM]])
    onPoint({ x: centre.x + m[0][0] * u + m[0][1] * v, y: centre.y + m[1][0] * u + m[1][1] * v })
  }
  const chipUrl = centre
    ? `/api/results/${encodeURIComponent(jobId)}/chip?which=${which}&x=${centre.x.toFixed(2)}&y=${centre.y.toFixed(2)}&size=${CHIP}&zoom=${ZOOM}&rectify=${useRectify ? 'true' : 'false'}`
    : null

  return (
    <div>
      <div className="flex items-center justify-between gap-2 text-xs font-semibold uppercase tracking-wider text-ink-2">
        {title}
        {rectifyWith && (
          <label className="flex items-center gap-1 font-normal normal-case tracking-normal text-muted">
            <input type="checkbox" checked={rectify} onChange={(e) => setRectify(e.target.checked)} />
            Show in the reference orientation
          </label>
        )}
      </div>
      <div className="mt-2 grid gap-2 sm:grid-cols-[minmax(0,1fr)_minmax(0,320px)]">
        <div className="flex max-h-[320px] items-center justify-center overflow-hidden rounded-lg bg-black/40">
          <img src={overview} alt={`${title} overview; click to choose where to look`} onClick={fromOverview} className="max-h-[320px] w-auto max-w-full cursor-crosshair" />
        </div>
        <div className="relative flex aspect-square items-center justify-center overflow-hidden rounded-lg bg-black/40 text-center text-xs text-muted">
          {chipUrl ? (
            <img src={chipUrl} alt={`${title} at ${ZOOM} times magnification; click the feature`} onClick={fromChip} className="h-full w-full cursor-crosshair" style={{ imageRendering: 'pixelated' }} />
          ) : (
            <span className="px-4">Click the overview to look closer, then click the feature here.</span>
          )}
        </div>
      </div>
      <p className="tabular mt-1 text-xs text-ink-2">
        Point: {point ? `${point.x.toFixed(2)}, ${point.y.toFixed(2)} px (original image)` : 'not placed yet'}
      </p>
    </div>
  )
}

export default function ControlPoints({ jobId, result, overviewSource, overviewReference, onDone }) {
  const [source, setSource] = useState(null)
  const [reference, setReference] = useState(null)
  const [pairs, setPairs] = useState([])
  const [provenance, setProvenance] = useState('')
  const [csv, setCsv] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const add = () => {
    setPairs([...pairs, { source_x: source.x, source_y: source.y, reference_x: reference.x, reference_y: reference.y }])
    setSource(null)
    setReference(null)
  }
  const submit = async (e) => {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      await api.controlPoints(jobId, pairs.length ? { provenance, points: pairs } : { provenance, csv })
      setPairs([])
      setCsv('')
      onDone()
    } catch (err) {
      setError(err)
    } finally {
      setBusy(false)
    }
  }
  const readFile = async (e) => {
    const file = e.target.files?.[0]
    if (file) setCsv(await file.text())
  }
  const field = 'mt-1 block w-full rounded-lg border border-white/15 bg-surface-2 px-3 py-2 text-sm text-ink'

  return (
    <Card>
      <p className="text-sm text-ink-2">
        Measure this result against points you identified yourself. They are applied to the finished transformation and are never used to fit anything. Find the
        same feature in both images, without looking at the overlay. Pick features that do not move with the Sun: small fresh craters rather than shadow edges.
      </p>
      <div className="mt-4 grid gap-6 xl:grid-cols-2">
        <PickSide jobId={jobId} which="source" title="Source" overview={overviewSource} width={result.source.width} height={result.source.height} rectifyWith={result.transformation?.matrix} point={source} onPoint={setSource} />
        <PickSide jobId={jobId} which="reference" title="Reference" overview={overviewReference} width={result.reference.width} height={result.reference.height} point={reference} onPoint={setReference} />
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-3">
        <Button variant="secondary" onClick={add} disabled={!source || !reference}>
          Add this pair
        </Button>
        <span className="text-xs text-muted">{pairs.length} pair(s) ready; at least 6 well-spread pairs give a figure worth reading.</span>
      </div>
      {pairs.length > 0 && (
        <div className="mt-3 overflow-x-auto">
          <table className="tabular w-full text-xs">
            <thead className="text-left text-ink-2">
              <tr>
                <th className="py-1 pr-3 font-medium">#</th>
                <th className="py-1 pr-3 font-medium">Source (px)</th>
                <th className="py-1 pr-3 font-medium">Reference (px)</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {pairs.map((p, i) => (
                <tr key={i} className="border-t border-white/5">
                  <td className="py-1 pr-3">{i + 1}</td>
                  <td className="py-1 pr-3">
                    {p.source_x.toFixed(2)}, {p.source_y.toFixed(2)}
                  </td>
                  <td className="py-1 pr-3">
                    {p.reference_x.toFixed(2)}, {p.reference_y.toFixed(2)}
                  </td>
                  <td className="py-1 text-right">
                    <button type="button" className="text-accent hover:underline" onClick={() => setPairs(pairs.filter((_, j) => j !== i))}>
                      Remove
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <form onSubmit={submit} className="mt-4 grid gap-3">
        {pairs.length === 0 && (
          <details className="text-xs text-ink-2">
            <summary className="cursor-pointer">Or paste or upload a CSV</summary>
            <label className="mt-2 block">
              Original image pixels: source_x, source_y, reference_x, reference_y (optional id, note)
              <textarea value={csv} onChange={(e) => setCsv(e.target.value)} rows={4} className={`${field} font-mono text-xs`} placeholder={'source_x,source_y,reference_x,reference_y\n'} />
            </label>
            <input type="file" accept=".csv,text/csv" onChange={readFile} className="mt-2 text-sm text-ink-2 file:mr-3 file:rounded-md file:border-0 file:bg-white/10 file:px-3 file:py-1.5 file:text-ink" />
          </details>
        )}
        <label className="text-xs text-ink-2">
          Where do these points come from?
          <input required value={provenance} onChange={(e) => setProvenance(e.target.value)} className={field} placeholder="Who identified them, and how" />
        </label>
        <div>
          <Button type="submit" disabled={busy || (pairs.length === 0 && !csv.trim())}>
            {busy ? 'EVALUATING…' : 'EVALUATE CONTROL POINTS'}
          </Button>
        </div>
      </form>
      <ErrorNote error={error} />
    </Card>
  )
}
