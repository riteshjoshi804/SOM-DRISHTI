import { useState } from 'react'

// Single-series histogram of raster values (counts per value bin) from the backend.
// One hue, no legend (the title names the series), hairline axis, hover tooltip per bar.
const W = 520
const H = 150
const PAD = { l: 8, r: 8, t: 10, b: 26 }
const GAP = 1

function compact(v) {
  const a = Math.abs(v)
  if (a >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 })
  if (a >= 10) return v.toFixed(1)
  return v.toFixed(2)
}

export default function Histogram({ counts, edges, title }) {
  const [hover, setHover] = useState(null)
  if (!counts?.length || edges?.length !== counts.length + 1) {
    return <p className="text-sm text-muted">Histogram not available.</p>
  }
  const max = Math.max(...counts)
  const plotW = W - PAD.l - PAD.r
  const plotH = H - PAD.t - PAD.b
  const bw = plotW / counts.length
  const total = counts.reduce((a, b) => a + b, 0)

  return (
    <div className="relative">
      <div className="mb-1 text-xs text-ink-2">{title}</div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={`${title}: histogram of pixel values`}>
        <line x1={PAD.l} x2={W - PAD.r} y1={PAD.t + plotH} y2={PAD.t + plotH} stroke="var(--color-axis)" strokeWidth="1" />
        {counts.map((c, i) => {
          const h = max ? (c / max) * plotH : 0
          const x = PAD.l + i * bw + GAP / 2
          const y = PAD.t + plotH - h
          const r = Math.min(2, (bw - GAP) / 2, h)
          return (
            <g key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
              {/* Hit target spans the full column height so thin bars are easy to hover. */}
              <rect x={PAD.l + i * bw} y={PAD.t} width={bw} height={plotH} fill="transparent" />
              {h > 0 && (
                <path
                  d={`M${x},${PAD.t + plotH} V${y + r} Q${x},${y} ${x + r},${y} H${x + bw - GAP - r} Q${x + bw - GAP},${y} ${x + bw - GAP},${y + r} V${PAD.t + plotH} Z`}
                  fill="var(--color-accent)"
                  opacity={hover === null || hover === i ? 1 : 0.55}
                />
              )}
            </g>
          )
        })}
        <text x={PAD.l} y={H - 8} fill="var(--color-muted)" fontSize="11" className="tabular">
          {compact(edges[0])}
        </text>
        <text x={W - PAD.r} y={H - 8} fill="var(--color-muted)" fontSize="11" textAnchor="end" className="tabular">
          {compact(edges[edges.length - 1])}
        </text>
        <text x={W / 2} y={H - 8} fill="var(--color-muted)" fontSize="11" textAnchor="middle">
          pixel value (0.5–99.5 percentile range)
        </text>
      </svg>
      {hover !== null && (
        <div
          className="pointer-events-none absolute top-5 rounded-md border border-white/15 bg-surface-2 px-2.5 py-1.5 text-xs shadow-lg"
          style={{ left: `min(calc(${((hover + 0.5) / counts.length) * 100}% + 8px), calc(100% - 11rem))` }}
        >
          <div className="text-ink-2">
            {compact(edges[hover])} – {compact(edges[hover + 1])}
          </div>
          <div className="tabular font-semibold">
            {counts[hover].toLocaleString()} px ({((counts[hover] / total) * 100).toFixed(1)}%)
          </div>
        </div>
      )}
    </div>
  )
}
