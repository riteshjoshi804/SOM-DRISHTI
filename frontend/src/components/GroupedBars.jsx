import { useState } from 'react'

// Horizontal bars grouped by scene, one fixed colour per matcher. A result that does not exist is
// marked in words at the baseline, so that "no result" can never be mistaken for a value of zero.
// Fixed colour per matcher (validated for the dark surface); see MATCHER_COLOURS in Benchmark.jsx.
const W = 620
const LABEL_W = 124
const RIGHT = 56
const BAR = 10
const GAP = 2
const GROUP_GAP = 14
const TOP = 26
const BOTTOM = 24

function ticks(max) {
  if (!(max > 0)) return [0]
  const rough = max / 4
  const power = 10 ** Math.floor(Math.log10(rough))
  const unit = [1, 2, 5, 10].map((m) => m * power).find((u) => u >= rough) ?? 10 * power
  const out = []
  for (let v = 0; v <= max + 1e-9; v += unit) out.push(v)
  return out
}

function roundedBar(x, y, width, height) {
  const r = Math.min(4, height / 2, width)
  return `M${x},${y} H${x + width - r} Q${x + width},${y} ${x + width},${y + r} V${y + height - r} Q${x + width},${y + height} ${x + width - r},${y + height} H${x} Z`
}

export default function GroupedBars({ title, subtitle, categories, series, value, format, reason }) {
  const [hover, setHover] = useState(null)
  const values = categories.flatMap((c) => series.map((s) => value(c.key, s.key))).filter((v) => v !== null && v !== undefined && Number.isFinite(v))
  const max = values.length ? Math.max(...values) : 0
  const axis = ticks(max)
  const top = axis[axis.length - 1] || 1
  const plot = W - LABEL_W - RIGHT
  const groupH = series.length * (BAR + GAP) - GAP
  const H = TOP + categories.length * (groupH + GROUP_GAP) - GROUP_GAP + BOTTOM
  const scale = (v) => (v / top) * plot

  return (
    <figure className="relative rounded-xl border border-white/10 bg-surface p-4">
      <figcaption className="mb-2">
        <div className="text-sm font-semibold">{title}</div>
        {subtitle && <div className="text-xs text-muted">{subtitle}</div>}
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={`${title}, by scene and matcher. The table below holds every value.`}>
        {axis.map((t) => (
          <g key={t}>
            <line x1={LABEL_W + scale(t)} x2={LABEL_W + scale(t)} y1={TOP - 6} y2={H - BOTTOM} stroke={t === 0 ? 'var(--color-axis)' : 'var(--color-grid)'} strokeWidth="1" />
            <text x={LABEL_W + scale(t)} y={H - 8} fill="var(--color-muted)" fontSize="11" textAnchor="middle" className="tabular">
              {format(t, true)}
            </text>
          </g>
        ))}
        {categories.map((c, i) => {
          const y0 = TOP + i * (groupH + GROUP_GAP)
          return (
            <g key={c.key}>
              <text x={LABEL_W - 8} y={y0 + groupH / 2 + 4} fill="var(--color-ink-2)" fontSize="12" textAnchor="end">
                {c.label}
              </text>
              {series.map((s, j) => {
                const v = value(c.key, s.key)
                const y = y0 + j * (BAR + GAP)
                const missing = v === null || v === undefined || !Number.isFinite(v)
                const active = hover?.c === c.key && hover?.s === s.key
                const dimmed = hover && !active
                return (
                  <g
                    key={s.key}
                    tabIndex={0}
                    onMouseEnter={() => setHover({ c: c.key, s: s.key, y })}
                    onMouseLeave={() => setHover(null)}
                    onFocus={() => setHover({ c: c.key, s: s.key, y })}
                    onBlur={() => setHover(null)}
                    style={{ outline: 'none' }}
                  >
                    {/* The hit target spans the whole row, so short bars are as easy to reach as long ones. */}
                    <rect x={LABEL_W} y={y - 1} width={plot} height={BAR + GAP} fill="transparent" />
                    {missing ? (
                      <text x={LABEL_W + 6} y={y + BAR - 1} fill="var(--color-muted)" fontSize="10">
                        {reason?.(c.key, s.key) || 'no result'}
                      </text>
                    ) : (
                      v > 0 && <path d={roundedBar(LABEL_W, y, Math.max(scale(v), 2), BAR)} fill={s.colour} opacity={dimmed ? 0.45 : 1} />
                    )}
                  </g>
                )
              })}
            </g>
          )
        })}
      </svg>
      <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2">
        {series.map((s) => (
          <span key={s.key} className="inline-flex items-center gap-1.5">
            <span aria-hidden="true" className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: s.colour }} />
            {s.label}
          </span>
        ))}
      </div>
      {hover && (
        <div
          className="pointer-events-none absolute right-4 rounded-md border border-white/15 bg-surface-2 px-2.5 py-1.5 text-xs shadow-lg"
          style={{ top: `calc(3rem + ${(hover.y / H) * 70}%)` }}
        >
          <div className="tabular font-semibold">
            {(() => {
              const v = value(hover.c, hover.s)
              return v === null || v === undefined || !Number.isFinite(v) ? reason?.(hover.c, hover.s) || 'no result' : format(v)
            })()}
          </div>
          <div className="text-ink-2">
            {series.find((s) => s.key === hover.s)?.label} · {categories.find((c) => c.key === hover.c)?.label}
          </div>
        </div>
      )}
    </figure>
  )
}
