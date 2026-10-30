import { NA } from '../format.js'

const STATUS = {
  SUCCESS: { icon: '✓', label: 'Success', cls: 'border-good/60 text-good bg-good/10' },
  LOW_CONFIDENCE: { icon: '!', label: 'Low confidence', cls: 'border-warning/60 text-warning bg-warning/10' },
  FAILED: { icon: '✕', label: 'Failed', cls: 'border-critical/60 text-critical bg-critical/10' },
  queued: { icon: '…', label: 'Queued', cls: 'border-white/20 text-ink-2 bg-white/5' },
  running: { icon: '◌', label: 'Running', cls: 'border-accent/60 text-accent bg-accent/10' },
}

export function StatusBadge({ status, className = '' }) {
  const s = STATUS[status]
  if (!s) return <span className="text-muted">{NA}</span>
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-semibold ${s.cls} ${className}`}
    >
      <span aria-hidden="true">{s.icon}</span>
      {s.label}
    </span>
  )
}

export function Card({ children, className = '' }) {
  return <div className={`rounded-xl border border-white/10 bg-surface p-5 ${className}`}>{children}</div>
}

export function SectionTitle({ children, hint }) {
  return (
    <div className="mb-3 mt-8 flex items-baseline justify-between gap-4">
      <h2 className="text-lg font-semibold tracking-tight">{children}</h2>
      {hint && <p className="text-xs text-muted">{hint}</p>}
    </div>
  )
}

export function MetricCard({ label, value, hint }) {
  return (
    <div className="rounded-xl border border-white/10 bg-surface p-4">
      <div className="text-xs text-ink-2">{label}</div>
      <div className={`mt-1 text-2xl font-semibold ${value === NA ? 'text-muted' : ''}`}>{value}</div>
      {hint && <div className="mt-1 text-[11px] leading-snug text-muted">{hint}</div>}
    </div>
  )
}

export function KeyValue({ rows }) {
  return (
    <dl className="grid grid-cols-[minmax(9rem,max-content)_1fr] gap-x-6 gap-y-1.5 text-sm">
      {rows.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-ink-2">{k}</dt>
          <dd className={`tabular break-all ${v === NA || v === null || v === undefined ? 'text-muted' : ''}`}>
            {v === null || v === undefined ? NA : v}
          </dd>
        </div>
      ))}
    </dl>
  )
}

export function ImagePanel({ title, src, caption, missing = 'Not produced for this run' }) {
  return (
    <figure className="flex flex-col overflow-hidden rounded-xl border border-white/10 bg-surface">
      <figcaption className="flex items-center justify-between border-b border-white/10 px-4 py-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-ink-2">{title}</span>
        {src && (
          <a href={src} target="_blank" rel="noreferrer" className="text-xs text-accent hover:underline">
            Open full size
          </a>
        )}
      </figcaption>
      <div className="flex min-h-40 flex-1 items-center justify-center bg-black/40 p-2">
        {src ? (
          <a href={src} target="_blank" rel="noreferrer" className="block">
            <img src={src} alt={title} loading="lazy" className="max-h-[520px] w-auto max-w-full object-contain" />
          </a>
        ) : (
          <span className="px-4 text-center text-sm text-muted">{missing}</span>
        )}
      </div>
      {caption && <p className="border-t border-white/10 px-4 py-2 text-xs leading-relaxed text-muted">{caption}</p>}
    </figure>
  )
}

export function Button({ children, onClick, disabled, variant = 'primary', type = 'button' }) {
  const styles =
    variant === 'primary'
      ? 'bg-accent text-white hover:bg-accent/85 disabled:bg-white/10 disabled:text-muted'
      : 'border border-white/15 text-ink-2 hover:bg-white/5 disabled:text-muted'
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`rounded-lg px-4 py-2 text-sm font-semibold tracking-wide transition-colors disabled:cursor-not-allowed ${styles}`}
    >
      {children}
    </button>
  )
}

export function ErrorNote({ error }) {
  if (!error) return null
  return (
    <div className="my-4 rounded-lg border border-critical/50 bg-critical/10 px-4 py-3 text-sm text-ink">
      <span className="font-semibold text-critical">✕ Error: </span>
      {String(error.message || error)}
    </div>
  )
}

export function Loading({ what = 'Loading' }) {
  return <p className="my-6 text-sm text-muted">{what}…</p>
}
