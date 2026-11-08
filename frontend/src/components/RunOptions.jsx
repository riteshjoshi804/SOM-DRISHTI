import { useEffect, useState } from 'react'
import { api } from '../api.js'

export const REPRESENTATIONS = [
  ['', 'Default for the matcher'],
  ['raw_normalized', 'Normalized grayscale'],
  ['clahe', 'CLAHE'],
  ['gradient', 'Gradient'],
  ['structural', 'Structural (local gradient)'],
  ['phase_congruency', 'Phase congruency'],
  ['shadow_aware', 'Shadow-aware'],
]

const KEY = 'som-drishti-run-options'

function saved() {
  try {
    return { matcher: 'AUTO', representation: '', ...JSON.parse(localStorage.getItem(KEY) || '{}') }
  } catch {
    return { matcher: 'AUTO', representation: '' }
  }
}

// The options chosen last, kept across pages so that every RUN button means the same thing.
export function useRunOptions() {
  const [options, setOptions] = useState(saved)
  const [matchers, setMatchers] = useState(null)

  useEffect(() => {
    api.matchers().then(setMatchers).catch(() => setMatchers([]))
  }, [])

  const update = (change) => {
    setOptions((current) => {
      const next = { ...current, ...change }
      localStorage.setItem(KEY, JSON.stringify(next))
      return next
    })
  }
  return { options, update, matchers }
}

export function runConfig({ matcher, representation }) {
  const config = { matcher }
  if (representation) {
    config.preprocessing_modes = [representation]
    config.guided = { representation }
  }
  return config
}

function usable(status) {
  return status.available || status.needs_download
}

export function RunOptions({ options, update, matchers }) {
  const field = 'mt-1 block w-full rounded-lg border border-white/15 bg-surface-2 px-3 py-2 text-sm text-ink'
  const selected = matchers?.find((m) => m.name === options.matcher)
  return (
    <div className="grid gap-4 md:grid-cols-[1fr_1fr_2fr] md:items-start">
      <label className="text-xs text-ink-2">
        Matcher
        <select className={field} value={options.matcher} onChange={(e) => update({ matcher: e.target.value })}>
          <option value="AUTO">AUTO: compare the available matchers</option>
          {(matchers || []).map((m) => (
            <option key={m.name} value={m.name} disabled={!usable(m)}>
              {m.label}
              {usable(m) ? '' : ' (unavailable)'}
            </option>
          ))}
        </select>
      </label>
      <label className="text-xs text-ink-2">
        Image representation
        <select className={field} value={options.representation} onChange={(e) => update({ representation: e.target.value })}>
          {REPRESENTATIONS.map(([value, label]) => (
            <option key={value} value={value}>
              {label}
            </option>
          ))}
        </select>
      </label>
      <p className="text-xs leading-relaxed text-muted md:pt-5">
        {options.matcher === 'AUTO'
          ? 'Runs every available matcher, then chooses by coverage, held-out error and inlier ratio, never by match count. Expect several minutes per scene on a CPU.'
          : selected
            ? `${selected.implementation}. Filtering: ${selected.filtering}.${selected.needs_download ? ' Its weights are downloaded on first use.' : ''}${
                selected.experimental ? ' Experimental: AUTO does not use it.' : ''
              }`
            : ''}
      </p>
      {matchers?.filter((m) => !usable(m)).map((m) => (
        <p key={m.name} className="text-xs text-warning md:col-span-3">
          ! {m.label} is unavailable: {m.reason}
        </p>
      ))}
    </div>
  )
}
