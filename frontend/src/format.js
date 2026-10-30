// Formatting helpers. Anything missing renders as "N/A" - never a made-up value.

export const NA = 'N/A'

const missing = (v) => v === null || v === undefined || (typeof v === 'number' && !Number.isFinite(v))

export function num(v, digits = 2) {
  if (missing(v)) return NA
  return Number(v).toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })
}

export function int(v) {
  if (missing(v)) return NA
  return Math.round(Number(v)).toLocaleString()
}

export function pct(v, digits = 1) {
  if (missing(v)) return NA
  return `${(Number(v) * 100).toFixed(digits)}%`
}

export function withUnit(v, unit, digits = 2) {
  const s = num(v, digits)
  return s === NA ? NA : `${s} ${unit}`
}

export function deg(v, digits = 2) {
  return withUnit(v, '°', digits).replace(' °', '°')
}

export function shortId(sceneId) {
  // OHRXXD18CHO2436502NNNN25039175231280_V2_1_01 -> CHO2436502 … V2_1_01
  if (!sceneId) return NA
  const m = sceneId.match(/(CHO\d+).*?(_V[\w_]+)$/)
  return m ? `${m[1]}${m[2]}` : sceneId
}

export function when(iso) {
  if (!iso) return NA
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString()
}
