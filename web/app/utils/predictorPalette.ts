// Predictor colours and labels shared by every chart.
//
// Colour follows the predictor, never its rank: a filter that hides one series
// must not repaint the others. Baselines (timetable, constant speed) are a
// recessive grey — they are the floor, not a contender. The live feed (gtfsrt)
// is the incumbent to beat and always gets orange. Models take the remaining
// hues in a fixed order.
//
// The hue order was validated for this dark surface (#0d1512) with the dataviz
// palette checker: lightness band, chroma floor, adjacent-pair CVD ΔE ≥ 9.4,
// normal-vision ΔE ≥ 19.7, contrast ≥ 3:1. With more than ~4 series on one
// scatter, some pairs get close for colour-blind readers; legends and tooltips
// always carry the name, so colour is never the only cue.

export const BASELINE_COLOR = '#7c8a83'
export const INCUMBENT_COLOR = '#d95926'
export const MODEL_COLORS: readonly string[] = [
  '#199e70', // aqua-green
  '#3987e5', // blue
  '#d55181', // magenta
  '#9085e9', // violet
  '#c98500', // ochre
]

export function isBaseline(id: string): boolean {
  return id.startsWith('baseline:')
}

export function isIncumbent(id: string): boolean {
  return id === 'gtfsrt' || id.startsWith('gtfsrt:')
}

/** Stable colour per predictor, computed over the full (unfiltered) predictor list. */
export function buildColorMap(predictorIds: readonly string[]): Record<string, string> {
  const map: Record<string, string> = {}
  let modelIdx = 0
  const baselineShades = ['#7c8a83', '#5f6c66', '#9aa59f']
  let baselineIdx = 0
  for (const id of [...predictorIds].sort()) {
    if (isBaseline(id)) {
      map[id] = baselineShades[baselineIdx++ % baselineShades.length] as string
    } else if (isIncumbent(id)) {
      map[id] = INCUMBENT_COLOR
    } else {
      map[id] = MODEL_COLORS[modelIdx++ % MODEL_COLORS.length] as string
    }
  }
  return map
}

/** Short display label: drops the `stoptimes:` namespace, names the baselines plainly. */
export function predictorLabel(id: string): string {
  if (id === 'gtfsrt') return 'GTFS-RT feed'
  if (id === 'baseline:schedule') return 'timetable'
  if (id === 'baseline:constant_speed') return 'constant speed'
  if (id.startsWith('baseline:')) return id.slice('baseline:'.length)
  if (id.startsWith('stoptimes:')) return id.slice('stoptimes:'.length)
  return id
}

/**
 * Charts key series by data field. Predictor ids contain `:`, `.` and `-`, which
 * the chart library may read as property paths, so every chart re-keys its
 * series as `s0, s1, …` and keeps the mapping here.
 */
export interface SeriesKeys {
  keys: string[]
  byId: Record<string, string>
  categories: Record<string, { name: string; color: string }>
}

export function seriesKeys(predictorIds: readonly string[], colorMap: Record<string, string>): SeriesKeys {
  const keys: string[] = []
  const byId: Record<string, string> = {}
  const categories: Record<string, { name: string; color: string }> = {}
  predictorIds.forEach((id, i) => {
    const key = `s${i}`
    keys.push(key)
    byId[id] = key
    categories[key] = { name: predictorLabel(id), color: colorMap[id] ?? BASELINE_COLOR }
  })
  return { keys, byId, categories }
}

export function fmtSeconds(v: number): string {
  if (!Number.isFinite(v)) return '—'
  const a = Math.abs(v)
  if (a >= 120) return `${(v / 60).toFixed(a >= 600 || a % 60 === 0 ? 0 : 1)}m`
  return `${Math.round(v)}s`
}

export function fmtSignedSeconds(v: number): string {
  if (!Number.isFinite(v)) return '—'
  return `${v > 0 ? '+' : v < 0 ? '−' : ''}${fmtSeconds(Math.abs(v))}`
}
