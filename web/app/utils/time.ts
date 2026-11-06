/**
 * The API sends run times as Unix epoch **seconds** (floats); older payloads may be ISO strings.
 * `new Date(seconds)` would read them as milliseconds and show January 1970.
 * Auto-imported by Nuxt (app/utils).
 */
export function toDate(value: string | number | null | undefined): Date | null {
  if (value === null || value === undefined || value === '') return null
  const d = typeof value === 'number' ? new Date(value * 1000) : new Date(value)
  return Number.isNaN(d.getTime()) ? null : d
}
