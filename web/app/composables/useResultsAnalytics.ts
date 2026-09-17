import type { ResultRow } from '~/types/api'

// ── Aggregate types exported for use in components ─────────────────────────

export interface PredictorAgg {
  predictor_id: string
  mae: number
  rmse: number
  mape: number
  bias: number
  count: number
}

/** Horizon bucket datum – one row per bucket, predictor IDs as dynamic keys holding MAE. */
export interface BucketDatum {
  bucketIdx: number
  bucketLabel: string
  [predictor_id: string]: number | string
}

/** Error histogram bin – one row per bin center, predictor IDs as dynamic keys holding count. */
export interface HistBin {
  binCenter: number
  [predictor_id: string]: number
}

/** Route datum – one row per route, predictor IDs as dynamic keys holding MAE. */
export interface RouteDatum {
  routeIdx: number
  routeLabel: string
  [predictor_id: string]: number | string
}

/** Convergence datum – one row per horizon bin, predictor IDs as dynamic keys holding avg |error_s|. */
export interface ConvDatum {
  horizonBin: number
  [predictor_id: string]: number
}

export interface HeadToHeadResult {
  predictor_id: string
  wins: number
  contests: number
  winShare: number
}

// ── Internal helpers ───────────────────────────────────────────────────────

function mean(vals: number[]): number {
  if (!vals.length) return 0
  return vals.reduce((a, b) => a + b, 0) / vals.length
}

const PALETTE: readonly string[] = [
  '#2fd67b', '#7ee0a8', '#b8e8d0', '#e0b040', '#e06060',
  '#6fd6d0', '#a0d68f', '#d0c060', '#8090e0', '#d08fd0',
]

const BUCKET_LABELS: readonly string[] = ['<2m', '2-5m', '5-10m', '>10m']

function getBucketIdx(h: number): 0 | 1 | 2 | 3 {
  if (h < 120) return 0
  if (h < 300) return 1
  if (h < 600) return 2
  return 3
}

// Fixed histogram bin edges (in seconds)
const HIST_EDGES: readonly number[] = [-300, -240, -180, -120, -60, 0, 60, 120, 180, 240, 300]
const HIST_CENTERS: readonly number[] = HIST_EDGES.slice(0, -1).map((e, i) => (e + HIST_EDGES[i + 1]!) / 2)

function getHistBinIdx(e: number): number {
  for (let i = 0; i < HIST_EDGES.length - 1; i++) {
    if (e < HIST_EDGES[i + 1]!) return i
  }
  return HIST_EDGES.length - 2
}

// ── Main composable ────────────────────────────────────────────────────────

export function useResultsAnalytics(runId: string) {
  const api = useApi()

  // Raw data
  const rows = ref<ResultRow[]>([])
  const loading = ref(false)
  const fetchError = ref<string | null>(null)

  // Filter state (all reactive)
  const selectedPredictors = ref<string[]>([])
  const selectedRoute = ref<string | null>(null)
  const horizonRange = ref<[number, number] | null>(null)
  const selectedTripStop = ref<{ trip_key: string; stop_id: string } | null>(null)

  // ── Load ──────────────────────────────────────────────────────────────────

  async function load(): Promise<void> {
    if (loading.value) return
    loading.value = true
    fetchError.value = null
    try {
      rows.value = await api.getResults(runId)
      // Initialise selectedPredictors to all that are present
      selectedPredictors.value = [...new Set(rows.value.map((r) => r.predictor_id))].sort()
    } catch (e) {
      fetchError.value = e instanceof Error ? e.message : String(e)
    } finally {
      loading.value = false
    }
  }

  // ── Metadata (from full unfiltered dataset) ──────────────────────────────

  const predictors = computed<string[]>(() =>
    [...new Set(rows.value.map((r) => r.predictor_id))].sort()
  )

  const routes = computed<string[]>(() =>
    [...new Set(rows.value.map((r) => r.route_id))].sort()
  )

  /** Stable color map keyed by predictor_id – computed from all predictors so colours don't shift when filtering. */
  const colorMap = computed<Record<string, string>>(() => {
    const map: Record<string, string> = {}
    predictors.value.forEach((p, i) => {
      map[p] = PALETTE[i % PALETTE.length] as string
    })
    return map
  })

  /** Predictors that are currently toggled on. */
  const activePredictors = computed<string[]>(() =>
    predictors.value.filter((p) => selectedPredictors.value.includes(p))
  )

  // ── Filtered rows ─────────────────────────────────────────────────────────

  const filtered = computed<ResultRow[]>(() =>
    rows.value.filter((r) => {
      if (!selectedPredictors.value.includes(r.predictor_id)) return false
      if (selectedRoute.value && r.route_id !== selectedRoute.value) return false
      if (horizonRange.value) {
        const [lo, hi] = horizonRange.value
        if (r.horizon_s < lo || r.horizon_s > hi) return false
      }
      return true
    })
  )

  // ── Per-predictor aggregates from filtered ────────────────────────────────

  const predictorAggs = computed<PredictorAgg[]>(() => {
    const groups = new Map<string, ResultRow[]>()
    for (const r of filtered.value) {
      const g = groups.get(r.predictor_id)
      if (g) g.push(r)
      else groups.set(r.predictor_id, [r])
    }

    return predictors.value
      .filter((p) => groups.has(p))
      .map((pid) => {
        const gr = groups.get(pid)!
        return {
          predictor_id: pid,
          mae: mean(gr.map((r) => Math.abs(r.error_s))),
          rmse: Math.sqrt(mean(gr.map((r) => r.error_s ** 2))),
          mape: mean(
            gr.map((r) => {
              const actual_s = r.actual_arrival_epoch - r.made_at_epoch
              return actual_s > 0 ? (Math.abs(r.error_s) / actual_s) * 100 : 0
            })
          ),
          bias: mean(gr.map((r) => r.error_s)),
          count: gr.length,
        }
      })
  })

  // ── Horizon bucket pivot ──────────────────────────────────────────────────

  const bucketData = computed<BucketDatum[]>(() => {
    const preds = activePredictors.value
    const sums = new Map<string, number>()
    const cnts = new Map<string, number>()

    for (const r of filtered.value) {
      const b = getBucketIdx(r.horizon_s)
      const key = `${b}|${r.predictor_id}`
      sums.set(key, (sums.get(key) ?? 0) + Math.abs(r.error_s))
      cnts.set(key, (cnts.get(key) ?? 0) + 1)
    }

    return ([0, 1, 2, 3] as const).map((b) => {
      const datum: BucketDatum = {
        bucketIdx: b,
        bucketLabel: BUCKET_LABELS[b] as string,
      }
      for (const p of preds) {
        const key = `${b}|${p}`
        const cnt = cnts.get(key)
        datum[p] = cnt ? sums.get(key)! / cnt : NaN
      }
      return datum
    })
  })

  // ── Error histogram bins ──────────────────────────────────────────────────

  const histData = computed<HistBin[]>(() => {
    const preds = activePredictors.value
    const cnts = new Map<string, number>()

    for (const r of filtered.value) {
      const b = getHistBinIdx(r.error_s)
      const key = `${b}|${r.predictor_id}`
      cnts.set(key, (cnts.get(key) ?? 0) + 1)
    }

    return HIST_CENTERS.map((c, b) => {
      const datum: HistBin = { binCenter: c }
      for (const p of preds) {
        datum[p] = cnts.get(`${b}|${p}`) ?? 0
      }
      return datum
    })
  })

  // ── Per-route MAE pivot ───────────────────────────────────────────────────

  const routeData = computed<RouteDatum[]>(() => {
    const preds = activePredictors.value
    const sums = new Map<string, number>()
    const cnts = new Map<string, number>()
    const routeSet = new Set<string>()

    for (const r of filtered.value) {
      routeSet.add(r.route_id)
      const key = `${r.route_id}|${r.predictor_id}`
      sums.set(key, (sums.get(key) ?? 0) + Math.abs(r.error_s))
      cnts.set(key, (cnts.get(key) ?? 0) + 1)
    }

    return [...routeSet].sort().map((route, idx) => {
      const datum: RouteDatum = { routeIdx: idx, routeLabel: route }
      for (const p of preds) {
        const key = `${route}|${p}`
        const cnt = cnts.get(key)
        datum[p] = cnt ? sums.get(key)! / cnt : NaN
      }
      return datum
    })
  })

  // ── Head-to-head win rates ────────────────────────────────────────────────

  const headToHeadData = computed<HeadToHeadResult[]>(() => {
    const preds = activePredictors.value
    if (preds.length < 2) {
      return preds.map((p) => ({ predictor_id: p, wins: 0, contests: 0, winShare: 0 }))
    }

    // Group by (trip_key, stop_id) → predictor_id → |error_s| values
    const groups = new Map<string, Map<string, number[]>>()
    for (const r of filtered.value) {
      const key = `${r.trip_key}||${r.stop_id}`
      let predMap = groups.get(key)
      if (!predMap) { predMap = new Map(); groups.set(key, predMap) }
      const vals = predMap.get(r.predictor_id)
      if (vals) vals.push(Math.abs(r.error_s))
      else predMap.set(r.predictor_id, [Math.abs(r.error_s)])
    }

    const wins: Record<string, number> = Object.fromEntries(preds.map((p) => [p, 0]))
    let totalContests = 0

    for (const [, predMap] of groups) {
      const present = preds.filter((p) => predMap.has(p))
      if (present.length < 2) continue
      totalContests++
      let best = present[0]!
      let bestMae = mean(predMap.get(best)!)
      for (const p of present.slice(1)) {
        const mae = mean(predMap.get(p)!)
        if (mae < bestMae) { bestMae = mae; best = p }
      }
      wins[best] = (wins[best] ?? 0) + 1
    }

    return preds.map((p) => ({
      predictor_id: p,
      wins: wins[p] ?? 0,
      contests: totalContests,
      winShare: totalContests > 0 ? (wins[p] ?? 0) / totalContests : 0,
    }))
  })

  // ── Convergence data for selected (trip_key, stop_id) ────────────────────

  const convergenceData = computed<ConvDatum[]>(() => {
    const preds = activePredictors.value
    if (!selectedTripStop.value || !preds.length) return []

    const { trip_key, stop_id } = selectedTripStop.value
    const relevant = filtered.value.filter(
      (r) => r.trip_key === trip_key && r.stop_id === stop_id
    )
    if (!relevant.length) return []

    const maxH = Math.max(...relevant.map((r) => r.horizon_s))
    if (!isFinite(maxH) || maxH <= 0) return []

    const BIN_COUNT = 20
    const binSize = Math.max(maxH / BIN_COUNT, 1)

    const sums = new Map<string, number>()
    const cnts = new Map<string, number>()
    const binSet = new Set<number>()

    for (const r of relevant) {
      const b = Math.floor(r.horizon_s / binSize)
      binSet.add(b)
      const key = `${b}|${r.predictor_id}`
      sums.set(key, (sums.get(key) ?? 0) + Math.abs(r.error_s))
      cnts.set(key, (cnts.get(key) ?? 0) + 1)
    }

    return [...binSet].sort((a, b) => a - b).map((b) => {
      const datum: ConvDatum = { horizonBin: b * binSize }
      for (const p of preds) {
        const key = `${b}|${p}`
        const cnt = cnts.get(key)
        datum[p] = cnt ? sums.get(key)! / cnt : NaN
      }
      return datum
    })
  })

  // ── Filter helpers ─────────────────────────────────────────────────────────

  function togglePredictor(p: string): void {
    if (selectedPredictors.value.includes(p)) {
      selectedPredictors.value = selectedPredictors.value.filter((x) => x !== p)
    } else {
      selectedPredictors.value = [...selectedPredictors.value, p]
    }
  }

  function selectRoute(r: string | null): void {
    selectedRoute.value = r
  }

  function setHorizonRange(r: [number, number] | null): void {
    horizonRange.value = r
  }

  function selectTripStop(trip_key: string, stop_id: string): void {
    selectedTripStop.value = { trip_key, stop_id }
  }

  function clearTripStop(): void {
    selectedTripStop.value = null
  }

  function reset(): void {
    selectedPredictors.value = [...predictors.value]
    selectedRoute.value = null
    horizonRange.value = null
    selectedTripStop.value = null
  }

  return {
    // Raw data
    rows,
    loading,
    fetchError,
    filtered,
    // Metadata
    predictors,
    activePredictors,
    routes,
    colorMap,
    // Filter state (read from template)
    selectedPredictors,
    selectedRoute,
    horizonRange,
    selectedTripStop,
    // Derived aggregates
    predictorAggs,
    bucketData,
    histData,
    routeData,
    headToHeadData,
    convergenceData,
    // Actions
    load,
    togglePredictor,
    selectRoute,
    setHorizonRange,
    selectTripStop,
    clearTripStop,
    reset,
  }
}

export type ResultsAnalytics = ReturnType<typeof useResultsAnalytics>
