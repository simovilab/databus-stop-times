// ─── Backend contract types ───────────────────────────────────────────────────
// Keep in sync with the stoptimes val FastAPI backend

export interface Source {
  name: string
  poll_interval_s: number
  vehicle_positions_url: string
  trip_updates_url: string
  gtfs_static_url: string
}

export interface RunParams {
  source: string
  predictors: string[]
  routes: string[]
  duration: string
}

export interface RunSummary {
  ticks: number
  samples_saved: number
  arrivals_saved: number
  results_saved: number
}

export type RunStatus = 'pending' | 'running' | 'done' | 'error'

// ─── Provenance ────────────────────────────────────────────────────────────────
// Mirrors stoptimes.val.provenance.RunProvenance.to_dict() – optional/best-effort;
// every field may be null/absent when it could not be determined. Field names
// (etaval_git_sha, etaval_dirty, ...) are kept as-is: they're the JSON keys the
// backend actually serializes, not a display label -- see RunProvenance for why.

export interface RunProvenance {
  etaval_git_sha?: string | null
  etaval_dirty?: boolean | null
  gtfs_eta_git_sha?: string | null
  gtfs_eta_dirty?: boolean | null
  source_name?: string | null
  source_config_sha256?: string | null
  gtfs_snapshot?: string | null
  detectors?: Record<string, Record<string, unknown>>
  seed?: number | null
  predictor_ids?: string[]
  versions?: Record<string, string | null>
  created_at?: string | null
}

export interface RunInfo {
  id: string
  params: RunParams
  status: RunStatus
  summary: RunSummary
  db_path: string
  started_at: string | null
  ended_at: string | null
  error: string | null
  provenance?: RunProvenance | null
}

export interface CreateRunRequest {
  source: string
  predictors: string[]
  routes: string[]
  duration: string
}

export interface CreateRunResponse {
  run_id: string
}

// ─── Metrics ──────────────────────────────────────────────────────────────────

// Bootstrap CI on MAE. Present only when metrics were fetched with ?stats=1.
export interface MaeCi {
  lo: number
  hi: number
  ci: number // e.g. 0.95
  n_boot: number // e.g. 1000
}

// Per-predictor inference-latency summary. null when unavailable (e.g. the
// predictor did not report timings for this run).
export interface LatencyStats {
  p50_s: number
  p95_s: number
  p99_s: number
  count: number
}

export interface PerPredictorMetric {
  predictor_id: string
  mae: number
  rmse: number
  mape: number
  bias: number
  count: number
  // Stats-layer additions (?stats=1 only) — optional/nullable, always.
  mae_ci?: MaeCi | null
  latency?: LatencyStats | null
}

export interface HorizonBucket {
  label: string
  predictor_id: string
  mae: number
  bias: number
  count: number
}

// Pairwise significance test between two predictors' errors.
export interface SignificanceResult {
  a: string
  b: string
  test: 'wilcoxon' | 'diebold_mariano'
  statistic: number
  p_value: number
  p_adjusted: number
  reject: boolean
  n: number
  diff_mean: number
}

// One horizon x tolerance calibration bin (nominal vs empirical coverage).
// Field names are the pinned contract; every field is treated as possibly
// absent since the exact backend serialization may vary.
export interface CalibrationBin {
  horizon_lo?: number
  horizon_hi?: number
  band_lo_s?: number
  band_hi_s?: number
  band_label?: string
  tolerance_s?: number
  n?: number
  nominal_coverage?: number
  empirical_coverage?: number
}

export interface RunMetrics {
  per_predictor: PerPredictorMetric[]
  horizon_buckets: HorizonBucket[]
  convergence: Record<string, unknown>[]
  head_to_head: Record<string, unknown>[]
  // Stats-layer additions (?stats=1 only) — optional, may be absent entirely
  // on a plain (non-stats) response.
  significance?: SignificanceResult[]
  calibration?: CalibrationBin[]
  seed?: number
}

// ─── Results ──────────────────────────────────────────────────────────────────

export interface ResultRow {
  trip_key: string
  predictor_id: string
  stop_id: string
  stop_sequence: number
  route_id: string
  error_s: number
  horizon_s: number
  dist_remaining_m: number
  stops_remaining: number
  predicted_arrival_epoch: number
  actual_arrival_epoch: number
  made_at_epoch: number
}

// ─── WebSocket stream frame types ─────────────────────────────────────────────

export interface Vehicle {
  trip_key: string
  route_id: string
  lat: number
  lon: number
  progress_m: number
}

export interface SnapshotFrame {
  type: 'snapshot'
  data: {
    run_id: string
    status: RunStatus
    summary: RunSummary
    started_at: string | null
  }
}

export interface TickFrame {
  type: 'tick'
  data: {
    tick: number
    header_timestamp: number
    samples_saved: number
    arrivals_saved: number
    results_saved: number
    new_results: ResultRow[]
    vehicles: Vehicle[]
  }
}

export interface EndFrame {
  type: 'end'
  data: {
    status: RunStatus
    summary: RunSummary
    error: string | null
  }
}

export type StreamFrame = SnapshotFrame | TickFrame | EndFrame
