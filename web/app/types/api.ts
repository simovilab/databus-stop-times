// ─── Backend contract types ───────────────────────────────────────────────────
// Keep in sync with etaval FastAPI backend

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

export interface RunInfo {
  id: string
  params: RunParams
  status: RunStatus
  summary: RunSummary
  db_path: string
  started_at: string | null
  ended_at: string | null
  error: string | null
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

export interface PerPredictorMetric {
  predictor_id: string
  mae: number
  rmse: number
  mape: number
  bias: number
  count: number
}

export interface HorizonBucket {
  label: string
  predictor_id: string
  mae: number
  bias: number
  count: number
}

export interface RunMetrics {
  per_predictor: PerPredictorMetric[]
  horizon_buckets: HorizonBucket[]
  convergence: Record<string, unknown>[]
  head_to_head: Record<string, unknown>[]
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
