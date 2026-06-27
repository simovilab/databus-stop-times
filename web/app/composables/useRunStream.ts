import type {
  RunStatus,
  RunSummary,
  ResultRow,
  Vehicle,
  StreamFrame
} from '~/types/api'

const RESULT_BUFFER = 200 // keep last N accumulated results

/**
 * Opens a WebSocket to /api/runs/{id}/stream and maps the three frame types
 * to reactive state:
 *
 *  snapshot  → status, summary (initial state from backend)
 *  tick      → ticks (counter), summary (live counts), latestResults (rolling
 *               buffer of last RESULT_BUFFER entries), vehicles (latest list)
 *  end       → status, summary (final), ended = true, socket closed
 *
 * Call start() after confirming the run is live; close() is also exposed for
 * manual teardown. onUnmounted cleans up automatically.
 */
export function useRunStream(runId: string) {
  const config = useRuntimeConfig()
  const apiBase = config.public.apiBase as string
  // Derive WebSocket base: http(s)://… → ws(s)://…
  const wsBase = apiBase.replace(/^https?/, (m) => (m === 'https' ? 'wss' : 'ws'))

  const status = ref<RunStatus>('pending')
  const summary = ref<RunSummary>({
    ticks: 0,
    samples_saved: 0,
    arrivals_saved: 0,
    results_saved: 0
  })
  const ticks = ref(0)
  const latestResults = ref<ResultRow[]>([])
  const vehicles = ref<Vehicle[]>([])
  const connected = ref(false)
  const ended = ref(false)
  const connectError = ref<string | null>(null)

  let ws: WebSocket | null = null

  function start() {
    if (!import.meta.client) return
    const url = `${wsBase}/api/runs/${runId}/stream`

    try {
      ws = new WebSocket(url)
    } catch (err) {
      connectError.value = String(err)
      return
    }

    ws.onopen = () => {
      connected.value = true
      connectError.value = null
    }

    ws.onmessage = (event: MessageEvent) => {
      let frame: StreamFrame
      try {
        frame = JSON.parse(event.data as string) as StreamFrame
      } catch {
        return
      }

      if (frame.type === 'snapshot') {
        status.value = frame.data.status
        summary.value = { ...frame.data.summary }
      } else if (frame.type === 'tick') {
        const d = frame.data
        ticks.value = d.tick
        summary.value = {
          ticks: d.tick,
          samples_saved: d.samples_saved,
          arrivals_saved: d.arrivals_saved,
          results_saved: d.results_saved
        }
        // Rolling buffer – keep freshest entries
        const combined = [...latestResults.value, ...d.new_results]
        latestResults.value = combined.length > RESULT_BUFFER
          ? combined.slice(-RESULT_BUFFER)
          : combined
        vehicles.value = d.vehicles
      } else if (frame.type === 'end') {
        status.value = frame.data.status
        summary.value = { ...frame.data.summary }
        ended.value = true
        connected.value = false
        ws?.close()
        ws = null
      }
    }

    ws.onerror = () => {
      connectError.value = 'WebSocket connection error'
      connected.value = false
    }

    ws.onclose = () => {
      connected.value = false
    }
  }

  function close() {
    ws?.close()
    ws = null
  }

  // Always register cleanup so callers can call start() conditionally
  onUnmounted(() => {
    close()
  })

  return {
    status,
    summary,
    ticks,
    latestResults,
    vehicles,
    connected,
    ended,
    connectError,
    start,
    close
  }
}
