<script setup lang="ts">
import type { RunInfo, RunMetrics, ResultRow } from '~/types/api'

const route = useRoute()
const id = route.params.id as string

useHead({ title: `run ${id} – etaval` })

const api = useApi()
const router = useRouter()

// ── Run state ─────────────────────────────────────────────────────────────────
const run = ref<RunInfo | null>(null)
const loadError = ref<string | null>(null)

// ── Post-run data ─────────────────────────────────────────────────────────────
const metrics = ref<RunMetrics | null>(null)
const resultsSample = ref<ResultRow[]>([])
const metricsError = ref<string | null>(null)

// ── Delete ────────────────────────────────────────────────────────────────────
const deleting = ref(false)
const deleteError = ref<string | null>(null)

// ── Live stream (always call at top level; start() called conditionally) ───────
const stream = useRunStream(id)

type BadgeColor = 'success' | 'info' | 'warning' | 'error' | 'neutral'
const statusColors: Record<string, BadgeColor> = {
  pending: 'warning',
  running: 'info',
  done: 'success',
  error: 'error'
}

async function loadPostRunData() {
  try {
    const [m, r] = await Promise.all([
      api.getMetrics(id),
      api.getResults(id, { limit: 50 })
    ])
    metrics.value = m
    resultsSample.value = r
  } catch {
    metricsError.value = 'Metrics not yet available.'
  }
}

// Watch for stream ending → refresh run info + load metrics
watch(stream.ended, async (isEnded) => {
  if (!isEnded) return
  try {
    run.value = await api.getRun(id)
  } catch { /* ignore */ }
  await loadPostRunData()
})

onMounted(async () => {
  try {
    run.value = await api.getRun(id)
  } catch {
    loadError.value = 'Run not found or API is unreachable.'
    return
  }

  const s = run.value.status
  if (s === 'pending' || s === 'running') {
    stream.start()
  } else if (s === 'done') {
    await loadPostRunData()
  }
})

async function handleDelete() {
  if (!confirm(`Delete run ${id}? This cannot be undone.`)) return
  deleting.value = true
  deleteError.value = null
  try {
    await api.deleteRun(id)
    await router.push('/runs')
  } catch {
    deleteError.value = 'Failed to delete run.'
    deleting.value = false
  }
}

function fmtDate(iso: string | null): string {
  if (!iso) return '—'
  return new Date(iso).toLocaleString()
}

function fmtNum(n: number | undefined): string {
  if (n === undefined || n === null) return '—'
  return n.toFixed(1)
}

// ── Winner (lowest MAE) ────────────────────────────────────────────────────
const winner = computed(() => {
  if (!metrics.value || metrics.value.per_predictor.length === 0) return null
  return [...metrics.value.per_predictor].sort((a, b) => a.mae - b.mae)[0] ?? null
})

// Display reactive status from stream if live, else from run snapshot
const displayStatus = computed(() => {
  if (stream.connected.value || stream.ended.value) return stream.status.value
  return run.value?.status ?? 'pending'
})

const displaySummary = computed(() => {
  if (stream.connected.value || stream.ended.value) return stream.summary.value
  return run.value?.summary ?? { ticks: 0, samples_saved: 0, arrivals_saved: 0, results_saved: 0 }
})

const isLive = computed(() =>
  displayStatus.value === 'pending' || displayStatus.value === 'running'
)
const isDone = computed(() => displayStatus.value === 'done')
const isError = computed(() => displayStatus.value === 'error')
</script>

<template>
  <div class="p-3 space-y-3">
    <!-- Header -->
    <div class="flex items-start justify-between gap-4">
      <div>
        <div class="flex items-center gap-3 mb-1">
          <h1 class="text-bright font-bold font-mono truncate max-w-xs" :title="id">
            {{ id }}
          </h1>
          <UBadge
            v-if="run || stream.connected.value"
            :color="statusColors[displayStatus] ?? 'neutral'"
            variant="subtle"
            :label="displayStatus"
          />
          <span v-if="stream.connected.value" class="flex items-center gap-1.5 text-[10px] text-accent border border-accent px-2 py-0.5 tracking-widest uppercase">
            <span class="status-led ok" />live
          </span>
        </div>
        <p v-if="run" class="text-[11px] text-dim">
          source <span class="text-main">{{ run.params.source }}</span> ·
          predictors <span class="text-main">{{ run.params.predictors.join(', ') }}</span> ·
          duration <span class="text-main">{{ run.params.duration }}</span>
          <span v-if="run.started_at"> · started {{ fmtDate(run.started_at) }}</span>
        </p>
      </div>
      <UButton
        label="Delete run"
        color="error"
        variant="ghost"
        icon="i-heroicons-trash"
        :loading="deleting"
        @click="handleDelete"
      />
    </div>

    <!-- Load error -->
    <UAlert v-if="loadError" color="error" variant="subtle" :title="loadError" />
    <UAlert v-if="deleteError" color="error" variant="subtle" :title="deleteError" />

    <!-- ── LIVE VIEW ──────────────────────────────────────────────────────── -->
    <template v-if="isLive && run">
      <ConsoleLiveRunPanel
        :status="displayStatus"
        :connected="stream.connected.value"
        :vehicle-count="stream.vehicles.value.length"
      />

      <ConsoleCounterTiles :summary="displaySummary" />

      <ConsoleResultsTicker :results="stream.latestResults.value" :limit="20" />

      <UAlert
        v-if="stream.connectError.value"
        color="warning"
        variant="subtle"
        :title="`Stream: ${stream.connectError.value}`"
      />
    </template>

    <!-- ── ERROR VIEW ─────────────────────────────────────────────────────── -->
    <template v-else-if="isError && run">
      <UAlert
        color="error"
        variant="subtle"
        title="Run failed"
        :description="run.error ?? 'Unknown error'"
      />
    </template>

    <!-- ── DONE VIEW ──────────────────────────────────────────────────────── -->
    <template v-else-if="isDone && run">
      <ConsoleCounterTiles :summary="run.summary" />

      <UAlert v-if="metricsError" color="warning" variant="subtle" :title="metricsError" />

      <ConsoleWinnerSpotlight v-if="winner" :winner="winner" />

      <!-- Per-predictor metrics table -->
      <div v-if="metrics" class="panel">
        <div class="panel-header"><span class="micro-label">Per-predictor metrics</span></div>
        <div class="overflow-x-auto">
          <table class="w-full text-[11px]">
            <thead>
              <tr class="border-b border-line text-dim uppercase tracking-wide text-[10px]">
                <th class="text-left py-1.5 px-3 font-normal">Predictor</th>
                <th class="text-right py-1.5 px-3 font-normal">MAE (s)</th>
                <th class="text-right py-1.5 px-3 font-normal">RMSE (s)</th>
                <th class="text-right py-1.5 px-3 font-normal">MAPE (%)</th>
                <th class="text-right py-1.5 px-3 font-normal">Bias (s)</th>
                <th class="text-right py-1.5 px-3 font-normal">Count</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="m in metrics.per_predictor"
                :key="m.predictor_id"
                class="border-b border-[#141f1a]"
                :class="winner && m.predictor_id === winner.predictor_id ? 'bg-accent-dim' : ''"
              >
                <td class="py-1 px-3" :class="winner && m.predictor_id === winner.predictor_id ? 'text-accent' : 'text-bright'">
                  {{ m.predictor_id }}
                </td>
                <td class="py-1 px-3 text-right tabular-nums">{{ fmtNum(m.mae) }}</td>
                <td class="py-1 px-3 text-right tabular-nums">{{ fmtNum(m.rmse) }}</td>
                <td class="py-1 px-3 text-right tabular-nums">{{ fmtNum(m.mape) }}</td>
                <td class="py-1 px-3 text-right tabular-nums">{{ fmtNum(m.bias) }}</td>
                <td class="py-1 px-3 text-right tabular-nums text-dim">{{ m.count.toLocaleString() }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Horizon-bucket pivot -->
      <div v-if="metrics && metrics.horizon_buckets.length > 0" class="panel">
        <div class="panel-header"><span class="micro-label">Horizon buckets</span></div>
        <div class="overflow-x-auto">
          <table class="w-full text-[11px]">
            <thead>
              <tr class="border-b border-line text-dim uppercase tracking-wide text-[10px]">
                <th class="text-left py-1.5 px-3 font-normal">Bucket</th>
                <th class="text-left py-1.5 px-3 font-normal">Predictor</th>
                <th class="text-right py-1.5 px-3 font-normal">MAE (s)</th>
                <th class="text-right py-1.5 px-3 font-normal">Bias (s)</th>
                <th class="text-right py-1.5 px-3 font-normal">Count</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(b, i) in metrics.horizon_buckets" :key="`${b.label}-${b.predictor_id}-${i}`" class="border-b border-[#141f1a]">
                <td class="py-1 px-3 text-dim">{{ b.label }}</td>
                <td class="py-1 px-3 text-bright">{{ b.predictor_id }}</td>
                <td class="py-1 px-3 text-right tabular-nums">{{ fmtNum(b.mae) }}</td>
                <td class="py-1 px-3 text-right tabular-nums">{{ fmtNum(b.bias) }}</td>
                <td class="py-1 px-3 text-right tabular-nums text-dim">{{ b.count.toLocaleString() }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Results sample proof -->
      <div v-if="resultsSample.length > 0" class="panel px-3 py-2">
        <span class="micro-label">Result rows (sample)</span>
        <p class="text-xs text-dim mt-1">
          data path verified — {{ resultsSample.length }} rows returned from
          <code class="font-mono">/api/runs/{{ id }}/results?limit=50</code>
        </p>
      </div>

      <!-- Analytics dashboard -->
      <div class="panel">
        <div class="panel-header"><span class="micro-label">Analytics</span></div>
        <div class="p-3">
          <AnalyticsDashboard :run-id="id" />
        </div>
      </div>
    </template>

    <!-- Loading skeleton when run is being fetched for the first time -->
    <template v-else-if="!loadError && !run">
      <div class="space-y-3 animate-pulse">
        <div class="h-24 bg-panel border border-line" />
        <div class="h-48 bg-panel border border-line" />
      </div>
    </template>
  </div>
</template>
