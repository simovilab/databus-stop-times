<script setup lang="ts">
import type { RunInfo, RunMetrics, ResultRow } from '~/types/api'

const route = useRoute()
const id = route.params.id as string

useHead({ title: `Run ${id} – etaval` })

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

const statusColors: Record<string, string> = {
  pending: 'yellow',
  running: 'blue',
  done: 'green',
  error: 'red'
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

// ── Metric table columns ──────────────────────────────────────────────────────
const metricColumns = [
  { accessorKey: 'predictor_id', header: 'Predictor' },
  { accessorKey: 'mae', header: 'MAE (s)' },
  { accessorKey: 'rmse', header: 'RMSE (s)' },
  { accessorKey: 'mape', header: 'MAPE (%)' },
  { accessorKey: 'bias', header: 'Bias (s)' },
  { accessorKey: 'count', header: 'Count' }
]

// ── Live results feed columns ─────────────────────────────────────────────────
const liveResultColumns = [
  { accessorKey: 'predictor_id', header: 'Predictor' },
  { accessorKey: 'stop_id', header: 'Stop' },
  { accessorKey: 'error_s', header: 'Error (s)' },
  { accessorKey: 'horizon_s', header: 'Horizon (s)' }
]

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
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-start justify-between gap-4">
      <div>
        <div class="flex items-center gap-3 mb-1">
          <h1 class="text-xl font-bold text-gray-900 dark:text-white font-mono truncate max-w-xs" :title="id">
            {{ id }}
          </h1>
          <UBadge
            v-if="run || stream.connected.value"
            :color="(statusColors[displayStatus] as 'yellow' | 'blue' | 'green' | 'red') ?? 'gray'"
            variant="subtle"
            :label="displayStatus"
          />
          <UBadge v-if="stream.connected.value" color="blue" variant="solid" label="● live" />
        </div>
        <p v-if="run" class="text-sm text-gray-500">
          Source: <strong>{{ run.params.source }}</strong> ·
          Predictors: <strong>{{ run.params.predictors.join(', ') }}</strong> ·
          Duration: <strong>{{ run.params.duration }}</strong>
          <span v-if="run.started_at"> · Started: {{ fmtDate(run.started_at) }}</span>
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
      <!-- Live counters -->
      <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <UCard>
          <div class="text-center">
            <p class="text-3xl font-bold tabular-nums">{{ displaySummary.ticks }}</p>
            <p class="text-xs text-gray-500 mt-1">Ticks</p>
          </div>
        </UCard>
        <UCard>
          <div class="text-center">
            <p class="text-3xl font-bold tabular-nums">{{ displaySummary.samples_saved }}</p>
            <p class="text-xs text-gray-500 mt-1">Samples saved</p>
          </div>
        </UCard>
        <UCard>
          <div class="text-center">
            <p class="text-3xl font-bold tabular-nums">{{ displaySummary.arrivals_saved }}</p>
            <p class="text-xs text-gray-500 mt-1">Arrivals saved</p>
          </div>
        </UCard>
        <UCard>
          <div class="text-center">
            <p class="text-3xl font-bold tabular-nums">{{ displaySummary.results_saved }}</p>
            <p class="text-xs text-gray-500 mt-1">Results saved</p>
          </div>
        </UCard>
      </div>

      <!-- Vehicles tracked -->
      <UCard>
        <div class="flex items-center gap-2">
          <span class="text-sm text-gray-600 dark:text-gray-400">Vehicles tracked this tick:</span>
          <span class="font-bold tabular-nums">{{ stream.vehicles.value.length }}</span>
        </div>
      </UCard>

      <!-- Live results feed -->
      <UCard>
        <template #header>
          <h2 class="font-semibold text-gray-800 dark:text-gray-200">
            Live result feed
            <span class="text-xs font-normal text-gray-400 ml-2">(last {{ stream.latestResults.value.length }} entries)</span>
          </h2>
        </template>

        <div v-if="stream.latestResults.value.length === 0" class="text-sm text-gray-500 py-4 text-center">
          Waiting for first results…
        </div>
        <UTable
          v-else
          :data="stream.latestResults.value.slice().reverse().slice(0, 20)"
          :columns="liveResultColumns"
        >
          <template #error_s-cell="{ row }">
            <span :class="Math.abs(row.original.error_s) > 60 ? 'text-red-500' : ''">
              {{ fmtNum(row.original.error_s) }}
            </span>
          </template>
        </UTable>
      </UCard>

      <!-- Stream connection error -->
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
      <!-- Final summary -->
      <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <UCard>
          <div class="text-center">
            <p class="text-3xl font-bold tabular-nums">{{ run.summary.ticks }}</p>
            <p class="text-xs text-gray-500 mt-1">Ticks</p>
          </div>
        </UCard>
        <UCard>
          <div class="text-center">
            <p class="text-3xl font-bold tabular-nums">{{ run.summary.samples_saved }}</p>
            <p class="text-xs text-gray-500 mt-1">Samples saved</p>
          </div>
        </UCard>
        <UCard>
          <div class="text-center">
            <p class="text-3xl font-bold tabular-nums">{{ run.summary.arrivals_saved }}</p>
            <p class="text-xs text-gray-500 mt-1">Arrivals saved</p>
          </div>
        </UCard>
        <UCard>
          <div class="text-center">
            <p class="text-3xl font-bold tabular-nums">{{ run.summary.results_saved }}</p>
            <p class="text-xs text-gray-500 mt-1">Results saved</p>
          </div>
        </UCard>
      </div>

      <UAlert
        v-if="metricsError"
        color="warning"
        variant="subtle"
        :title="metricsError"
      />

      <!-- Per-predictor metrics table -->
      <UCard v-if="metrics">
        <template #header>
          <h2 class="font-semibold text-gray-800 dark:text-gray-200">Per-predictor metrics</h2>
        </template>
        <UTable
          :data="metrics.per_predictor"
          :columns="metricColumns"
        >
          <template #mae-cell="{ row }">{{ fmtNum(row.original.mae) }}</template>
          <template #rmse-cell="{ row }">{{ fmtNum(row.original.rmse) }}</template>
          <template #mape-cell="{ row }">{{ fmtNum(row.original.mape) }}</template>
          <template #bias-cell="{ row }">{{ fmtNum(row.original.bias) }}</template>
        </UTable>
      </UCard>

      <!-- Results sample proof -->
      <UCard v-if="resultsSample.length > 0">
        <template #header>
          <h2 class="font-semibold text-gray-800 dark:text-gray-200">
            Result rows (sample)
            <span class="text-xs font-normal text-gray-400 ml-2">{{ resultsSample.length }} rows fetched</span>
          </h2>
        </template>
        <p class="text-sm text-gray-500">Data path verified — {{ resultsSample.length }} rows returned from <code class="font-mono">/api/runs/{{ id }}/results?limit=50</code>.</p>
      </UCard>

      <!-- ════════════════════════════════════════════════════════════════════
           C2: Analytics dashboard placeholder
           @unovis charts (XY plots, scatter, horizon buckets, convergence,
           head-to-head) will mount here in the next phase.
           ════════════════════════════════════════════════════════════════════ -->
      <UCard class="border-2 border-dashed border-gray-300 dark:border-gray-700">
        <template #header>
          <h2 class="font-semibold text-gray-500 dark:text-gray-400">
            Analytics
            <span class="ml-2 text-xs font-normal text-gray-400">(C2 – coming next)</span>
          </h2>
        </template>
        <!-- C2: analytics dashboard goes here -->
        <div class="py-12 text-center text-gray-400 dark:text-gray-600">
          <p class="text-lg">📊 @unovis charts will render here in Phase C2</p>
          <p class="text-sm mt-2">
            Planned: horizon-bucket bars · convergence lines · head-to-head scatter · error distributions
          </p>
        </div>
      </UCard>
    </template>

    <!-- Loading skeleton when run is being fetched for the first time -->
    <template v-else-if="!loadError && !run">
      <div class="space-y-4 animate-pulse">
        <div class="h-24 bg-gray-200 dark:bg-gray-800 rounded-lg" />
        <div class="h-48 bg-gray-200 dark:bg-gray-800 rounded-lg" />
      </div>
    </template>
  </div>
</template>
