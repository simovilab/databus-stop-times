<script setup lang="ts">
import type { RunInfo } from '~/types/api'

useHead({ title: 'leaderboard – etaval' })

const api = useApi()
const router = useRouter()

const runs = ref<RunInfo[]>([])
const loading = ref(false)
const loadError = ref<string | null>(null)

async function loadRuns() {
  loading.value = true
  loadError.value = null
  try {
    runs.value = await api.getRuns()
    // Most recent first
    runs.value = [...runs.value].sort((a, b) => {
      const ta = a.started_at ?? ''
      const tb = b.started_at ?? ''
      return tb.localeCompare(ta)
    })
  } catch {
    loadError.value = 'Could not load runs. Is the backend running?'
  } finally {
    loading.value = false
  }
}

onMounted(loadRuns)

function fmtDate(iso: string | null): string {
  if (!iso) return '—'
  return new Date(iso).toLocaleString()
}

function fmtDuration(run: RunInfo): string {
  if (!run.started_at) return '—'
  const start = new Date(run.started_at).getTime()
  const end = run.ended_at ? new Date(run.ended_at).getTime() : Date.now()
  const secs = Math.max(0, Math.round((end - start) / 1000))
  const m = Math.floor(secs / 60)
  const s = secs % 60
  return `${m}m ${s}s`
}

function shortId(id: string): string {
  return id.length > 22 ? `${id.slice(0, 22)}…` : id
}

function ledClass(status: string): string {
  if (status === 'error') return 'status-led err'
  if (status === 'running' || status === 'pending') return 'status-led warn'
  if (status === 'done') return 'status-led ok'
  return 'status-led idle'
}

function goTo(id: string) {
  router.push(`/runs/${id}`)
}
</script>

<template>
  <div class="p-3">
    <div class="flex items-center justify-between mb-3">
      <div class="sec-h flex-1">Run history</div>
      <UButton
        label="Refresh"
        variant="outline"
        icon="i-heroicons-arrow-path"
        :loading="loading"
        size="xs"
        class="ml-3"
        @click="loadRuns"
      />
    </div>

    <UAlert v-if="loadError" color="error" variant="subtle" :title="loadError" class="mb-4" />

    <!-- Empty state -->
    <div v-if="!loading && !loadError && runs.length === 0" class="text-center py-16 text-dim">
      <p class="text-sm">No runs yet.</p>
      <p class="text-xs mt-2">
        <NuxtLink to="/" class="text-accent hover:underline">start one →</NuxtLink>
      </p>
    </div>

    <!-- Bento cards -->
    <div v-if="runs.length > 0" class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
      <div
        v-for="run in runs"
        :key="run.id"
        class="panel p-3 cursor-pointer hover:border-accent transition-colors"
        @click="goTo(run.id)"
      >
        <div class="flex items-center gap-2 mb-2">
          <span :class="ledClass(run.status)" />
          <span class="text-bright font-mono text-xs truncate" :title="run.id">{{ shortId(run.id) }}</span>
          <span class="ml-auto text-[10px] uppercase tracking-wide text-dim">{{ run.status }}</span>
        </div>

        <div class="kv-row"><span class="k">Source</span><span class="v">{{ run.params.source }}</span></div>
        <div class="kv-row"><span class="k">Duration</span><span class="v">{{ fmtDuration(run) }}</span></div>
        <div class="kv-row"><span class="k">Started</span><span class="v">{{ fmtDate(run.started_at) }}</span></div>

        <div class="flex flex-wrap gap-1 mt-2 mb-2">
          <span v-for="p in run.params.predictors" :key="p" class="chip pointer-events-none">{{ p }}</span>
        </div>

        <div class="grid grid-cols-4 gap-px bg-line mt-2">
          <div class="bg-panel px-1.5 py-1">
            <div class="tile-l">Ticks</div>
            <div class="text-sm text-bright tabular-nums">{{ run.summary.ticks }}</div>
          </div>
          <div class="bg-panel px-1.5 py-1">
            <div class="tile-l">Samples</div>
            <div class="text-sm text-bright tabular-nums">{{ run.summary.samples_saved }}</div>
          </div>
          <div class="bg-panel px-1.5 py-1">
            <div class="tile-l">Arrivals</div>
            <div class="text-sm text-bright tabular-nums">{{ run.summary.arrivals_saved }}</div>
          </div>
          <div class="bg-panel px-1.5 py-1">
            <div class="tile-l">Results</div>
            <div class="text-sm text-bright tabular-nums">{{ run.summary.results_saved }}</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
