<script setup lang="ts">
useHead({ title: 'telemetry – etaval' })

interface PerRunFeed {
  last_poll_age_s: number
  ok: boolean
}

interface PerRunTelemetry {
  run_id: string
  status: string
  ticks: number
  tick_rate: number
  last_tick_age_s: number | null
  feed: PerRunFeed | null
  predictor_errors: Record<string, number>
}

interface TelemetrySnapshot {
  api_uptime_s: number
  active_runs: number
  per_run: PerRunTelemetry[]
}

interface TelemetryEvent {
  ts: string
  run_id: string | null
  level: string
  source: string
  message: string
}

const config = useRuntimeConfig()
const apiBase = config.public.apiBase as string

const telemetry = ref<TelemetrySnapshot | null>(null)
const events = ref<TelemetryEvent[]>([])
const offline = ref(false)

async function pollTelemetry() {
  try {
    const [t, e] = await Promise.all([
      $fetch<TelemetrySnapshot>('/api/telemetry', { baseURL: apiBase }),
      $fetch<TelemetryEvent[]>('/api/telemetry/events', { baseURL: apiBase, query: { limit: 100 } })
    ])
    telemetry.value = t
    events.value = e
    offline.value = false
  } catch {
    offline.value = true
  }
}

let timer: ReturnType<typeof setInterval> | undefined

onMounted(() => {
  pollTelemetry()
  timer = setInterval(pollTelemetry, 5000)
})

onUnmounted(() => {
  if (timer) clearInterval(timer)
})

function fmtUptime(s: number): string {
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = Math.floor(s % 60)
  return `${h}h ${m}m ${sec}s`
}

function fmtNum(n: number | null | undefined, digits = 1): string {
  if (n === null || n === undefined || !isFinite(n)) return '—'
  return n.toFixed(digits)
}

function eventLevelClass(level: string): string {
  const l = level.toUpperCase()
  if (l === 'ERROR') return 'text-err'
  if (l === 'WARN' || l === 'WARNING') return 'text-warn'
  return 'text-dim'
}

const reversedEvents = computed(() => events.value.slice().reverse())
</script>

<template>
  <div class="flex flex-col lg:flex-row gap-3 p-3 lg:h-[calc(100vh-40px)] lg:overflow-hidden">
    <!-- MAIN: uptime/active + per-run tiles -->
    <div class="flex-1 min-w-0 overflow-y-auto space-y-3">
      <UAlert
        v-if="offline"
        color="error"
        variant="subtle"
        title="Telemetry API unreachable"
        description="Retrying every 5s…"
      />

      <template v-if="telemetry">
        <div class="grid grid-cols-2 gap-px bg-line">
          <div class="tile bg-panel px-3 py-2">
            <div class="tile-l">Api uptime</div>
            <div class="tile-v">{{ fmtUptime(telemetry.api_uptime_s) }}</div>
          </div>
          <div class="tile bg-panel px-3 py-2">
            <div class="tile-l">Active runs</div>
            <div class="tile-v" :class="telemetry.active_runs > 0 ? 'text-accent' : ''">
              {{ telemetry.active_runs }}
            </div>
          </div>
        </div>

        <div v-if="telemetry.per_run.length === 0" class="text-center py-12 text-dim text-xs">
          No active runs.
        </div>

        <div v-for="pr in telemetry.per_run" :key="pr.run_id" class="panel">
          <div class="panel-header">
            <span class="flex items-center gap-2">
              <span :class="pr.status === 'error' ? 'status-led err' : pr.status === 'running' ? 'status-led ok' : 'status-led warn'" />
              <span class="text-bright font-mono text-xs">{{ pr.run_id }}</span>
            </span>
            <span class="micro-label">{{ pr.status }}</span>
          </div>
          <div class="grid grid-cols-2 sm:grid-cols-4 gap-px bg-line">
            <div class="bg-panel px-3 py-2">
              <div class="tile-l">Ticks</div>
              <div class="text-lg text-bright tabular-nums">{{ pr.ticks }}</div>
            </div>
            <div class="bg-panel px-3 py-2">
              <div class="tile-l">Tick rate</div>
              <div class="text-lg text-bright tabular-nums">{{ fmtNum(pr.tick_rate, 2) }}<small class="text-dim text-xs">/s</small></div>
            </div>
            <div class="bg-panel px-3 py-2">
              <div class="tile-l">Last tick age</div>
              <div class="text-lg text-bright tabular-nums">{{ fmtNum(pr.last_tick_age_s) }}<small class="text-dim text-xs">s</small></div>
            </div>
            <div class="bg-panel px-3 py-2">
              <div class="tile-l">Feed</div>
              <div v-if="pr.feed" class="text-lg tabular-nums" :class="pr.feed.ok ? 'text-accent' : 'text-err'">
                {{ pr.feed.ok ? 'OK' : 'FAIL' }}
                <small class="text-dim text-xs">{{ fmtNum(pr.feed.last_poll_age_s) }}s</small>
              </div>
              <div v-else class="text-lg text-dim">—</div>
            </div>
          </div>
          <div
            v-if="Object.keys(pr.predictor_errors).length > 0"
            class="px-3 py-2 border-t border-line flex flex-wrap gap-1.5"
          >
            <span
              v-for="(count, pid) in pr.predictor_errors"
              :key="pid"
              class="chip text-err border-err pointer-events-none"
            >
              {{ pid }}: {{ count }}
            </span>
          </div>
        </div>
      </template>

      <div v-else-if="!offline" class="text-center py-12 text-dim text-xs">
        Loading telemetry&hellip;
      </div>
    </div>

    <!-- RIGHT: sticky event log rail -->
    <div class="panel w-full lg:w-96 shrink-0 flex flex-col lg:h-full">
      <div class="panel-header shrink-0">
        <span class="micro-label">Event log</span>
        <span class="micro-label">{{ events.length }}</span>
      </div>
      <div class="flex-1 overflow-y-auto">
        <div v-if="reversedEvents.length === 0" class="text-xs text-dim text-center py-6">
          No events yet.
        </div>
        <div
          v-for="(ev, i) in reversedEvents"
          :key="`${ev.ts}-${i}`"
          class="flex gap-2 px-3 py-1 text-[10.5px] font-mono whitespace-nowrap overflow-hidden"
        >
          <span class="text-dim shrink-0">{{ ev.ts }}</span>
          <span class="shrink-0 w-10" :class="eventLevelClass(ev.level)">{{ ev.level }}</span>
          <span class="text-main truncate">{{ ev.message }}</span>
        </div>
      </div>
    </div>
  </div>
</template>
