<script setup lang="ts">
// RunMaeForest – the run's headline result as an interval plot: one row per
// predictor, MAE as a dot with its bootstrap 95% CI as a bar, on one shared
// axis. Overlapping intervals mean the difference may be noise; the
// significance panel below gives the formal test.

import type { PerPredictorMetric } from '~/types/api'

const props = defineProps<{
  metrics: PerPredictorMetric[]
}>()

const colorMap = computed(() => buildColorMap(props.metrics.map((m) => m.predictor_id)))
const ranked = computed(() => [...props.metrics].sort((a, b) => a.mae - b.mae))
const best = computed(() => ranked.value[0])

// Shared axis: 0 … nice ceiling over the widest CI or MAE.
const axisMax = computed(() => {
  const hi = Math.max(1, ...props.metrics.map((m) => (m.mae_ci && Number.isFinite(m.mae_ci.hi) ? m.mae_ci.hi : m.mae)))
  const step = hi > 300 ? 60 : hi > 120 ? 30 : hi > 40 ? 10 : 5
  return Math.ceil((hi * 1.08) / step) * step
})
const ticks = computed(() => {
  const n = 4
  return Array.from({ length: n + 1 }, (_, i) => Math.round((axisMax.value / n) * i))
})
const pct = (v: number) => `${Math.min(100, Math.max(0, (v / axisMax.value) * 100))}%`

const hasCi = computed(() => props.metrics.some((m) => m.mae_ci))

function fmt1(n: number | undefined | null): string {
  return n === undefined || n === null || !Number.isFinite(n) ? '—' : n.toFixed(1)
}
function fmtMs(sec: number | undefined | null): string {
  return sec === undefined || sec === null || !Number.isFinite(sec) ? '—' : `${Math.round(sec * 1000).toLocaleString()}`
}

const runnerUp = computed(() => ranked.value[1])
const separated = computed(() => {
  const a = best.value?.mae_ci
  const b = runnerUp.value?.mae_ci
  return a && b ? a.hi < b.lo : null
})
</script>

<template>
  <AnalyticsChartCard
    title="Mean absolute error"
    :meta="hasCi ? 'dot = MAE · bar = 95% CI' : 'lower is better'"
    :empty="metrics.length === 0"
    :height="160"
  >
    <template v-if="best" #lede>
      <b>{{ predictorLabel(best.predictor_id) }}</b> leads at <b>{{ fmtSeconds(best.mae) }}</b><template v-if="runnerUp">, {{ fmtSeconds(runnerUp.mae - best.mae) }} ahead of {{ predictorLabel(runnerUp.predictor_id) }}<template v-if="separated === true">, and the intervals don't overlap</template><template v-else-if="separated === false">, but the intervals overlap</template></template>.
    </template>

    <div class="overflow-x-auto">
      <div class="min-w-[640px]">
        <!-- header -->
        <div class="grid grid-cols-[minmax(150px,1.1fr)_minmax(240px,3fr)_repeat(4,minmax(64px,0.6fr))] gap-x-4 px-2 pb-2 text-[9.5px] uppercase tracking-[0.1em] text-dim">
          <span>Predictor</span>
          <span class="relative h-3">
            <span
              v-for="t in ticks"
              :key="t"
              class="absolute -translate-x-1/2 tabular-nums normal-case tracking-normal"
              :style="{ left: pct(t) }"
            >{{ t }}s</span>
          </span>
          <span class="text-right">RMSE</span>
          <span class="text-right">Bias</span>
          <span class="text-right">P95 ms</span>
          <span class="text-right">n</span>
        </div>

        <!-- rows -->
        <div
          v-for="m in ranked"
          :key="m.predictor_id"
          class="grid grid-cols-[minmax(150px,1.1fr)_minmax(240px,3fr)_repeat(4,minmax(64px,0.6fr))] gap-x-4 items-center px-2 py-2 border-t border-[#141f1a] text-[11px]"
          :class="m.predictor_id === best?.predictor_id ? 'bg-[#0f1a15]' : ''"
        >
          <span class="flex items-center gap-2 min-w-0">
            <span class="size-2 flex-none" :style="{ background: colorMap[m.predictor_id] }" />
            <span class="truncate" :class="m.predictor_id === best?.predictor_id ? 'text-bright' : 'text-main'" :title="m.predictor_id">
              {{ predictorLabel(m.predictor_id) }}
            </span>
          </span>

          <span class="relative h-5" :title="m.mae_ci ? `MAE ${fmt1(m.mae)}s · 95% CI ${fmt1(m.mae_ci.lo)}–${fmt1(m.mae_ci.hi)}s` : `MAE ${fmt1(m.mae)}s`">
            <!-- gridlines -->
            <span v-for="t in ticks" :key="t" class="absolute inset-y-0 w-px bg-[#16221d]" :style="{ left: pct(t) }" />
            <!-- CI bar -->
            <span
              v-if="m.mae_ci"
              class="absolute top-1/2 -translate-y-1/2 h-[6px] opacity-45"
              :style="{ left: pct(m.mae_ci.lo), width: `calc(${pct(m.mae_ci.hi)} - ${pct(m.mae_ci.lo)})`, background: colorMap[m.predictor_id] }"
            />
            <!-- MAE dot -->
            <span
              class="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 size-[11px] rounded-full! border-2"
              :style="{ left: pct(m.mae), background: colorMap[m.predictor_id], borderColor: 'var(--color-panel)' }"
            />
            <span
              class="absolute top-1/2 -translate-y-1/2 pl-2.5 tabular-nums text-[10px]"
              :class="m.predictor_id === best?.predictor_id ? 'text-bright' : 'text-main'"
              :style="{ left: m.mae_ci ? pct(m.mae_ci.hi) : pct(m.mae) }"
            >{{ fmt1(m.mae) }}s</span>
          </span>

          <span class="text-right tabular-nums text-main">{{ fmt1(m.rmse) }}</span>
          <span class="text-right tabular-nums" :class="Math.abs(m.bias) > 30 ? 'text-warn' : 'text-main'">{{ m.bias > 0 ? '+' : '' }}{{ fmt1(m.bias) }}</span>
          <span class="text-right tabular-nums text-dim">{{ fmtMs(m.latency?.p95_s) }}</span>
          <span class="text-right tabular-nums text-dim">{{ m.count.toLocaleString() }}</span>
        </div>
      </div>
    </div>
    <template #foot>
      Seconds. Bias &gt; 0 means predicted arrivals are late on average.<template v-if="!hasCi"> Confidence intervals appear once the run's statistics are computed.</template>
    </template>
  </AnalyticsChartCard>
</template>
