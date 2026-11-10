<script setup lang="ts">
// RunHorizonBuckets – whole-run MAE per horizon bucket: a line per predictor,
// then the exact numbers as a matrix tinted by error (darker = worse) so the
// table reads at a glance too.

import type { HorizonBucket } from '~/types/api'

const props = defineProps<{
  buckets: HorizonBucket[]
}>()

const predictors = computed(() => [...new Set(props.buckets.map((b) => b.predictor_id))])
const labels = computed(() => [...new Set(props.buckets.map((b) => b.label))])
const colorMap = computed(() => buildColorMap(predictors.value))
const series = computed(() => seriesKeys(predictors.value, colorMap.value))
const gapSeries = computed(() =>
  series.value.keys.map((k) => ({ key: k, name: series.value.categories[k]!.name, color: series.value.categories[k]!.color }))
)

const lookup = computed(() => {
  const m = new Map<string, HorizonBucket>()
  for (const b of props.buckets) m.set(`${b.label}|${b.predictor_id}`, b)
  return m
})

const rows = computed(() =>
  labels.value.map((label) => {
    const row: Record<string, string | number | null> = { label }
    for (const p of predictors.value) {
      const b = lookup.value.get(`${label}|${p}`)
      row[series.value.byId[p] as string] = b && Number.isFinite(b.mae) ? Math.round(b.mae * 10) / 10 : null
    }
    return row
  })
)

const maxMae = computed(() => Math.max(1, ...props.buckets.map((b) => (Number.isFinite(b.mae) ? b.mae : 0))))

// Single-hue magnitude ramp on the panel: transparent at 0, strongest at the run's worst cell.
function tint(mae: number | undefined): string {
  if (mae === undefined || !Number.isFinite(mae)) return 'transparent'
  const a = 0.06 + 0.5 * (mae / maxMae.value)
  return `rgba(217, 89, 38, ${a.toFixed(3)})`
}

// Best predictor per bucket, for the bold marker.
const bestIn = computed(() => {
  const out: Record<string, string> = {}
  for (const label of labels.value) {
    let best: HorizonBucket | undefined
    for (const p of predictors.value) {
      const b = lookup.value.get(`${label}|${p}`)
      if (b && Number.isFinite(b.mae) && (!best || b.mae < best.mae)) best = b
    }
    if (best) out[label] = best.predictor_id
  }
  return out
})

const showTable = ref(false)
</script>

<template>
  <AnalyticsChartCard title="Error by horizon" meta="whole run · MAE" :empty="buckets.length === 0">
    <template #lede>
      Error grows the further ahead a prediction looks. The steeper the line, the faster that predictor degrades.
    </template>
    <template #actions>
      <UButton
        size="xs"
        variant="ghost"
        color="neutral"
        :icon="showTable ? 'i-lucide-chart-line' : 'i-lucide-table-2'"
        :label="showTable ? 'Chart' : 'Table'"
        @click="showTable = !showTable"
      />
    </template>

    <AnalyticsGapLineChart
      v-if="!showTable"
      :rows="rows"
      x-key="label"
      :series="gapSeries"
      :height="260"
      :y-formatter="(v: number) => `${Math.round(v)}s`"
      y-label="MAE"
      x-label="prediction horizon"
      :tooltip-title="(r) => `${r.label} ahead`"
      aria-label="MAE by horizon bucket for the whole run"
    />

    <div v-else class="overflow-x-auto">
      <table class="w-full text-[11px] tabular-nums">
        <thead>
          <tr class="text-dim uppercase tracking-[0.1em] text-[9.5px]">
            <th class="text-left font-normal py-2 px-2">Predictor</th>
            <th v-for="l in labels" :key="l" class="text-right font-normal py-2 px-2 whitespace-nowrap">{{ l }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="p in predictors" :key="p" class="border-t border-[#141f1a]">
            <td class="py-1.5 px-2 whitespace-nowrap">
              <span class="inline-flex items-center gap-2">
                <span class="size-2 flex-none" :style="{ background: colorMap[p] }" />
                <span class="text-main" :title="p">{{ predictorLabel(p) }}</span>
              </span>
            </td>
            <td
              v-for="l in labels"
              :key="l"
              class="py-1.5 px-2 text-right"
              :style="{ background: tint(lookup.get(`${l}|${p}`)?.mae) }"
              :title="lookup.get(`${l}|${p}`) ? `bias ${lookup.get(`${l}|${p}`)!.bias.toFixed(1)}s · n ${lookup.get(`${l}|${p}`)!.count.toLocaleString()}` : 'no predictions'"
            >
              <span v-if="lookup.get(`${l}|${p}`)" :class="bestIn[l] === p ? 'text-bright font-semibold' : 'text-main'">
                {{ lookup.get(`${l}|${p}`)!.mae.toFixed(1) }}
              </span>
              <span v-else class="text-dim">—</span>
            </td>
          </tr>
        </tbody>
      </table>
      <p class="text-[10px] text-dim mt-2 px-2">Seconds. Bold = best in that horizon. Hover a cell for bias and count.</p>
    </div>
  </AnalyticsChartCard>
</template>
