<script setup lang="ts">
// ConvergenceChart – for one trip/stop, |error| as the bus approaches.
// Read right to left: far out on the right, arrival at the left. A good
// predictor's line falls towards zero as the horizon shrinks.

import type { ConvDatum } from '~/composables/useResultsAnalytics'

const props = defineProps<{
  data: ConvDatum[]
  predictors: string[]
  colorMap: Record<string, string>
  selectedTripStop: { trip_key: string; stop_id: string } | null
}>()

const emit = defineEmits<{
  clear: []
}>()

const series = computed(() => seriesKeys(props.predictors, props.colorMap))
const gapSeries = computed(() =>
  series.value.keys.map((k) => ({ key: k, name: series.value.categories[k]!.name, color: series.value.categories[k]!.color }))
)

// Far horizons first so time runs left → right towards arrival.
const rows = computed(() =>
  [...props.data]
    .sort((a, b) => b.horizonBin - a.horizonBin)
    .map((d) => {
      const row: Record<string, string | number | null> = { label: fmtSeconds(d.horizonBin), h: d.horizonBin }
      for (const p of props.predictors) {
        const v = d[p]
        row[series.value.byId[p] as string] = typeof v === 'number' && Number.isFinite(v) ? Math.round(v) : null
      }
      return row
    })
)

const hasData = computed(() => rows.value.length > 1)
</script>

<template>
  <AnalyticsChartCard
    title="Prediction convergence"
    :meta="selectedTripStop ? `stop ${selectedTripStop.stop_id}` : undefined"
    :empty="!hasData"
    :height="280"
  >
    <template v-if="selectedTripStop" #lede>
      How each predictor's error shrank as trip <b>{{ selectedTripStop.trip_key }}</b> approached
      the stop. Time runs left to right.
    </template>
    <template #actions>
      <UButton
        v-if="selectedTripStop"
        size="xs"
        variant="ghost"
        color="neutral"
        icon="i-lucide-x"
        aria-label="Clear selection"
        @click="emit('clear')"
      />
    </template>
    <template #empty>
      <div class="flex flex-col items-center gap-2 px-6">
        <UIcon name="i-lucide-mouse-pointer-click" class="size-5 text-dim" />
        <span v-if="selectedTripStop">Not enough predictions for this trip/stop under the current filters.</span>
        <span v-else>Click a dot in <b class="text-main">Error vs horizon</b> to trace one trip/stop here.</span>
      </div>
    </template>
    <AnalyticsGapLineChart
      :rows="rows"
      x-key="label"
      :series="gapSeries"
      :height="260"
      connect-nulls
      :y-formatter="(v: number) => `${Math.round(v)}s`"
      y-label="|error|"
      x-label="time to arrival"
      :tooltip-title="(r) => `${r.label} before arrival`"
      aria-label="Absolute error as the vehicle approaches the stop"
    />
  </AnalyticsChartCard>
</template>
