<script setup lang="ts">
// ErrorScatter – scatter plot of error_s (y) vs horizon_s (x), colored by predictor.
// Clicking a point selects its (trip_key, stop_id) to drive the ConvergenceChart.
// Capped at MAX_POINTS for performance when the dataset is very large.

import {
  VisXYContainer,
  VisScatter,
  VisAxis,
  VisBulletLegend,
  VisScatterSelectors,
} from '@unovis/vue'
import type { ResultRow } from '~/types/api'

const MAX_POINTS = 3000

const props = defineProps<{
  data: ResultRow[]
  predictors: string[]
  colorMap: Record<string, string>
}>()

const emit = defineEmits<{
  'select-trip-stop': [trip_key: string, stop_id: string]
}>()

// Cap rows for performance
const displayData = computed<ResultRow[]>(() =>
  props.data.length > MAX_POINTS ? props.data.slice(0, MAX_POINTS) : props.data
)

const isCapped = computed(() => props.data.length > MAX_POINTS)

const colorFn = (d: ResultRow) => props.colorMap[d.predictor_id] ?? '#999'

const xAccessor = (d: ResultRow) => d.horizon_s
const yAccessor = (d: ResultRow) => d.error_s

// Click on a point → select that trip+stop
const scatterEvents = computed(() => ({
  [VisScatterSelectors.point]: {
    click: (d: unknown) => {
      const row = d as ResultRow
      if (row?.trip_key && row?.stop_id) {
        emit('select-trip-stop', row.trip_key, row.stop_id)
      }
    },
  },
}))

const legendItems = computed(() =>
  props.predictors.map((p) => ({ name: p, color: props.colorMap[p] ?? '#999' }))
)

const hasData = computed(() => displayData.value.length > 0)

// Zero-line annotation via yDomain clamping: keep y-axis centred on 0
const yDomain = computed((): [number, number] | undefined => {
  if (!hasData.value) return undefined
  const absMax = Math.max(...displayData.value.map((d) => Math.abs(d.error_s)), 1)
  return [-absMax, absMax]
})
</script>

<template>
  <div class="panel">
    <div class="panel-header">
      <span class="micro-label">Error vs horizon</span>
      <span
        v-if="isCapped"
        class="text-[10px] text-warn"
      >
        first {{ MAX_POINTS.toLocaleString() }} of {{ data.length.toLocaleString() }}
      </span>
    </div>

    <div class="p-3">
      <ClientOnly>
        <div v-if="hasData">
          <VisBulletLegend :items="legendItems" class="mb-3" />
          <VisXYContainer
            :data="displayData"
            :height="280"
            :y-domain="yDomain"
          >
            <VisScatter
              :x="xAccessor"
              :y="yAccessor"
              :color="colorFn"
              :size="4"
              cursor="pointer"
              :events="scatterEvents"
            />
            <VisAxis type="x" label="Horizon (s)" />
            <VisAxis type="y" label="Error (s)  [+ late / − early]" />
          </VisXYContainer>
          <p class="mt-1 text-xs text-dim text-center">Click a point to see convergence for that trip/stop.</p>
        </div>
        <div v-else class="py-8 text-center text-xs text-dim">
          No data for selected filters.
        </div>

        <template #fallback>
          <div class="h-[280px] bg-panel animate-pulse" />
        </template>
      </ClientOnly>
    </div>
  </div>
</template>
