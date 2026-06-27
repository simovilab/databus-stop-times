<script setup lang="ts">
// RouteBreakdown – grouped bar of MAE per route per predictor.
// x = route index, y = MAE. X-axis labels show route IDs.

import { VisXYContainer, VisGroupedBar, VisAxis, VisBulletLegend } from '@unovis/vue'
import type { RouteDatum } from '~/composables/useResultsAnalytics'

const props = defineProps<{
  data: RouteDatum[]
  predictors: string[]
  colorMap: Record<string, string>
}>()

const xAccessor = (d: RouteDatum) => d.routeIdx

const yAccessors = computed(() =>
  props.predictors.map((p) => (d: RouteDatum) => {
    const v = d[p]
    return typeof v === 'number' && isFinite(v) ? v : 0
  })
)

const barColors = computed(() => props.predictors.map((p) => props.colorMap[p] ?? '#999'))

const legendItems = computed(() =>
  props.predictors.map((p) => ({ name: p, color: props.colorMap[p] ?? '#999' }))
)

const hasData = computed(() => props.data.length > 0)

// X-axis label formatter: index → route ID
const xTickFormat = computed(() => {
  const labels = props.data.map((d) => d.routeLabel)
  return (v: number) => labels[Math.round(v)] ?? ''
})
</script>

<template>
  <UCard>
    <template #header>
      <h3 class="text-sm font-semibold text-gray-800 dark:text-gray-200">MAE by Route</h3>
    </template>

    <ClientOnly>
      <div v-if="hasData">
        <VisBulletLegend :items="legendItems" class="mb-3" />
        <VisXYContainer :data="data" :height="220">
          <VisGroupedBar
            :x="xAccessor"
            :y="yAccessors"
            :color="barColors"
            :data-step="1"
            :rounded-corners="2"
          />
          <VisAxis
            type="x"
            label="Route"
            :tick-format="xTickFormat"
            :num-ticks="Math.min(data.length, 10)"
          />
          <VisAxis type="y" label="MAE (s)" />
        </VisXYContainer>
      </div>
      <div v-else class="py-8 text-center text-sm text-gray-400 dark:text-gray-600">
        No data for selected filters.
      </div>

      <template #fallback>
        <div class="h-[220px] bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
      </template>
    </ClientOnly>
  </UCard>
</template>
