<script setup lang="ts">
// ErrorDistribution – grouped bar histogram of error_s per predictor.
// x = bin centre (seconds), y = count per predictor. Bins are in [-300s, +300s].

import { VisXYContainer, VisGroupedBar, VisAxis, VisBulletLegend } from '@unovis/vue'
import type { HistBin } from '~/composables/useResultsAnalytics'

const props = defineProps<{
  data: HistBin[]
  predictors: string[]
  colorMap: Record<string, string>
}>()

const xAccessor = (d: HistBin) => d.binCenter

const yAccessors = computed(() =>
  props.predictors.map((p) => (d: HistBin) => {
    const v = d[p]
    return typeof v === 'number' ? v : 0
  })
)

const barColors = computed(() => props.predictors.map((p) => props.colorMap[p] ?? '#999'))

const legendItems = computed(() =>
  props.predictors.map((p) => ({ name: p, color: props.colorMap[p] ?? '#999' }))
)

const hasData = computed(() => props.data.some((d) => props.predictors.some((p) => (d[p] as number) > 0)))

// x-axis tick format (show seconds with s suffix)
const xTickFormat = (v: number) => `${v > 0 ? '+' : ''}${v}s`
</script>

<template>
  <UCard>
    <template #header>
      <h3 class="text-sm font-semibold text-gray-800 dark:text-gray-200">Error Distribution</h3>
      <p class="text-xs text-gray-400 mt-0.5">+ = late, − = early</p>
    </template>

    <ClientOnly>
      <div v-if="hasData">
        <VisBulletLegend :items="legendItems" class="mb-3" />
        <VisXYContainer :data="data" :height="220">
          <VisGroupedBar
            :x="xAccessor"
            :y="yAccessors"
            :color="barColors"
            :data-step="60"
            :rounded-corners="2"
          />
          <VisAxis type="x" label="Error (s)" :tick-format="xTickFormat" />
          <VisAxis type="y" label="Count" />
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
