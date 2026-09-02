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
  <div class="panel">
    <div class="panel-header">
      <span class="micro-label">Error distribution</span>
      <span class="micro-label">+ late / − early</span>
    </div>
    <div class="p-3">
      <ClientOnly>
        <div v-if="hasData">
          <VisBulletLegend :items="legendItems" class="mb-3" />
          <VisXYContainer :data="data" :height="220">
            <VisGroupedBar
              :x="xAccessor"
              :y="yAccessors"
              :color="barColors"
              :data-step="60"
              :rounded-corners="0"
            />
            <VisAxis type="x" label="Error (s)" :tick-format="xTickFormat" />
            <VisAxis type="y" label="Count" />
          </VisXYContainer>
        </div>
        <div v-else class="py-8 text-center text-xs text-dim">
          No data for selected filters.
        </div>

        <template #fallback>
          <div class="h-[220px] bg-panel animate-pulse" />
        </template>
      </ClientOnly>
    </div>
  </div>
</template>
