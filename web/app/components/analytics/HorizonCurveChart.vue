<script setup lang="ts">
// HorizonCurveChart – headline multi-line chart of MAE per horizon bucket per predictor.
// x = bucket order (0–3), y = MAE (s). One line per active predictor.

import { VisXYContainer, VisLine, VisAxis, VisBulletLegend, VisCrosshair } from '@unovis/vue'
import type { BucketDatum } from '~/composables/useResultsAnalytics'

const props = defineProps<{
  data: BucketDatum[]
  predictors: string[]
  colorMap: Record<string, string>
}>()

// Y accessors – one per predictor
const yAccessors = computed(() =>
  props.predictors.map(
    (p) => (d: BucketDatum) => {
      const v = d[p]
      return typeof v === 'number' && isFinite(v) ? v : undefined
    }
  )
)

// Colors array matching predictor order
const lineColors = computed(() => props.predictors.map((p) => props.colorMap[p] ?? '#999'))

// X accessor
const xAccessor = (d: BucketDatum) => d.bucketIdx

// Tick formatter for x-axis (show bucket labels)
const xTickFormat = (v: number) => {
  const labels = ['<2m', '2-5m', '5-10m', '>10m']
  return labels[Math.round(v)] ?? ''
}

// Crosshair y accessors
const crosshairY = computed(() =>
  props.predictors.map((p) => (d: BucketDatum) => {
    const v = d[p]
    return typeof v === 'number' && isFinite(v) ? v : undefined
  })
)

// Legend items for VisBulletLegend
const legendItems = computed(() =>
  props.predictors.map((p) => ({ name: p, color: props.colorMap[p] ?? '#999' }))
)

const hasData = computed(() => props.data.some((d) => props.predictors.some((p) => typeof d[p] === 'number' && isFinite(d[p] as number))))
</script>

<template>
  <UCard>
    <template #header>
      <h3 class="text-sm font-semibold text-gray-800 dark:text-gray-200">MAE by Horizon Bucket</h3>
    </template>

    <ClientOnly>
      <div v-if="hasData">
        <VisBulletLegend :items="legendItems" class="mb-3" />
        <VisXYContainer :data="data" :height="220" :x-domain="[-0.5, 3.5]">
          <VisLine
            v-for="(yFn, i) in yAccessors"
            :key="predictors[i]"
            :x="xAccessor"
            :y="yFn"
            :color="lineColors[i]"
            :fallback-value="undefined"
            :line-width="2"
            :highlight-on-hover="true"
          />
          <VisAxis type="x" :tick-format="xTickFormat" label="Horizon" />
          <VisAxis type="y" label="MAE (s)" />
          <VisCrosshair :x="xAccessor" :y="crosshairY" :color="lineColors" />
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
