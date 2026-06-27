<script setup lang="ts">
// ConvergenceChart – shows |error_s| vs horizon_s for a selected (trip, stop).
// Multiple lines, one per predictor. Empty state when no trip/stop is selected.

import { VisXYContainer, VisLine, VisAxis, VisBulletLegend } from '@unovis/vue'
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

const hasSelection = computed(() => !!props.selectedTripStop)
const hasData = computed(() => props.data.length > 1)

const xAccessor = (d: ConvDatum) => d.horizonBin

const yAccessors = computed(() =>
  props.predictors.map((p) => (d: ConvDatum) => {
    const v = d[p]
    return typeof v === 'number' && isFinite(v) ? v : undefined
  })
)

const lineColors = computed(() => props.predictors.map((p) => props.colorMap[p] ?? '#999'))

const legendItems = computed(() =>
  props.predictors.map((p) => ({ name: p, color: props.colorMap[p] ?? '#999' }))
)

const selectionLabel = computed(() =>
  props.selectedTripStop
    ? `Trip: ${props.selectedTripStop.trip_key} / Stop: ${props.selectedTripStop.stop_id}`
    : ''
)
</script>

<template>
  <UCard>
    <template #header>
      <div class="flex items-center justify-between gap-2 min-w-0">
        <div class="min-w-0">
          <h3 class="text-sm font-semibold text-gray-800 dark:text-gray-200">Prediction Convergence</h3>
          <p v-if="hasSelection" class="text-xs text-gray-400 truncate mt-0.5">{{ selectionLabel }}</p>
        </div>
        <UButton
          v-if="hasSelection"
          size="xs"
          variant="ghost"
          icon="i-heroicons-x-mark"
          label="Clear"
          @click="emit('clear')"
        />
      </div>
    </template>

    <ClientOnly>
      <!-- No selection yet -->
      <div v-if="!hasSelection" class="py-8 text-center text-sm text-gray-400 dark:text-gray-600">
        Click a point in the Error Scatter to see convergence for that trip / stop.
      </div>

      <!-- Selection but no rows survived filters -->
      <div v-else-if="!hasData" class="py-8 text-center text-sm text-gray-400 dark:text-gray-600">
        No predictions found for this trip/stop under the current filters.
      </div>

      <!-- Chart -->
      <div v-else>
        <VisBulletLegend :items="legendItems" class="mb-3" />
        <VisXYContainer :data="data" :height="220">
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
          <VisAxis type="x" label="Horizon (s)" />
          <VisAxis type="y" label="|Error| (s)" />
        </VisXYContainer>
        <p class="mt-1 text-xs text-gray-400 text-center">
          Lower is better. X = time before arrival when prediction was made (larger = earlier).
        </p>
      </div>

      <template #fallback>
        <div class="h-[220px] bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
      </template>
    </ClientOnly>
  </UCard>
</template>
