<script setup lang="ts">
// HeadToHead – win count bar chart + win-share donut per predictor.
// A "win" is awarded when a predictor has the lowest MAE on a (trip_key, stop_id) contest.

import { VisXYContainer, VisGroupedBar, VisAxis, VisSingleContainer, VisDonut, VisBulletLegend } from '@unovis/vue'
import type { HeadToHeadResult } from '~/composables/useResultsAnalytics'

const props = defineProps<{
  data: HeadToHeadResult[]
  colorMap: Record<string, string>
}>()

// ── Bar chart – win counts ────────────────────────────────────────────────

const barData = computed(() => props.data.map((d, i) => ({ ...d, predIdx: i })))

const xAccessor = (d: { predIdx: number }) => d.predIdx
const yWins = (d: { wins: number }) => d.wins

const barColors = computed(() => props.data.map((d) => props.colorMap[d.predictor_id] ?? '#999'))

const xTickFormat = computed(() => {
  const labels = props.data.map((d) => d.predictor_id)
  return (v: number) => labels[Math.round(v)] ?? ''
})

// ── Donut – win share ─────────────────────────────────────────────────────

const donutValue = (d: HeadToHeadResult) => d.winShare
const donutColor = (d: HeadToHeadResult) => props.colorMap[d.predictor_id] ?? '#999'

const totalContests = computed(() => props.data[0]?.contests ?? 0)

// ── Legend ────────────────────────────────────────────────────────────────

const legendItems = computed(() =>
  props.data.map((d) => ({ name: d.predictor_id, color: props.colorMap[d.predictor_id] ?? '#999' }))
)

const hasContests = computed(() => totalContests.value > 0)
</script>

<template>
  <UCard>
    <template #header>
      <div>
        <h3 class="text-sm font-semibold text-gray-800 dark:text-gray-200">Head-to-Head Win Rate</h3>
        <p v-if="hasContests" class="text-xs text-gray-400 mt-0.5">
          {{ totalContests.toLocaleString() }} contested (trip, stop) pairs
        </p>
      </div>
    </template>

    <ClientOnly>
      <div v-if="hasContests">
        <VisBulletLegend :items="legendItems" class="mb-3" />
        <div class="grid grid-cols-2 gap-4">
          <!-- Win count bars -->
          <div>
            <p class="text-xs text-gray-500 mb-2 text-center">Win count</p>
            <VisXYContainer :data="barData" :height="180">
              <VisGroupedBar
                :x="xAccessor"
                :y="yWins"
                :color="barColors"
                :data-step="1"
                :rounded-corners="2"
              />
              <VisAxis
                type="x"
                :tick-format="xTickFormat"
                :num-ticks="props.data.length"
              />
              <VisAxis type="y" label="Wins" />
            </VisXYContainer>
          </div>

          <!-- Win share donut -->
          <div>
            <p class="text-xs text-gray-500 mb-2 text-center">Win share</p>
            <VisSingleContainer :data="data" :height="180">
              <VisDonut
                :value="donutValue"
                :color="donutColor"
                :arc-width="28"
                :pad-angle="0.02"
              />
            </VisSingleContainer>
          </div>
        </div>
      </div>

      <!-- Not enough predictors for head-to-head -->
      <div v-else-if="props.data.length < 2" class="py-8 text-center text-sm text-gray-400 dark:text-gray-600">
        Head-to-head requires at least 2 active predictors.
      </div>

      <!-- No contested pairs -->
      <div v-else class="py-8 text-center text-sm text-gray-400 dark:text-gray-600">
        No shared (trip, stop) pairs between predictors in the current filter.
      </div>

      <template #fallback>
        <div class="h-[200px] bg-gray-100 dark:bg-gray-800 rounded animate-pulse" />
      </template>
    </ClientOnly>
  </UCard>
</template>
