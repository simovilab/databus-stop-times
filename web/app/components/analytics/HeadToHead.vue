<script setup lang="ts">
// HeadToHead – for every trip/stop that two or more predictors scored, who had
// the lowest MAE. A donut shows the split; the ranked list beside it gives the
// exact numbers.

import type { HeadToHeadResult } from '~/composables/useResultsAnalytics'

const props = defineProps<{
  data: HeadToHeadResult[]
  colorMap: Record<string, string>
}>()

const ranked = computed(() => [...props.data].sort((a, b) => b.wins - a.wins))
const totalContests = computed(() => props.data[0]?.contests ?? 0)
const hasContests = computed(() => totalContests.value > 0)
const leader = computed(() => ranked.value[0])

const donutData = computed(() => ranked.value.map((d) => d.wins))
const donutCategories = computed(() =>
  Object.fromEntries(
    ranked.value.map((d) => [predictorLabel(d.predictor_id), { name: predictorLabel(d.predictor_id), color: props.colorMap[d.predictor_id] ?? BASELINE_COLOR }])
  )
)
</script>

<template>
  <AnalyticsChartCard
    title="Head-to-head"
    :meta="hasContests ? `${totalContests.toLocaleString()} contested trip/stops` : undefined"
    :empty="!hasContests"
    empty-text="Needs at least two predictors scoring the same trip/stop."
    :height="200"
  >
    <template v-if="hasContests && leader" #lede>
      <b>{{ predictorLabel(leader.predictor_id) }}</b> had the lowest error on
      <b>{{ Math.round(leader.winShare * 100) }}%</b> of trip/stops where predictors met.
    </template>
    <div class="grid grid-cols-1 sm:grid-cols-[220px_1fr] gap-6 items-center px-2 py-2">
      <DonutChart
        :data="donutData"
        :categories="donutCategories"
        :height="200"
        :radius="0"
        :arc-width="18"
        :pad-angle="0.02"
        hide-legend
        aria-label="Share of head-to-head wins"
      >
        <div class="text-center leading-tight">
          <div class="text-[22px] text-bright tabular-nums">{{ Math.round((leader?.winShare ?? 0) * 100) }}%</div>
          <div class="micro-label mt-1">top win share</div>
        </div>
      </DonutChart>

      <ol class="space-y-3 min-w-0">
        <li v-for="(d, i) in ranked" :key="d.predictor_id" class="min-w-0">
          <div class="flex items-baseline justify-between gap-3 text-[11px] mb-1.5">
            <span class="flex items-center gap-2 min-w-0">
              <span class="text-dim tabular-nums w-4">{{ i + 1 }}</span>
              <span class="inline-block size-2 flex-none" :style="{ background: colorMap[d.predictor_id] }" />
              <span class="truncate" :class="i === 0 ? 'text-bright' : 'text-main'" :title="d.predictor_id">
                {{ predictorLabel(d.predictor_id) }}
              </span>
            </span>
            <span class="tabular-nums whitespace-nowrap">
              <span :class="i === 0 ? 'text-bright' : 'text-main'">{{ Math.round(d.winShare * 100) }}%</span>
              <span class="text-dim"> · {{ d.wins.toLocaleString() }} wins</span>
            </span>
          </div>
          <div class="mbar"><span :style="{ width: `${d.winShare * 100}%`, background: colorMap[d.predictor_id] }" /></div>
        </li>
      </ol>
    </div>
  </AnalyticsChartCard>
</template>
