<script setup lang="ts">
// RouteBreakdown – MAE per route, horizontal grouped bars so route names stay
// readable. Routes are sorted worst-first by the mean across shown predictors,
// so the problem routes sit at the top.

import type { RouteDatum } from '~/composables/useResultsAnalytics'

const MAX_ROUTES = 12

const props = defineProps<{
  data: RouteDatum[]
  predictors: string[]
  colorMap: Record<string, string>
}>()

const series = computed(() => seriesKeys(props.predictors, props.colorMap))

function meanOf(d: RouteDatum): number {
  const vals = props.predictors.map((p) => d[p]).filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
  return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : -1
}

function complete(d: RouteDatum): boolean {
  return props.predictors.every((p) => typeof d[p] === 'number' && Number.isFinite(d[p] as number))
}

// A bar of 0 s would read as perfect, so routes that some shown predictor
// never scored are left out (and counted in the footer) rather than zero-filled.
const ranked = computed(() => props.data.filter(complete).sort((a, b) => meanOf(b) - meanOf(a)))
const incomplete = computed(() => props.data.length - ranked.value.length)

const rows = computed(() =>
  ranked.value.slice(0, MAX_ROUTES).map((d) => {
    const row: Record<string, string | number> = { route: d.routeLabel }
    for (const p of props.predictors) {
      const v = d[p]
      row[series.value.byId[p] as string] = Math.round((v as number) * 10) / 10
    }
    return row
  })
)

// BarChart types `yAxis` as the numeric keys of the row type, which a dynamic
// record can't express; the keys are the s0…sN series fields built above.
const yKeys = computed(() => series.value.keys as never[])

const hasData = computed(() => rows.value.length > 0 && props.predictors.length > 0)

const height = computed(() => Math.min(520, Math.max(200, rows.value.length * (props.predictors.length * 9 + 14) + 60)))

const worst = computed(() => {
  const d = ranked.value[0]
  return d && props.data.length > 1 ? { route: d.routeLabel, mae: meanOf(d) } : null
})
</script>

<template>
  <AnalyticsChartCard
    title="MAE by route"
    :meta="data.length > MAX_ROUTES ? `worst ${MAX_ROUTES} of ${data.length}` : `${data.length} route${data.length === 1 ? '' : 's'}`"
    :empty="!hasData"
    :height="220"
  >
    <template v-if="worst" #lede>
      Route <b>{{ worst.route }}</b> is the hardest, at <b>{{ fmtSeconds(worst.mae) }}</b> mean error
      across the shown predictors.
    </template>
    <BarChart
      :data="rows"
      :categories="series.categories"
      :y-axis="yKeys"
      x-axis="route"
      :orientation="Orientation.Horizontal"
      :height="height"
      :x-formatter="(v: number) => `${v}s`"
      :x-num-ticks="5"
      y-label="mean absolute error"
      :padding="{ top: 8, right: 24, bottom: 0, left: 0 }"
      :x-grid-line="true"
      :y-grid-line="false"
      :radius="0"
      :bar-gap="2"
      bar-category-gap="22%"
      :legend-position="LegendPosition.TopRight"
      legend-variant="square"
      hover-highlight
      :duration="500"
      aria-label="Mean absolute error by route"
    />
    <template v-if="incomplete > 0" #foot>
      {{ incomplete }} route{{ incomplete === 1 ? '' : 's' }} not scored by every shown predictor {{ incomplete === 1 ? 'is' : 'are' }} left out.
    </template>
  </AnalyticsChartCard>
</template>
