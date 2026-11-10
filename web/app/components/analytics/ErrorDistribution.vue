<script setup lang="ts">
// ErrorDistribution – overlaid error densities, one area per predictor.
// Each series is normalised to its own total, so predictors with different
// result counts compare fairly. A tall, narrow peak centred on zero wins.

import type { HistBin } from '~/composables/useResultsAnalytics'

const props = defineProps<{
  data: HistBin[]
  predictors: string[]
  colorMap: Record<string, string>
}>()

const series = computed(() => seriesKeys(props.predictors, props.colorMap))

const totals = computed(() => {
  const t: Record<string, number> = {}
  for (const p of props.predictors) {
    t[p] = props.data.reduce((acc, d) => acc + (typeof d[p] === 'number' ? (d[p] as number) : 0), 0)
  }
  return t
})

const EDGE = 300
function binLabel(c: number): string {
  if (c === 0) return '0'
  const a = Math.abs(c)
  const body = a % 60 === 0 ? `${a / 60}m` : `${a}s`
  if (c <= -EDGE) return `≤−${body}`
  if (c >= EDGE) return `≥+${body}`
  return `${c > 0 ? '+' : '−'}${body}`
}

const rows = computed(() =>
  props.data.map((d) => {
    const row: Record<string, string | number> = { label: binLabel(d.binCenter), center: d.binCenter }
    for (const p of props.predictors) {
      const n = typeof d[p] === 'number' ? (d[p] as number) : 0
      const tot = totals.value[p] ?? 0
      row[series.value.byId[p] as string] = tot > 0 ? Math.round((n / tot) * 1000) / 10 : 0
    }
    return row
  })
)

const hasData = computed(() => props.predictors.some((p) => (totals.value[p] ?? 0) > 0))

// Takeaway: who has the most mass in the three centre bins (−45 … +45 s).
const lede = computed(() => {
  const centre = props.data.filter((d) => Math.abs(d.binCenter) <= 30)
  const ranked = props.predictors
    .map((p) => {
      const tot = totals.value[p] ?? 0
      const inC = centre.reduce((a, d) => a + (typeof d[p] === 'number' ? (d[p] as number) : 0), 0)
      return { p, share: tot > 0 ? inC / tot : 0 }
    })
    .filter((x) => x.share > 0)
    .sort((a, b) => b.share - a.share)
  return ranked.length ? ranked[0] : null
})

const zeroLabel = computed(() => rows.value.find((r) => r.center === 0)?.label as string | undefined)
</script>

<template>
  <AnalyticsChartCard title="Error distribution" meta="− early · + late" :empty="!hasData" :height="250">
    <template v-if="lede" #lede>
      <b>{{ predictorLabel(lede.p) }}</b> lands <b>{{ Math.round(lede.share * 100) }}%</b> of its
      predictions within 45 s of the actual arrival.
    </template>
    <AreaChart
      :data="rows"
      :categories="series.categories"
      x-axis="label"
      :height="250"
      :y-num-ticks="4"
      :y-formatter="(v: number) => `${v}%`"
      y-label="share of predictions"
      :y-grid-line="true"
      :padding="{ top: 8, right: 24, bottom: 0, left: 0 }"
      :x-grid-line="false"
      :curve-type="CurveType.MonotoneX"
      :line-width="1.75"
      variant="gradient"
      :gradient-stops="[{ offset: '0%', stopOpacity: 0.32 }, { offset: '100%', stopOpacity: 0 }]"
      :reference-lines="zeroLabel ? [{ x: zeroLabel, strokeDasharray: '3 3' }] : []"
      :legend-position="LegendPosition.TopRight"
      legend-variant="horizontal-bar"
      :x-num-ticks="7"
      :duration="500"
      aria-label="Distribution of prediction error"
    >
      <template #tooltip="{ values }">
        <div v-if="values" class="ctip">
          <div class="ctip__title">error ≈ {{ values.label }}</div>
          <div v-for="p in predictors" :key="p" class="ctip__row">
            <span class="sw" :style="{ background: colorMap[p] }" />
            <span class="n">{{ predictorLabel(p) }}</span>
            <span class="v">{{ values[series.byId[p]!] }}%</span>
          </div>
        </div>
      </template>
    </AreaChart>
  </AnalyticsChartCard>
</template>
