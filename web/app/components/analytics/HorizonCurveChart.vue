<script setup lang="ts">
// HorizonCurveChart – MAE per horizon bucket, one line per predictor.
// Error grows with horizon; the gap between lines is the story.

import type { BucketDatum } from '~/composables/useResultsAnalytics'

const props = defineProps<{
  data: BucketDatum[]
  predictors: string[]
  colorMap: Record<string, string>
}>()

const series = computed(() => seriesKeys(props.predictors, props.colorMap))
const gapSeries = computed(() =>
  series.value.keys.map((k) => ({ key: k, name: series.value.categories[k]!.name, color: series.value.categories[k]!.color }))
)

const allRows = computed(() =>
  props.data.map((d) => {
    const row: Record<string, string | number | null> = { label: d.bucketLabel }
    props.predictors.forEach((p) => {
      const v = d[p]
      row[series.value.byId[p] as string] = typeof v === 'number' && Number.isFinite(v) ? Math.round(v * 10) / 10 : null
    })
    return row
  })
)

// Drop buckets that no shown predictor reached (a short run never sees >10 min).
const rows = computed(() =>
  allRows.value.filter((r) => series.value.keys.some((k) => typeof r[k] === 'number'))
)

const hasData = computed(() =>
  rows.value.some((r) => series.value.keys.some((k) => typeof r[k] === 'number'))
)

// Takeaway: who leads at the longest horizon that has data, and by how much.
const lede = computed(() => {
  for (let i = rows.value.length - 1; i >= 0; i--) {
    const r = rows.value[i]!
    const ranked = props.predictors
      .map((p) => ({ p, v: r[series.value.byId[p] as string] }))
      .filter((x): x is { p: string; v: number } => typeof x.v === 'number')
      .sort((a, b) => a.v - b.v)
    if (ranked.length >= 2) {
      return { bucket: String(r.label), best: ranked[0]!, next: ranked[1]! }
    }
  }
  return null
})
</script>

<template>
  <AnalyticsChartCard title="MAE by horizon" meta="lower is better" :empty="!hasData" :height="250">
    <template v-if="lede" #lede>
      At <b>{{ lede.bucket }}</b> ahead, <b>{{ predictorLabel(lede.best.p) }}</b> leads at
      <b>{{ fmtSeconds(lede.best.v) }}</b>, {{ fmtSeconds(lede.next.v - lede.best.v) }} better than
      {{ predictorLabel(lede.next.p) }}.
    </template>
    <AnalyticsGapLineChart
      :rows="rows"
      x-key="label"
      :series="gapSeries"
      :height="250"
      :y-formatter="(v: number) => `${Math.round(v)}s`"
      y-label="MAE"
      :tooltip-title="(r) => `${r.label} ahead`"
      aria-label="Mean absolute error by prediction horizon"
    />
  </AnalyticsChartCard>
</template>
