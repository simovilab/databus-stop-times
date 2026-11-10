<script setup lang="ts">
// ErrorScatter – every prediction as a dot: horizon (x) vs signed error (y).
// The cloud should hug zero and fan out with horizon. Clicking a dot selects
// its trip/stop and drives the convergence chart.
//
// The chart library has no click event on points, so the tooltip slot records
// the row under the pointer and a click on the plot selects that row.

import type { ResultRow } from '~/types/api'

const MAX_POINTS = 3000

const props = defineProps<{
  data: ResultRow[]
  predictors: string[]
  colorMap: Record<string, string>
}>()

const emit = defineEmits<{
  'select-trip-stop': [trip_key: string, stop_id: string]
}>()

const series = computed(() => seriesKeys(props.predictors, props.colorMap))

// BubbleChart's legend prints the category value itself, so key the series by
// display label ('.' is swapped out because the chart may read it as a path).
function seriesName(id: string): string {
  return predictorLabel(id).replaceAll('.', '·')
}
const categories = computed(() =>
  Object.fromEntries(props.predictors.map((p) => [seriesName(p), { name: seriesName(p), color: props.colorMap[p] ?? BASELINE_COLOR }]))
)

interface Point {
  x: number
  y: number
  z: number
  series: string
  row: ResultRow
}

// Even sampling across the whole run, not the first N rows.
const points = computed<Point[]>(() => {
  const src = props.data.filter((r) => series.value.byId[r.predictor_id] !== undefined)
  const step = Math.max(1, Math.ceil(src.length / MAX_POINTS))
  const out: Point[] = []
  for (let i = 0; i < src.length; i += step) {
    const r = src[i]!
    out.push({ x: r.horizon_s, y: r.error_s, z: 1, series: seriesName(r.predictor_id), row: r })
  }
  return out
})

const isSampled = computed(() => points.value.length < props.data.length)

// Symmetric y around zero on a "nice" minute scale, clipped at the 95th
// percentile of |error| so a few wild misses don't flatten the cloud.
const Y_LIMITS = [60, 120, 240, 480, 960, 1920]
const yDomain = computed<[number, number]>(() => {
  const abs = points.value.map((p) => Math.abs(p.y)).sort((a, b) => a - b)
  const q = abs[Math.floor(abs.length * 0.95)] ?? 60
  const lim = Y_LIMITS.find((l) => l >= q * 1.15) ?? Y_LIMITS[Y_LIMITS.length - 1]!
  return [-lim, lim]
})
// Explicit x ticks on whole-minute steps (the chart accepts x ticks, not y).
const X_STEPS = [30, 60, 120, 300, 600, 900, 1800, 3600]
const xTicks = computed<number[]>(() => {
  const max = Math.max(60, ...points.value.map((p) => p.x))
  const step = X_STEPS.find((st) => max / st <= 6) ?? X_STEPS[X_STEPS.length - 1]!
  return Array.from({ length: Math.floor(max / step) + 1 }, (_, i) => i * step)
})

const clipped = computed(() => points.value.filter((p) => Math.abs(p.y) > yDomain.value[1]).length)

// Misses beyond the scale are pinned to its edge (the tooltip keeps the true value).
const plotted = computed<Point[]>(() => {
  const lim = yDomain.value[1]
  return points.value.map((p) => (Math.abs(p.y) > lim ? { ...p, y: Math.sign(p.y) * lim } : p))
})

// The chart is typed for plain records; these keep the Point shape at the edges.
type Datum = Record<string, unknown>
const chartData = computed<Datum[]>(() => plotted.value as unknown as Datum[])
const xOf = (d: Datum) => (d as unknown as Point).x
const yOf = (d: Datum) => (d as unknown as Point).y
const zOf = (d: Datum) => (d as unknown as Point).z
function asPoint(v: unknown): Point | null {
  return v && typeof v === 'object' && 'row' in v ? (v as Point) : null
}

const hovered = shallowRef<Point | null>(null)
function track(v: unknown): Point | null {
  hovered.value = asPoint(v)
  return hovered.value
}
function onClick() {
  const r = hovered.value?.row
  if (r?.trip_key && r.stop_id) emit('select-trip-stop', r.trip_key, r.stop_id)
}
</script>

<template>
  <AnalyticsChartCard
    title="Error vs horizon"
    :meta="isSampled ? `${points.length.toLocaleString()} of ${data.length.toLocaleString()} shown` : `${points.length.toLocaleString()} predictions`"
    :empty="points.length === 0"
    :height="280"
  >
    <template #lede>
      Each dot is one prediction. Above the line is <b>late</b>, below is <b>early</b>.
      Click a dot to trace that trip and stop.
    </template>
    <!-- eslint-disable-next-line vuejs-accessibility/click-events-have-key-events -->
    <div class="cursor-crosshair" @click="onClick" @mouseleave="hovered = null">
      <BubbleChart
        :data="chartData"
        :categories="categories"
        :x-accessor="xOf"
        :y-accessor="yOf"
        :size-accessor="zOf"
        category-key="series"
        :size-range="[18, 18]"
        :opacity="0.55"
        :height="280"
        :y-num-ticks="5"
        :x-explicit-ticks="xTicks"
        :x-formatter="(v: number) => fmtSeconds(v)"
        :y-formatter="(v: number) => `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v)}s`"
        x-label="horizon"
        y-label="error"
        :x-grid-line="false"
        :y-grid-line="true"
        :max-data-points="MAX_POINTS"
        :accessible-data-table="false"
        :legend-position="LegendPosition.TopRight"
        legend-variant="circle"
        aria-label="Prediction error against horizon"
      >
        <template #tooltip="{ values }">
          <template v-for="pt in [track(values)]" :key="0">
            <div v-if="pt" class="ctip">
              <div class="ctip__row">
                <span class="sw" :style="{ background: colorMap[pt.row.predictor_id] }" />
                <span class="n">{{ predictorLabel(pt.row.predictor_id) }}</span>
              </div>
              <div class="ctip__row"><span class="n">error</span><span class="v">{{ fmtSignedSeconds(pt.row.error_s) }}</span></div>
              <div class="ctip__row"><span class="n">horizon</span><span class="v">{{ fmtSeconds(pt.x) }}</span></div>
              <div class="ctip__row"><span class="n">route · stop</span><span class="v">{{ pt.row.route_id }} · {{ pt.row.stop_id }}</span></div>
              <div class="ctip__hint">click to trace this trip/stop</div>
            </div>
          </template>
        </template>
      </BubbleChart>
    </div>
    <template v-if="clipped > 0" #foot>
      {{ clipped }} predictions beyond ±{{ fmtSeconds(yDomain[1]) }} are pinned to the edge.
    </template>
  </AnalyticsChartCard>
</template>
