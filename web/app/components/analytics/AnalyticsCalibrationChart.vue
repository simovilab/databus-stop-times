<script setup lang="ts">
// AnalyticsCalibrationChart – for each horizon band and tolerance, the share
// of predictions that landed within tolerance (empirical) next to the share the
// predictor promised (nominal). Empirical below nominal means overconfident.

import type { CalibrationBin } from '~/types/api'

const props = defineProps<{
  data: CalibrationBin[]
}>()

const NOMINAL_COLOR = '#56645d'
const EMPIRICAL_COLOR = '#199e70'

function fmtHorizon(b: CalibrationBin): string {
  if (b.band_label) return b.band_label
  const lo = b.horizon_lo ?? b.band_lo_s
  const hi = b.horizon_hi ?? b.band_hi_s
  if (lo === undefined || hi === undefined) return '?'
  const fmt = (s: number) => (Number.isFinite(s) ? `${Math.round(s / 60)}m` : '∞')
  return `${fmt(lo)}–${fmt(hi)}`
}

const rows = computed(() =>
  props.data.map((b) => ({
    label: `${fmtHorizon(b)} ±${b.tolerance_s ?? '?'}s`,
    nominal: typeof b.nominal_coverage === 'number' ? b.nominal_coverage : 0,
    empirical: typeof b.empirical_coverage === 'number' ? b.empirical_coverage : 0,
  }))
)

const categories = {
  nominal: { name: 'promised', color: NOMINAL_COLOR },
  empirical: { name: 'achieved', color: EMPIRICAL_COLOR },
}

const worstGap = computed(() => {
  const gaps = rows.value.map((r) => ({ label: r.label, gap: r.empirical - r.nominal })).sort((a, b) => a.gap - b.gap)
  return gaps[0] && gaps[0].gap < -0.02 ? gaps[0] : null
})
</script>

<template>
  <AnalyticsChartCard title="Calibration" meta="coverage" :empty="rows.length === 0" :height="240">
    <template #lede>
      <template v-if="worstGap">
        Biggest shortfall at <b>{{ worstGap.label }}</b>: achieved coverage is
        <b>{{ Math.round(-worstGap.gap * 100) }} pts</b> below what was promised.
      </template>
      <template v-else>Achieved coverage meets the promised coverage in every band.</template>
    </template>
    <BarChart
      :data="rows"
      :categories="categories"
      :y-axis="['nominal', 'empirical']"
      x-axis="label"
      :height="240"
      :y-domain="[0, 1]"
      :y-num-ticks="5"
      :y-formatter="(v: number) => `${Math.round(v * 100)}%`"
      :y-grid-line="true"
      :x-grid-line="false"
      :radius="0"
      :bar-gap="2"
      :legend-position="LegendPosition.TopRight"
      legend-variant="square"
      :duration="500"
      aria-label="Promised against achieved coverage by horizon band"
    />
  </AnalyticsChartCard>
</template>
