<script setup lang="ts">
// AnalyticsCalibrationChart – grouped bar of nominal vs empirical coverage
// per horizon x tolerance calibration bin (stats layer, ?stats=1).
// Degrades to caller's null/empty placeholder when no bins are present.

import { VisXYContainer, VisGroupedBar, VisAxis, VisBulletLegend } from '@unovis/vue'
import type { CalibrationBin } from '~/types/api'

const props = defineProps<{
  data: CalibrationBin[]
}>()

interface CalDatum {
  idx: number
  label: string
  nominal: number
  empirical: number
}

const NOMINAL_COLOR = '#5b6b64'
const EMPIRICAL_COLOR = '#2fd67b'

function fmtHorizon(b: CalibrationBin): string {
  if (b.band_label) return b.band_label
  const lo = b.horizon_lo ?? b.band_lo_s
  const hi = b.horizon_hi ?? b.band_hi_s
  if (lo === undefined || hi === undefined) return '?'
  const fmt = (s: number) => (isFinite(s) ? `${Math.round(s / 60)}m` : '∞')
  return `${fmt(lo)}-${fmt(hi)}`
}

const chartData = computed<CalDatum[]>(() =>
  props.data.map((b, i) => ({
    idx: i,
    label: `${fmtHorizon(b)}@${b.tolerance_s ?? '?'}s`,
    nominal: typeof b.nominal_coverage === 'number' ? b.nominal_coverage : 0,
    empirical: typeof b.empirical_coverage === 'number' ? b.empirical_coverage : 0
  }))
)

const hasData = computed(() => chartData.value.length > 0)

const xAccessor = (d: CalDatum) => d.idx
const yAccessors = [(d: CalDatum) => d.nominal, (d: CalDatum) => d.empirical]
const barColors = [NOMINAL_COLOR, EMPIRICAL_COLOR]
const legendItems = [
  { name: 'Nominal', color: NOMINAL_COLOR },
  { name: 'Empirical', color: EMPIRICAL_COLOR }
]

const xTickFormat = computed(() => {
  const labels = chartData.value.map((d) => d.label)
  return (v: number) => labels[Math.round(v)] ?? ''
})
</script>

<template>
  <div class="panel">
    <div class="panel-header">
      <span class="micro-label">Calibration</span>
    </div>

    <div class="p-3">
      <ClientOnly>
        <div v-if="hasData">
          <VisBulletLegend :items="legendItems" class="mb-3" />
          <VisXYContainer :data="chartData" :height="220" :y-domain="[0, 1]">
            <VisGroupedBar
              :x="xAccessor"
              :y="yAccessors"
              :color="barColors"
              :data-step="1"
              :rounded-corners="0"
            />
            <VisAxis
              type="x"
              label="Horizon @ tolerance"
              :tick-format="xTickFormat"
              :num-ticks="Math.min(chartData.length, 10)"
            />
            <VisAxis type="y" label="Coverage" />
          </VisXYContainer>
        </div>
        <div v-else class="px-3 py-8 text-center text-xs text-dim">
          No calibration data for this run.
        </div>

        <template #fallback>
          <div class="h-[220px] bg-panel animate-pulse" />
        </template>
      </ClientOnly>
    </div>
  </div>
</template>
