<script setup lang="ts">
// GapLineChart – a multi-series line chart that tolerates missing values.
//
// nuxt-charts' LineChart drops every row in which any series is missing, so a
// predictor without data in one bucket would erase that bucket for everyone
// (the GTFS-RT feed, for one, only predicts the next few stops). This draws the
// same chart from the vccs primitives nuxt-charts is built on, where a missing
// value is a gap (or, with `connectNulls`, a straight join across it). It sits
// inside `.vue-chrts`, so it shares the global chart theme.

import { CartesianGrid, LineChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'vccs'

export interface GapSeries {
  key: string
  name: string
  color: string
}

type Row = Record<string, string | number | null>

const props = withDefaults(defineProps<{
  rows: Row[]
  xKey: string
  series: GapSeries[]
  height?: number
  yFormatter?: (v: number) => string
  yLabel?: string
  xLabel?: string
  yDomain?: [number | string, number | string]
  connectNulls?: boolean
  zeroLine?: boolean
  tooltipTitle?: (row: Row) => string
  ariaLabel?: string
}>(), {
  height: 250,
  yFormatter: (v: number) => String(v),
  yLabel: undefined,
  xLabel: undefined,
  yDomain: () => [0, 'auto'],
  connectNulls: false,
  zeroLine: false,
  tooltipTitle: undefined,
  ariaLabel: 'Line chart',
})

const hidden = ref<Set<string>>(new Set())
function toggle(key: string) {
  const next = new Set(hidden.value)
  if (next.has(key)) next.delete(key)
  else next.add(key)
  hidden.value = next
}

const tickStyle = { fill: 'var(--vc-tick-color)', fontSize: 10 }
const axisLine = { stroke: 'var(--vc-axis-line-color)' }

interface TipEntry { dataKey?: string | number; value?: unknown; payload?: Row }
function tipRows(payload: TipEntry[] | undefined) {
  const row = payload?.[0]?.payload
  if (!row) return []
  return props.series
    .filter((s) => !hidden.value.has(s.key))
    .map((s) => ({ ...s, v: row[s.key] }))
    .filter((s): s is GapSeries & { v: number } => typeof s.v === 'number')
    .sort((a, b) => a.v - b.v)
}
</script>

<template>
  <div class="vue-chrts" role="group" :aria-label="ariaLabel">
    <ul class="flex flex-wrap justify-end gap-x-4 gap-y-1 mb-2 text-[11px]" aria-label="Series (click to hide)">
      <li v-for="s in series" :key="s.key">
        <button
          type="button"
          class="inline-flex items-center gap-1.5 transition-opacity"
          :class="hidden.has(s.key) ? 'opacity-35' : 'opacity-100'"
          :aria-pressed="!hidden.has(s.key)"
          @click="toggle(s.key)"
        >
          <span class="inline-block w-3 h-[3px]" :style="{ background: s.color }" />
          <span class="text-main">{{ s.name }}</span>
        </button>
      </li>
    </ul>
    <ResponsiveContainer width="100%" :height="height">
      <LineChart :data="rows" :margin="{ top: 8, right: 24, bottom: 4, left: 0 }">
        <CartesianGrid
          :vertical="false"
          stroke="var(--vc-grid-color)"
          stroke-dasharray="var(--vc-grid-dash)"
        />
        <XAxis
          :data-key="xKey"
          type="category"
          :tick="tickStyle"
          :tick-line="false"
          :axis-line="axisLine"
          :tick-margin="8"
          :height="xLabel ? 44 : 28"
          :label="xLabel ? { value: xLabel, position: 'insideBottom', offset: 0, style: { fill: 'var(--vc-axis-label-color)', fontSize: 10 } } : undefined"
        />
        <YAxis
          type="number"
          :domain="yDomain"
          :tick="tickStyle"
          :tick-line="false"
          :axis-line="false"
          :tick-count="5"
          :tick-formatter="yFormatter"
          :width="yLabel ? 56 : 44"
          :label="yLabel ? { value: yLabel, angle: -90, position: 'insideLeft', style: { fill: 'var(--vc-axis-label-color)', fontSize: 10, textAnchor: 'middle' } } : undefined"
        />
        <ReferenceLine v-if="zeroLine" :y="0" stroke="var(--vc-reference-line-color)" stroke-dasharray="3 3" />
        <Tooltip :cursor="{ stroke: 'var(--vc-axis-line-color)', strokeWidth: 1 }" :is-animation-active="false">
          <template #content="{ payload, active }">
            <div v-if="active && tipRows(payload as TipEntry[]).length" class="ctip">
              <div class="ctip__title">
                {{ tooltipTitle && (payload as TipEntry[])[0]?.payload ? tooltipTitle((payload as TipEntry[])[0]!.payload!) : (payload as TipEntry[])[0]?.payload?.[xKey] }}
              </div>
              <div v-for="(s, i) in tipRows(payload as TipEntry[])" :key="s.key" class="ctip__row">
                <span class="sw" :style="{ background: s.color }" />
                <span class="n">{{ i === 0 ? '★ ' : '' }}{{ s.name }}</span>
                <span class="v">{{ yFormatter(s.v) }}</span>
              </div>
            </div>
          </template>
        </Tooltip>
        <Line
          v-for="s in series"
          :key="s.key"
          :data-key="s.key"
          :name="s.name"
          type="monotoneX"
          :stroke="s.color"
          :stroke-width="2.25"
          :hide="hidden.has(s.key)"
          :connect-nulls="connectNulls"
          :dot="{ r: 3.5, fill: 'var(--color-panel)', stroke: s.color, strokeWidth: 2 }"
          :active-dot="{ r: 5, fill: s.color, stroke: 'var(--color-panel)', strokeWidth: 2 }"
          :is-animation-active="true"
          :transition="{ duration: 0.5, ease: 'easeOut' }"
        />
      </LineChart>
    </ResponsiveContainer>
  </div>
</template>
