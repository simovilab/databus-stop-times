<script setup lang="ts">
// PredictorComparisonTable – standings for the filtered rows. MAE carries an
// inline bar (shared scale, so lengths compare), and bias a centred bar
// (left = early, right = late). Columns sort on click.

import type { PredictorAgg } from '~/composables/useResultsAnalytics'

const props = defineProps<{
  data: PredictorAgg[]
  colorMap: Record<string, string>
}>()

type SortKey = keyof Omit<PredictorAgg, 'predictor_id'>
type SortDir = 'asc' | 'desc'

const COLUMNS: { key: SortKey; label: string }[] = [
  { key: 'mae', label: 'MAE' },
  { key: 'rmse', label: 'RMSE' },
  { key: 'mape', label: 'MAPE' },
  { key: 'bias', label: 'Bias' },
  { key: 'count', label: 'n' },
]

const sortKey = ref<SortKey>('mae')
const sortDir = ref<SortDir>('asc')

function toggleSort(key: SortKey) {
  if (sortKey.value === key) {
    sortDir.value = sortDir.value === 'asc' ? 'desc' : 'asc'
  } else {
    sortKey.value = key
    sortDir.value = 'asc'
  }
}

const sorted = computed<PredictorAgg[]>(() => {
  const sk = sortKey.value
  const dir = sortDir.value === 'asc' ? 1 : -1
  return [...props.data].sort((a, b) => (a[sk] - b[sk]) * dir)
})

const maxMae = computed(() => Math.max(1, ...props.data.map((d) => (Number.isFinite(d.mae) ? d.mae : 0))))
const maxAbsBias = computed(() => Math.max(1, ...props.data.map((d) => (Number.isFinite(d.bias) ? Math.abs(d.bias) : 0))))
const best = computed(() => [...props.data].filter((d) => Number.isFinite(d.mae)).sort((a, b) => a.mae - b.mae)[0])

function fmtNum(n: number, digits = 1): string {
  return Number.isFinite(n) ? n.toFixed(digits) : '—'
}

function ariaSort(key: SortKey): 'ascending' | 'descending' | 'none' {
  if (sortKey.value !== key) return 'none'
  return sortDir.value === 'asc' ? 'ascending' : 'descending'
}
</script>

<template>
  <AnalyticsChartCard title="Standings" meta="filtered rows" :empty="data.length === 0" :height="200">
    <template v-if="best && data.length > 1" #lede>
      <b>{{ predictorLabel(best.predictor_id) }}</b> has the lowest MAE at <b>{{ fmtSeconds(best.mae) }}</b>
      over {{ best.count.toLocaleString() }} predictions.
    </template>
    <div class="overflow-x-auto -mx-1">
      <table class="w-full text-[11px] text-left">
        <thead>
          <tr class="text-dim uppercase tracking-[0.1em] text-[9.5px]">
            <th class="py-2 px-2 font-normal">Predictor</th>
            <th
              v-for="col in COLUMNS"
              :key="col.key"
              :aria-sort="ariaSort(col.key)"
              class="py-2 px-2 font-normal whitespace-nowrap select-none"
              :class="col.key === 'mae' || col.key === 'bias' ? 'min-w-[120px]' : 'text-right'"
            >
              <button
                type="button"
                class="inline-flex items-center gap-1 hover:text-main"
                :class="sortKey === col.key ? 'text-main' : ''"
                @click="toggleSort(col.key)"
              >
                {{ col.label }}
                <span v-if="sortKey === col.key" aria-hidden="true">{{ sortDir === 'asc' ? '↑' : '↓' }}</span>
              </button>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in sorted" :key="row.predictor_id" class="border-t border-[#141f1a] hover:bg-[#101a16]">
            <td class="py-2 px-2 whitespace-nowrap">
              <span class="inline-flex items-center gap-2">
                <span class="size-2 flex-none" :style="{ backgroundColor: colorMap[row.predictor_id] ?? BASELINE_COLOR }" />
                <span :class="row.predictor_id === best?.predictor_id ? 'text-bright' : 'text-main'" :title="row.predictor_id">
                  {{ predictorLabel(row.predictor_id) }}
                </span>
                <span v-if="isBaseline(row.predictor_id)" class="text-[9px] uppercase tracking-wider text-dim border border-line px-1">base</span>
              </span>
            </td>
            <td class="py-2 px-2">
              <div class="flex items-center gap-2">
                <span class="tabular-nums w-11 text-right" :class="row.predictor_id === best?.predictor_id ? 'text-bright' : 'text-main'">{{ fmtNum(row.mae) }}s</span>
                <div class="mbar flex-1 min-w-[48px]">
                  <span :style="{ width: `${(row.mae / maxMae) * 100}%`, background: colorMap[row.predictor_id] ?? BASELINE_COLOR }" />
                </div>
              </div>
            </td>
            <td class="py-2 px-2 tabular-nums text-right text-main">{{ fmtNum(row.rmse) }}s</td>
            <td class="py-2 px-2 tabular-nums text-right text-main">{{ fmtNum(row.mape) }}%</td>
            <td class="py-2 px-2">
              <div class="flex items-center gap-2">
                <span class="tabular-nums w-12 text-right text-main">{{ row.bias > 0 ? '+' : '' }}{{ fmtNum(row.bias) }}s</span>
                <div class="mbar flex-1 min-w-[48px]" :title="row.bias > 0 ? 'predicts late' : 'predicts early'">
                  <i class="absolute -top-0.5 -bottom-0.5 left-1/2 w-px bg-[#3a4a43]" aria-hidden="true" />
                  <span
                    :style="row.bias >= 0
                      ? { left: '50%', width: `${(row.bias / maxAbsBias) * 50}%`, background: 'var(--color-warn)' }
                      : { left: `${50 - (-row.bias / maxAbsBias) * 50}%`, width: `${(-row.bias / maxAbsBias) * 50}%`, background: '#3987e5' }"
                  />
                </div>
              </div>
            </td>
            <td class="py-2 px-2 tabular-nums text-right text-dim">{{ row.count.toLocaleString() }}</td>
          </tr>
        </tbody>
      </table>
    </div>
    <template #foot>Bias bar: left of centre predicts early, right predicts late.</template>
  </AnalyticsChartCard>
</template>
