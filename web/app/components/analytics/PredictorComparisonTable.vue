<script setup lang="ts">
// PredictorComparisonTable – sortable UTable of MAE/RMSE/MAPE/Bias/Count per predictor.
// Computed from filtered ResultRow[], so it reacts to all active filters.

import type { PredictorAgg } from '~/composables/useResultsAnalytics'

const props = defineProps<{
  data: PredictorAgg[]
  colorMap: Record<string, string>
}>()

type SortKey = keyof Omit<PredictorAgg, 'predictor_id'>
type SortDir = 'asc' | 'desc'

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

function fmtNum(n: number): string {
  return isFinite(n) ? n.toFixed(1) : '—'
}

function sortIcon(key: SortKey): string {
  if (sortKey.value !== key) return 'i-heroicons-chevron-up-down'
  return sortDir.value === 'asc' ? 'i-heroicons-chevron-up' : 'i-heroicons-chevron-down'
}
</script>

<template>
  <div class="panel">
    <div class="panel-header">
      <span class="micro-label">Predictor comparison (filtered)</span>
    </div>

    <div v-if="data.length === 0" class="py-6 text-center text-xs text-dim">
      No data for selected filters.
    </div>

    <div v-else class="overflow-x-auto">
      <table class="w-full text-[11px] text-left">
        <thead>
          <tr class="border-b border-line text-dim uppercase tracking-wide text-[10px]">
            <th class="py-1.5 px-3 font-normal whitespace-nowrap">Predictor</th>
            <th
              v-for="col in (['mae', 'rmse', 'mape', 'bias', 'count'] as SortKey[])"
              :key="col"
              class="py-1.5 px-3 font-normal whitespace-nowrap cursor-pointer hover:text-main select-none"
              @click="toggleSort(col)"
            >
              <span class="inline-flex items-center gap-1">
                {{ col === 'mape' ? 'MAPE (%)' : col === 'mae' ? 'MAE (s)' : col === 'rmse' ? 'RMSE (s)' : col === 'bias' ? 'Bias (s)' : 'Count' }}
                <UIcon :name="sortIcon(col)" class="w-3 h-3" />
              </span>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="row in sorted"
            :key="row.predictor_id"
            class="border-b border-[#141f1a]"
          >
            <td class="py-1 px-3 whitespace-nowrap">
              <span class="inline-flex items-center gap-2">
                <span
                  class="w-2 h-2 flex-shrink-0"
                  :style="{ backgroundColor: colorMap[row.predictor_id] ?? '#999' }"
                />
                <span class="text-bright">{{ row.predictor_id }}</span>
              </span>
            </td>
            <td class="py-1 px-3 tabular-nums text-main">{{ fmtNum(row.mae) }}</td>
            <td class="py-1 px-3 tabular-nums text-main">{{ fmtNum(row.rmse) }}</td>
            <td class="py-1 px-3 tabular-nums text-main">{{ fmtNum(row.mape) }}</td>
            <td
              class="py-1 px-3 tabular-nums"
              :class="row.bias > 0 ? 'text-warn' : row.bias < 0 ? 'text-accent' : 'text-main'"
            >{{ fmtNum(row.bias) }}</td>
            <td class="py-1 px-3 tabular-nums text-dim">{{ row.count.toLocaleString() }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>
