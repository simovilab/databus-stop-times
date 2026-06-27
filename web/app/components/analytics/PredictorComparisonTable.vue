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
  <UCard>
    <template #header>
      <h3 class="text-sm font-semibold text-gray-800 dark:text-gray-200">Predictor Comparison (filtered)</h3>
    </template>

    <div v-if="data.length === 0" class="py-6 text-center text-sm text-gray-400">
      No data for selected filters.
    </div>

    <div v-else class="overflow-x-auto">
      <table class="w-full text-sm text-left">
        <thead>
          <tr class="border-b border-gray-200 dark:border-gray-700">
            <th class="py-2 pr-4 font-semibold text-gray-600 dark:text-gray-400 whitespace-nowrap">Predictor</th>
            <th
              v-for="col in (['mae', 'rmse', 'mape', 'bias', 'count'] as SortKey[])"
              :key="col"
              class="py-2 px-3 font-semibold text-gray-600 dark:text-gray-400 whitespace-nowrap cursor-pointer hover:text-gray-900 dark:hover:text-gray-100 select-none"
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
            class="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800/50"
          >
            <td class="py-2 pr-4 whitespace-nowrap">
              <span class="inline-flex items-center gap-2">
                <span
                  class="w-2.5 h-2.5 rounded-full flex-shrink-0"
                  :style="{ backgroundColor: colorMap[row.predictor_id] ?? '#999' }"
                />
                <span class="font-medium text-gray-800 dark:text-gray-200">{{ row.predictor_id }}</span>
              </span>
            </td>
            <td class="py-2 px-3 tabular-nums text-gray-700 dark:text-gray-300">{{ fmtNum(row.mae) }}</td>
            <td class="py-2 px-3 tabular-nums text-gray-700 dark:text-gray-300">{{ fmtNum(row.rmse) }}</td>
            <td class="py-2 px-3 tabular-nums text-gray-700 dark:text-gray-300">{{ fmtNum(row.mape) }}</td>
            <td
              class="py-2 px-3 tabular-nums"
              :class="row.bias > 0 ? 'text-orange-600 dark:text-orange-400' : row.bias < 0 ? 'text-blue-600 dark:text-blue-400' : 'text-gray-700 dark:text-gray-300'"
            >{{ fmtNum(row.bias) }}</td>
            <td class="py-2 px-3 tabular-nums text-gray-500">{{ row.count.toLocaleString() }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </UCard>
</template>
