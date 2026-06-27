<script setup lang="ts">
// FilterBar – predictor toggles, route selector, horizon range, reset
// All state is lifted to AnalyticsDashboard; this component emits changes.

const props = defineProps<{
  predictors: string[]
  selectedPredictors: string[]
  routes: string[]
  selectedRoute: string | null
  horizonRange: [number, number] | null
  colorMap: Record<string, string>
}>()

const emit = defineEmits<{
  'toggle-predictor': [p: string]
  'select-route': [r: string | null]
  'set-horizon-range': [r: [number, number] | null]
  'reset': []
}>()

// ── Route select ──────────────────────────────────────────────────────────

const routeOptions = computed(() => [
  { label: 'All routes', value: '' },
  ...props.routes.map((r) => ({ label: r, value: r })),
])

const routeModel = computed({
  get: () => props.selectedRoute ?? '',
  set: (v: string) => emit('select-route', v || null),
})

// ── Horizon range (expressed in minutes in the UI, stored in seconds) ─────

const HORIZON_PRESETS = [
  { label: 'All', value: null },
  { label: '< 2 min', value: [0, 120] as [number, number] },
  { label: '2 – 5 min', value: [120, 300] as [number, number] },
  { label: '5 – 10 min', value: [300, 600] as [number, number] },
  { label: '> 10 min', value: [600, 99999] as [number, number] },
]

function setHorizon(val: [number, number] | null) {
  emit('set-horizon-range', val)
}

function isActiveHorizon(preset: [number, number] | null) {
  if (!preset && !props.horizonRange) return true
  if (!preset || !props.horizonRange) return false
  return preset[0] === props.horizonRange[0] && preset[1] === props.horizonRange[1]
}
</script>

<template>
  <UCard>
    <div class="flex flex-wrap items-center gap-4 min-w-0">

      <!-- Predictor toggles -->
      <div class="flex flex-wrap items-center gap-1.5">
        <span class="text-xs font-medium text-gray-500 dark:text-gray-400 mr-1 shrink-0">Predictors:</span>
        <button
          v-for="p in predictors"
          :key="p"
          class="inline-flex items-center gap-1.5 px-2 py-0.5 rounded border text-xs font-medium transition-all duration-150"
          :class="[
            selectedPredictors.includes(p)
              ? 'border-transparent bg-gray-100 dark:bg-gray-800 opacity-100'
              : 'border-transparent bg-gray-100 dark:bg-gray-800 opacity-35 line-through'
          ]"
          :style="{ outlineColor: colorMap[p] ?? '#999' }"
          @click="emit('toggle-predictor', p)"
        >
          <span
            class="w-2.5 h-2.5 rounded-full flex-shrink-0"
            :style="{ backgroundColor: colorMap[p] ?? '#999' }"
          />
          {{ p }}
        </button>
      </div>

      <!-- Route selector -->
      <div class="flex items-center gap-1.5">
        <span class="text-xs font-medium text-gray-500 dark:text-gray-400 shrink-0">Route:</span>
        <select
          v-model="routeModel"
          class="text-xs rounded border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 text-gray-700 dark:text-gray-300 px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
        >
          <option v-for="opt in routeOptions" :key="opt.value" :value="opt.value">
            {{ opt.label }}
          </option>
        </select>
      </div>

      <!-- Horizon presets -->
      <div class="flex items-center gap-1.5 flex-wrap">
        <span class="text-xs font-medium text-gray-500 dark:text-gray-400 shrink-0">Horizon:</span>
        <button
          v-for="preset in HORIZON_PRESETS"
          :key="preset.label"
          class="text-xs px-2 py-0.5 rounded border transition-colors duration-150"
          :class="[
            isActiveHorizon(preset.value)
              ? 'border-blue-400 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 font-medium'
              : 'border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 text-gray-600 dark:text-gray-400 hover:border-blue-300'
          ]"
          @click="setHorizon(preset.value)"
        >
          {{ preset.label }}
        </button>
      </div>

      <!-- Reset -->
      <UButton
        label="Reset"
        variant="ghost"
        size="xs"
        icon="i-heroicons-arrow-path"
        class="ml-auto shrink-0"
        @click="emit('reset')"
      />
    </div>
  </UCard>
</template>
