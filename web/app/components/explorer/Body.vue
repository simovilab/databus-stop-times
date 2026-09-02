<script setup lang="ts">
// ExplorerBody – left slicer spine + 3-column bento of analytics charts for one run.
// Mirrors AnalyticsDashboard's data flow (useResultsAnalytics) but with a
// restructured ops-console layout: sticky left filter spine, bento chart grid.

const props = defineProps<{
  runId: string
}>()

const analytics = useResultsAnalytics(props.runId)

const {
  loading,
  fetchError,
  filtered,
  rows,
  predictors,
  activePredictors,
  routes,
  colorMap,
  selectedPredictors,
  selectedRoute,
  horizonRange,
  selectedTripStop,
  predictorAggs,
  bucketData,
  histData,
  routeData,
  headToHeadData,
  convergenceData,
  load,
  togglePredictor,
  selectRoute,
  setHorizonRange,
  selectTripStop,
  clearTripStop,
  reset
} = analytics

onMounted(() => {
  load()
})

const hasRows = computed(() => rows.value.length > 0)
const hasFiltered = computed(() => filtered.value.length > 0)

// ── Route select ──────────────────────────────────────────────────────────
const routeItems = computed(() => [
  { label: 'All routes', value: '' },
  ...routes.value.map((r) => ({ label: r, value: r }))
])
const routeModel = computed({
  get: () => selectedRoute.value ?? '',
  set: (v: string) => selectRoute(v || null)
})

// ── Horizon presets ───────────────────────────────────────────────────────
const HORIZON_PRESETS: { label: string; value: [number, number] | null }[] = [
  { label: 'All', value: null },
  { label: '<2m', value: [0, 120] },
  { label: '2-5m', value: [120, 300] },
  { label: '5-10m', value: [300, 600] },
  { label: '>10m', value: [600, 99999] }
]

function isActiveHorizon(preset: [number, number] | null): boolean {
  if (!preset && !horizonRange.value) return true
  if (!preset || !horizonRange.value) return false
  return preset[0] === horizonRange.value[0] && preset[1] === horizonRange.value[1]
}
</script>

<template>
  <div>
    <!-- Loading -->
    <div v-if="loading" class="flex items-center gap-2 text-xs text-dim py-4">
      <UIcon name="i-heroicons-arrow-path" class="animate-spin w-4 h-4" />
      Loading analytics data&hellip;
    </div>

    <UAlert v-else-if="fetchError" color="warning" variant="subtle" :title="`Could not load results: ${fetchError}`" />

    <UAlert
      v-else-if="!hasRows"
      color="neutral"
      variant="subtle"
      title="No scored results yet."
      description="Analytics will appear once the run has completed evaluations."
    />

    <div v-else class="flex gap-3 items-start">
      <!-- LEFT: slicer spine -->
      <div class="panel w-56 shrink-0 sticky top-14">
        <div class="panel-header"><span class="micro-label">Slice controls</span></div>

        <div class="px-3 py-2 border-b border-line">
          <div class="tile-l mb-2">Predictors</div>
          <div class="flex flex-wrap gap-1.5">
            <button
              v-for="p in predictors"
              :key="p"
              class="chip"
              :class="{ on: selectedPredictors.includes(p) }"
              @click="togglePredictor(p)"
            >
              <span class="w-2 h-2 shrink-0" :style="{ backgroundColor: colorMap[p] ?? '#999' }" />
              {{ p }}
            </button>
          </div>
        </div>

        <div class="px-3 py-2 border-b border-line">
          <div class="tile-l mb-2">Route</div>
          <USelect v-model="routeModel" :items="routeItems" class="w-full" size="xs" />
        </div>

        <div class="px-3 py-2 border-b border-line">
          <div class="tile-l mb-2">Horizon</div>
          <div class="flex flex-wrap gap-1.5">
            <button
              v-for="preset in HORIZON_PRESETS"
              :key="preset.label"
              class="chip"
              :class="{ on: isActiveHorizon(preset.value) }"
              @click="setHorizonRange(preset.value)"
            >
              {{ preset.label }}
            </button>
          </div>
        </div>

        <div class="px-3 py-2">
          <UButton label="Reset" variant="ghost" size="xs" icon="i-heroicons-arrow-path" block @click="reset" />
        </div>
      </div>

      <!-- RIGHT: bento -->
      <div class="flex-1 min-w-0">
        <UAlert
          v-if="!hasFiltered"
          color="warning"
          variant="subtle"
          title="No rows match the current filters."
          description="Try broadening the predictor selection, route, or horizon range."
        />

        <div v-else class="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div class="md:col-span-2 md:row-span-2">
            <AnalyticsErrorScatter
              :data="filtered"
              :predictors="activePredictors"
              :color-map="colorMap"
              @select-trip-stop="selectTripStop"
            />
          </div>
          <AnalyticsPredictorComparisonTable :data="predictorAggs" :color-map="colorMap" />
          <AnalyticsHorizonCurveChart :data="bucketData" :predictors="activePredictors" :color-map="colorMap" />

          <AnalyticsConvergenceChart
            :data="convergenceData"
            :predictors="activePredictors"
            :color-map="colorMap"
            :selected-trip-stop="selectedTripStop"
            @clear="clearTripStop"
          />
          <AnalyticsErrorDistribution :data="histData" :predictors="activePredictors" :color-map="colorMap" />
          <AnalyticsRouteBreakdown :data="routeData" :predictors="activePredictors" :color-map="colorMap" />

          <div class="md:col-span-3">
            <AnalyticsHeadToHead :data="headToHeadData" :color-map="colorMap" />
          </div>

          <!-- Calibration placeholder – real chart lands in a later wave -->
          <div class="panel border-dashed">
            <div class="panel-header"><span class="micro-label">Calibration</span></div>
            <div class="px-3 py-8 text-center text-xs text-dim">
              CALIBRATION — pending stats layer
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
