<script setup lang="ts">
/**
 * AnalyticsDashboard (C2)
 *
 * Orchestrates all analytics chart components and the shared filter state.
 * Mounts when a run is "done" and has scored results.
 *
 * Layout (responsive Nuxt UI grid):
 *  - FilterBar (full width)
 *  - PredictorComparisonTable + HorizonCurveChart  (2-col on md+)
 *  - ErrorScatter + ConvergenceChart               (2-col on md+)
 *  - ErrorDistribution + RouteBreakdown            (2-col on md+)
 *  - HeadToHead                                    (full width)
 */

// ── Imports ────────────────────────────────────────────────────────────────

// Dynamic import deferred until client – analytics composable is isomorphic
// but we only load data on mount (client-side) so this is fine.

// ── Props ─────────────────────────────────────────────────────────────────

const props = defineProps<{
  runId: string
}>()

// ── Analytics state ────────────────────────────────────────────────────────

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
  reset,
} = analytics

// Load on mount (client only – getResults is a $fetch call, needs runtime config)
onMounted(() => {
  load()
})

// ── Empty state helpers ────────────────────────────────────────────────────

const hasRows = computed(() => rows.value.length > 0)
const hasFiltered = computed(() => filtered.value.length > 0)
</script>

<template>
  <div class="space-y-3">
    <!-- Loading -->
    <div v-if="loading" class="flex items-center gap-2 text-xs text-dim py-4">
      <UIcon name="i-heroicons-arrow-path" class="animate-spin w-4 h-4" />
      Loading analytics data…
    </div>

    <!-- Fetch error -->
    <UAlert
      v-else-if="fetchError"
      color="warning"
      variant="subtle"
      :title="`Could not load results: ${fetchError}`"
    />

    <!-- No scored results yet -->
    <UAlert
      v-else-if="!hasRows"
      color="neutral"
      variant="subtle"
      title="No scored results yet — vehicles haven't reached a stop."
      description="Analytics will appear once the run has completed evaluations."
    />

    <!-- Dashboard -->
    <template v-else>
      <!-- Filter bar -->
      <AnalyticsFilterBar
        :predictors="predictors"
        :selected-predictors="selectedPredictors"
        :routes="routes"
        :selected-route="selectedRoute"
        :horizon-range="horizonRange"
        :color-map="colorMap"
        @toggle-predictor="togglePredictor"
        @select-route="selectRoute"
        @set-horizon-range="setHorizonRange"
        @reset="reset"
      />

      <!-- Empty filtered state -->
      <UAlert
        v-if="!hasFiltered"
        color="warning"
        variant="subtle"
        title="No rows match the current filters."
        description="Try broadening the predictor selection, route, or horizon range."
      />

      <template v-else>
        <!-- Row 1: Comparison table + Horizon curve -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
          <AnalyticsPredictorComparisonTable
            :data="predictorAggs"
            :color-map="colorMap"
          />
          <AnalyticsHorizonCurveChart
            :data="bucketData"
            :predictors="activePredictors"
            :color-map="colorMap"
          />
        </div>

        <!-- Row 2: Scatter + Convergence -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
          <AnalyticsErrorScatter
            :data="filtered"
            :predictors="activePredictors"
            :color-map="colorMap"
            @select-trip-stop="selectTripStop"
          />
          <AnalyticsConvergenceChart
            :data="convergenceData"
            :predictors="activePredictors"
            :color-map="colorMap"
            :selected-trip-stop="selectedTripStop"
            @clear="clearTripStop"
          />
        </div>

        <!-- Row 3: Error distribution + Route breakdown -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
          <AnalyticsErrorDistribution
            :data="histData"
            :predictors="activePredictors"
            :color-map="colorMap"
          />
          <AnalyticsRouteBreakdown
            :data="routeData"
            :predictors="activePredictors"
            :color-map="colorMap"
          />
        </div>

        <!-- Row 4: Head-to-head (full width) -->
        <AnalyticsHeadToHead
          :data="headToHeadData"
          :color-map="colorMap"
        />
      </template>
    </template>
  </div>
</template>
