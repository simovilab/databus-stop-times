<script setup lang="ts">
import type { RunSummary } from '~/types/api'

defineProps<{
  summary: RunSummary
}>()
</script>

<template>
  <div class="grid grid-cols-2 sm:grid-cols-4 gap-px bg-line">
    <div class="tile bg-panel px-3 py-2">
      <div class="tile-l">Ticks</div>
      <div class="tile-v">{{ summary.ticks }}</div>
    </div>
    <div class="tile bg-panel px-3 py-2">
      <div class="tile-l">Samples</div>
      <div class="tile-v">{{ summary.samples_saved }}</div>
    </div>
    <div class="tile bg-panel px-3 py-2">
      <div class="tile-l">Arrivals</div>
      <div class="tile-v">{{ summary.arrivals_saved }}</div>
    </div>
    <div class="tile bg-panel px-3 py-2">
      <div class="tile-l">Results</div>
      <div class="tile-v">{{ summary.results_saved }}</div>
    </div>
  </div>
</template>
