<script setup lang="ts">
import type { PerPredictorMetric } from '~/types/api'

const props = defineProps<{
  winner: PerPredictorMetric | null
}>()

function fmtNum(n: number | undefined): string {
  if (n === undefined || n === null || !isFinite(n)) return '—'
  return n.toFixed(1)
}
</script>

<template>
  <div v-if="winner" class="panel-spot px-4 py-3">
    <div class="micro-label text-accent">Best predictor</div>
    <div class="text-2xl text-bright mt-1">{{ winner.predictor_id }}</div>
    <div class="text-3xl text-accent mt-1">
      {{ fmtNum(winner.mae) }}<span class="text-sm text-dim ml-1">s MAE</span>
    </div>
    <div class="grid grid-cols-2 sm:grid-cols-4 gap-x-6 mt-3">
      <div class="kv-row" style="border:none">
        <span class="k">RMSE</span><span class="v">{{ fmtNum(winner.rmse) }}s</span>
      </div>
      <div class="kv-row" style="border:none">
        <span class="k">MAPE</span><span class="v">{{ fmtNum(winner.mape) }}%</span>
      </div>
      <div class="kv-row" style="border:none">
        <span class="k">Bias</span><span class="v">{{ fmtNum(winner.bias) }}s</span>
      </div>
      <div class="kv-row" style="border:none">
        <span class="k">Count</span><span class="v">{{ winner.count.toLocaleString() }}</span>
      </div>
    </div>
  </div>
</template>
