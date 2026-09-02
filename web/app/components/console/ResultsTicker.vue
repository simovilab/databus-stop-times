<script setup lang="ts">
import type { ResultRow } from '~/types/api'

const props = defineProps<{
  results: ResultRow[]
  limit?: number
}>()

const rows = computed(() => {
  const lim = props.limit ?? 20
  return props.results.slice().reverse().slice(0, lim)
})

function fmtNum(n: number | undefined): string {
  if (n === undefined || n === null) return '—'
  return n.toFixed(1)
}
</script>

<template>
  <div class="panel">
    <div class="panel-header">
      <span class="micro-label">Live result feed</span>
      <span class="micro-label">last {{ results.length }}</span>
    </div>
    <div v-if="rows.length === 0" class="text-xs text-dim py-6 text-center">
      Waiting for first results&hellip;
    </div>
    <div v-else class="overflow-x-auto max-h-80 overflow-y-auto">
      <table class="w-full text-[11px]">
        <thead>
          <tr class="border-b border-line text-dim uppercase tracking-wide text-[10px]">
            <th class="text-left py-1.5 px-3 font-normal">Predictor</th>
            <th class="text-left py-1.5 px-3 font-normal">Stop</th>
            <th class="text-right py-1.5 px-3 font-normal">Error (s)</th>
            <th class="text-right py-1.5 px-3 font-normal">Horizon (s)</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="(r, i) in rows"
            :key="`${r.trip_key}-${r.stop_id}-${r.predictor_id}-${i}`"
            class="border-b border-[#141f1a]"
          >
            <td class="py-1 px-3 text-bright">{{ r.predictor_id }}</td>
            <td class="py-1 px-3 text-dim">{{ r.stop_id }}</td>
            <td
              class="py-1 px-3 text-right tabular-nums"
              :class="Math.abs(r.error_s) > 60 ? 'text-err' : 'text-main'"
            >
              {{ fmtNum(r.error_s) }}
            </td>
            <td class="py-1 px-3 text-right tabular-nums text-dim">{{ fmtNum(r.horizon_s) }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>
