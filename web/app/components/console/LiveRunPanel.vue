<script setup lang="ts">
import type { RunStatus } from '~/types/api'

const props = defineProps<{
  status: RunStatus
  connected: boolean
  vehicleCount: number
}>()

const ledClass = computed(() => {
  if (props.status === 'error') return 'status-led err'
  if (props.status === 'running') return 'status-led ok'
  return 'status-led warn'
})
</script>

<template>
  <div class="panel px-3 py-2 flex items-center gap-4 text-[11px]">
    <span :class="ledClass" />
    <span class="micro-label">status</span>
    <span class="text-bright uppercase">{{ status }}</span>
    <span class="micro-label">ws</span>
    <span :class="connected ? 'text-accent' : 'text-dim'">{{ connected ? 'connected' : 'disconnected' }}</span>
    <span class="micro-label">vehicles</span>
    <span class="text-bright tabular-nums">{{ vehicleCount }}</span>
  </div>
</template>
