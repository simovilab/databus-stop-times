<script setup lang="ts">
import type { RunInfo } from '~/types/api'

useHead({ title: 'explorer – stoptimes val' })

const api = useApi()

const runs = ref<RunInfo[]>([])
const loadError = ref<string | null>(null)
const selectedRunId = ref<string>('')

onMounted(async () => {
  try {
    const all = await api.getRuns()
    runs.value = all
      .filter((r) => r.status === 'done')
      .sort((a, b) => (b.started_at ?? '').localeCompare(a.started_at ?? ''))
    if (runs.value.length > 0) selectedRunId.value = runs.value[0]!.id
  } catch {
    loadError.value = 'Could not load runs. Is the backend running?'
  }
})

const runItems = computed(() => runs.value.map((r) => ({ label: r.id, value: r.id })))
</script>

<template>
  <div class="p-3 space-y-3">
    <div class="flex items-center gap-3">
      <div class="sec-h flex-1">Explorer</div>
      <USelect
        v-model="selectedRunId"
        :items="runItems"
        :disabled="runs.length === 0"
        placeholder="select a completed run…"
        class="w-72"
      />
    </div>

    <UAlert v-if="loadError" color="error" variant="subtle" :title="loadError" />

    <div v-if="!loadError && runs.length === 0" class="text-center py-16 text-dim text-sm">
      No completed runs yet. Finish a run to explore its results here.
    </div>

    <ExplorerBody v-else-if="selectedRunId" :key="selectedRunId" :run-id="selectedRunId" />
  </div>
</template>
