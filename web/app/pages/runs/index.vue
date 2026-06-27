<script setup lang="ts">
import type { TableRow } from '@nuxt/ui'
import type { RunInfo } from '~/types/api'

useHead({ title: 'History – etaval' })

const api = useApi()
const router = useRouter()

const runs = ref<RunInfo[]>([])
const loading = ref(false)
const loadError = ref<string | null>(null)

type BadgeColor = 'success' | 'info' | 'warning' | 'error' | 'neutral'
const statusColors: Record<string, BadgeColor> = {
  pending: 'warning',
  running: 'info',
  done: 'success',
  error: 'error'
}

async function loadRuns() {
  loading.value = true
  loadError.value = null
  try {
    runs.value = await api.getRuns()
    // Most recent first
    runs.value = [...runs.value].sort((a, b) => {
      const ta = a.started_at ?? ''
      const tb = b.started_at ?? ''
      return tb.localeCompare(ta)
    })
  } catch {
    loadError.value = 'Could not load runs. Is the backend running?'
  } finally {
    loading.value = false
  }
}

onMounted(loadRuns)

function fmtDate(iso: string | null): string {
  if (!iso) return '—'
  return new Date(iso).toLocaleString()
}

function fmtPredictors(preds: string[]): string {
  return preds.join(', ')
}

// TanStack Table column definitions for UTable
const columns = [
  { accessorKey: 'status', header: 'Status' },
  {
    id: 'source',
    header: 'Source',
    accessorFn: (row: RunInfo) => row.params.source
  },
  {
    id: 'predictors',
    header: 'Predictors',
    accessorFn: (row: RunInfo) => fmtPredictors(row.params.predictors)
  },
  {
    id: 'counts',
    header: 'Ticks / Samples / Results',
    accessorFn: (row: RunInfo) =>
      `${row.summary.ticks} / ${row.summary.samples_saved} / ${row.summary.results_saved}`
  },
  {
    id: 'started_at',
    header: 'Started',
    accessorFn: (row: RunInfo) => fmtDate(row.started_at)
  }
]
</script>

<template>
  <div>
    <div class="flex items-center justify-between mb-6">
      <h1 class="text-2xl font-bold text-gray-900 dark:text-white">Run history</h1>
      <UButton
        label="Refresh"
        variant="outline"
        icon="i-heroicons-arrow-path"
        :loading="loading"
        @click="loadRuns"
      />
    </div>

    <UAlert
      v-if="loadError"
      color="error"
      variant="subtle"
      :title="loadError"
      class="mb-4"
    />

    <!-- Empty state -->
    <div
      v-if="!loading && !loadError && runs.length === 0"
      class="text-center py-16 text-gray-500"
    >
      <p class="text-lg">No runs yet.</p>
      <p class="text-sm mt-2">
        <NuxtLink to="/" class="text-primary-500 hover:underline">Start one →</NuxtLink>
      </p>
    </div>

    <!-- Runs table -->
    <UTable
      v-if="runs.length > 0"
      :data="runs"
      :columns="columns"
      :loading="loading"
      class="cursor-pointer"
      @select="(_e: Event, row: TableRow<RunInfo>) => router.push(`/runs/${row.original.id}`)"
    >
      <!-- Status badge -->
      <template #status-cell="{ row }">
        <UBadge
          :color="statusColors[row.original.status] ?? 'neutral'"
          variant="subtle"
          :label="row.original.status"
        />
      </template>
    </UTable>
  </div>
</template>
