<script setup lang="ts">
import type { Source } from '~/types/api'

useHead({ title: 'New Run – etaval' })

const api = useApi()
const router = useRouter()

// ── Remote data ──────────────────────────────────────────────────────────────
const sources = ref<Source[]>([])
const predictors = ref<string[]>([])
const loadError = ref<string | null>(null)

// ── Form state ────────────────────────────────────────────────────────────────
const selectedSource = ref<string>('')
const selectedPredictors = ref<string[]>([])
const routesInput = ref('')
const duration = ref('30m')

// ── Submit state ──────────────────────────────────────────────────────────────
const submitting = ref(false)
const submitError = ref<string | null>(null)

// ── Validation ────────────────────────────────────────────────────────────────
const formValid = computed(
  () => selectedSource.value && selectedPredictors.value.length > 0 && duration.value.trim()
)

onMounted(async () => {
  try {
    const [srcs, preds] = await Promise.all([api.getSources(), api.getPredictors()])
    sources.value = srcs
    predictors.value = preds
    // Default: first source, all predictors selected
    if (srcs.length > 0) selectedSource.value = srcs[0]!.name
    selectedPredictors.value = [...preds]
  } catch {
    loadError.value = 'Could not reach the etaval API. Is the backend running on port 8010?'
  }
})

// ── Source select items ───────────────────────────────────────────────────────
const sourceItems = computed(() =>
  sources.value.map((s) => ({ label: s.name, value: s.name }))
)

// ── Checkbox helpers ──────────────────────────────────────────────────────────
function togglePredictor(pred: string) {
  const idx = selectedPredictors.value.indexOf(pred)
  if (idx === -1) {
    selectedPredictors.value = [...selectedPredictors.value, pred]
  } else {
    selectedPredictors.value = selectedPredictors.value.filter((p) => p !== pred)
  }
}

// ── Submit ────────────────────────────────────────────────────────────────────
async function handleSubmit() {
  if (!formValid.value) return
  submitting.value = true
  submitError.value = null

  const routes = routesInput.value.trim()
    ? routesInput.value.split(',').map((r) => r.trim()).filter(Boolean)
    : []

  try {
    const { run_id } = await api.createRun({
      source: selectedSource.value,
      predictors: selectedPredictors.value,
      routes,
      duration: duration.value.trim()
    })
    await router.push(`/runs/${run_id}`)
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err)
    submitError.value = `Failed to start run: ${msg}`
    submitting.value = false
  }
}
</script>

<template>
  <div class="max-w-xl">
    <h1 class="text-2xl font-bold text-gray-900 dark:text-white mb-6">
      Configure a new run
    </h1>

    <!-- API error banner -->
    <UAlert
      v-if="loadError"
      color="error"
      variant="subtle"
      :title="loadError"
      class="mb-6"
    />

    <form class="space-y-6" @submit.prevent="handleSubmit">
      <!-- Source -->
      <div>
        <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
          Source
        </label>
        <USelect
          v-model="selectedSource"
          :items="sourceItems"
          :disabled="sources.length === 0"
          placeholder="Select a source…"
          class="w-full"
        />
      </div>

      <!-- Predictors -->
      <div>
        <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Predictors
        </label>
        <div v-if="predictors.length === 0" class="text-sm text-gray-500">
          Loading predictors…
        </div>
        <div v-else class="space-y-2">
          <label
            v-for="pred in predictors"
            :key="pred"
            class="flex items-center gap-3 cursor-pointer select-none"
          >
            <input
              type="checkbox"
              :value="pred"
              :checked="selectedPredictors.includes(pred)"
              class="h-4 w-4 rounded border-gray-300 text-primary-600 focus:ring-primary-500"
              @change="togglePredictor(pred)"
            />
            <span class="text-sm text-gray-800 dark:text-gray-200 font-mono">{{ pred }}</span>
          </label>
        </div>
        <p v-if="selectedPredictors.length === 0 && predictors.length > 0" class="mt-1 text-xs text-red-500">
          Select at least one predictor.
        </p>
      </div>

      <!-- Routes (optional) -->
      <div>
        <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
          Routes
          <span class="font-normal text-gray-400">(optional, comma-separated)</span>
        </label>
        <UInput
          v-model="routesInput"
          placeholder="e.g. 1,2,Red (leave empty for all routes)"
          class="w-full"
        />
      </div>

      <!-- Duration -->
      <div>
        <label class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
          Duration
        </label>
        <UInput
          v-model="duration"
          placeholder="e.g. 30m, 1h"
          class="w-full"
        />
        <p class="mt-1 text-xs text-gray-500">
          Examples: <code class="font-mono">5m</code>, <code class="font-mono">30m</code>,
          <code class="font-mono">1h</code>
        </p>
      </div>

      <!-- Submit error -->
      <UAlert
        v-if="submitError"
        color="error"
        variant="subtle"
        :title="submitError"
      />

      <!-- Submit button -->
      <UButton
        type="submit"
        label="Start run"
        :loading="submitting"
        :disabled="!formValid || submitting"
        color="primary"
        size="lg"
        class="w-full justify-center"
      />
    </form>
  </div>
</template>
