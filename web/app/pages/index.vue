<script setup lang="ts">
import type { Source } from '~/types/api'

useHead({ title: 'etaval:// console' })

const api = useApi()
const router = useRouter()

// ── Remote data ──────────────────────────────────────────────────────────────
const sources = ref<Source[]>([])
const predictors = ref<string[]>([])
const runsCount = ref<number>(0)
const activeRunsCount = ref<number>(0)
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
    const [srcs, preds, runs] = await Promise.all([
      api.getSources(),
      api.getPredictors(),
      api.getRuns()
    ])
    sources.value = srcs
    predictors.value = preds
    runsCount.value = runs.length
    activeRunsCount.value = runs.filter((r) => r.status === 'pending' || r.status === 'running').length
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
  <div class="p-3 space-y-3">
    <div class="sec-h">New run</div>

    <UAlert v-if="loadError" color="error" variant="subtle" :title="loadError" />

    <div class="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-3 items-start">
      <!-- LEFT: terminal-style form -->
      <div class="panel p-4">
        <form class="space-y-5" @submit.prevent="handleSubmit">
          <!-- Source -->
          <div>
            <label class="field-label">Source//</label>
            <USelect
              v-model="selectedSource"
              :items="sourceItems"
              :disabled="sources.length === 0"
              placeholder="select a source…"
              class="w-full"
            />
          </div>

          <!-- Predictors -->
          <div>
            <label class="field-label">Predictors//</label>
            <div v-if="predictors.length === 0" class="text-xs text-dim">
              Loading predictors&hellip;
            </div>
            <div v-else class="grid grid-cols-2 sm:grid-cols-3 gap-2">
              <label
                v-for="pred in predictors"
                :key="pred"
                class="chip cursor-pointer"
                :class="{ on: selectedPredictors.includes(pred) }"
              >
                <input
                  type="checkbox"
                  :value="pred"
                  :checked="selectedPredictors.includes(pred)"
                  class="sr-only"
                  @change="togglePredictor(pred)"
                />
                <span class="truncate">{{ pred }}</span>
              </label>
            </div>
            <p v-if="selectedPredictors.length === 0 && predictors.length > 0" class="mt-2 text-xs text-err">
              Select at least one predictor.
            </p>
          </div>

          <!-- Routes -->
          <div>
            <label class="field-label">Routes// <span class="normal-case text-dim">(optional, comma-separated)</span></label>
            <UInput
              v-model="routesInput"
              placeholder="e.g. 1,2,Red — leave empty for all routes"
              class="w-full"
            />
          </div>

          <!-- Duration -->
          <div>
            <label class="field-label">Duration//</label>
            <UInput v-model="duration" placeholder="e.g. 30m, 1h" class="w-full" />
            <p class="mt-1 text-xs text-dim">
              examples: <code class="font-mono">5m</code>, <code class="font-mono">30m</code>,
              <code class="font-mono">1h</code>
            </p>
          </div>

          <UAlert v-if="submitError" color="error" variant="subtle" :title="submitError" />

          <UButton
            type="submit"
            label="▶ EXECUTE RUN"
            :loading="submitting"
            :disabled="!formValid || submitting"
            color="primary"
            size="lg"
            block
            class="tracking-widest uppercase text-xs font-bold"
          />
        </form>
      </div>

      <!-- RIGHT: status tiles + pipeline kv -->
      <div class="space-y-3">
        <div class="grid grid-cols-2 gap-px bg-line">
          <div class="tile bg-panel px-3 py-2">
            <div class="tile-l">Sources</div>
            <div class="tile-v">{{ sources.length }}</div>
          </div>
          <div class="tile bg-panel px-3 py-2">
            <div class="tile-l">Predictors</div>
            <div class="tile-v">{{ predictors.length }}</div>
          </div>
          <div class="tile bg-panel px-3 py-2">
            <div class="tile-l">Active runs</div>
            <div class="tile-v" :class="activeRunsCount > 0 ? 'text-accent' : ''">{{ activeRunsCount }}</div>
          </div>
          <div class="tile bg-panel px-3 py-2">
            <div class="tile-l">Runs recorded</div>
            <div class="tile-v">{{ runsCount }}</div>
          </div>
        </div>

        <div class="panel">
          <div class="panel-header">
            <span class="micro-label">Pipeline</span>
          </div>
          <div class="px-3 py-2">
            <div class="kv-row"><span class="k">Feed</span><span class="v">{{ selectedSource || '—' }}</span></div>
            <div class="kv-row"><span class="k">Predictors</span><span class="v">{{ selectedPredictors.length }} active</span></div>
            <div class="kv-row"><span class="k">Routes</span><span class="v">{{ routesInput.trim() || 'all' }}</span></div>
            <div class="kv-row"><span class="k">Duration</span><span class="v">{{ duration || '—' }}</span></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
