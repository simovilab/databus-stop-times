<script setup lang="ts">
const config = useRuntimeConfig()
const apiBase = config.public.apiBase as string

const healthOk = ref<boolean | null>(null)

async function pingHealth() {
  try {
    await $fetch('/api/health', { baseURL: apiBase })
    healthOk.value = true
  } catch {
    healthOk.value = false
  }
}

let timer: ReturnType<typeof setInterval> | undefined

onMounted(() => {
  pingHealth()
  timer = setInterval(pingHealth, 15000)
})

onUnmounted(() => {
  if (timer) clearInterval(timer)
})

const healthLedClass = computed(() => {
  if (healthOk.value === null) return 'status-led idle'
  return healthOk.value ? 'status-led ok' : 'status-led err'
})

const healthLabel = computed(() => {
  if (healthOk.value === null) return 'checking'
  return healthOk.value ? 'online' : 'offline'
})
</script>

<template>
  <UApp>
    <div class="min-h-screen bg-void text-main">
      <!-- Top navigation -->
      <header class="sticky top-0 z-10 border-b border-line bg-panel">
        <div class="flex items-center h-10 px-3 gap-4">
          <NuxtLink to="/" class="text-bright font-bold tracking-wide text-[13px] shrink-0">
            stoptimes val<span class="text-dim">://</span>
          </NuxtLink>
          <nav class="flex h-full">
            <NuxtLink to="/" class="nav-link" active-class="active" exact> Console </NuxtLink>
            <NuxtLink to="/runs" class="nav-link" active-class="active"> Leaderboard </NuxtLink>
            <NuxtLink to="/explorer" class="nav-link" active-class="active"> Explorer </NuxtLink>
            <NuxtLink to="/telemetry" class="nav-link" active-class="active"> Telemetry </NuxtLink>
          </nav>
          <div class="ml-auto flex items-center gap-2 text-[11px] text-dim shrink-0">
            <span class="micro-label">api</span>
            <span :class="healthLedClass" />
            <span>{{ healthLabel }}</span>
          </div>
        </div>
      </header>

      <!-- Page content -->
      <main class="w-full">
        <NuxtPage />
      </main>
    </div>
  </UApp>
</template>
