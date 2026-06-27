<template>
  <UApp>
    <div class="min-h-screen bg-gray-50 dark:bg-gray-950">
      <!-- Top navigation -->
      <header class="border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 sticky top-0 z-10">
        <div class="max-w-6xl mx-auto px-4 py-3 flex items-center gap-6">
          <NuxtLink to="/" class="font-bold text-lg tracking-tight text-gray-900 dark:text-white">
            etaval
          </NuxtLink>
          <nav class="flex gap-4">
            <NuxtLink
              to="/"
              class="text-sm font-medium text-gray-600 hover:text-primary-500 dark:text-gray-400 dark:hover:text-primary-400"
              active-class="text-primary-600 dark:text-primary-400"
              exact
            >
              New Run
            </NuxtLink>
            <NuxtLink
              to="/runs"
              class="text-sm font-medium text-gray-600 hover:text-primary-500 dark:text-gray-400 dark:hover:text-primary-400"
              active-class="text-primary-600 dark:text-primary-400"
            >
              History
            </NuxtLink>
          </nav>
        </div>
      </header>

      <!-- Page content -->
      <main class="max-w-6xl mx-auto px-4 py-8">
        <NuxtPage />
      </main>
    </div>
  </UApp>
</template>
