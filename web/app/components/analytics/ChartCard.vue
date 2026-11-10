<script setup lang="ts">
// ChartCard – panel chrome shared by every analytics chart.
// The header carries a title plus a one-line takeaway (`lede` slot), so a
// reader gets the answer before reading the axes.

withDefaults(defineProps<{
  title: string
  meta?: string
  empty?: boolean
  emptyText?: string
  height?: number
}>(), {
  meta: undefined,
  empty: false,
  emptyText: 'No data for the selected filters.',
  height: 240,
})
</script>

<template>
  <section class="chart-card">
    <header class="chart-card__head">
      <div class="min-w-0">
        <h3 class="chart-card__title">{{ title }}</h3>
        <p v-if="$slots.lede" class="chart-card__lede"><slot name="lede" /></p>
      </div>
      <div class="flex items-center gap-2">
        <span v-if="meta" class="chart-card__meta">{{ meta }}</span>
        <slot name="actions" />
      </div>
    </header>
    <div class="chart-card__body">
      <div v-if="empty" class="chart-empty" :style="{ height: `${height}px` }">
        <slot name="empty">{{ emptyText }}</slot>
      </div>
      <slot v-else />
    </div>
    <footer v-if="$slots.foot" class="chart-card__foot"><slot name="foot" /></footer>
  </section>
</template>
