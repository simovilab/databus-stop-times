// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2025-07-15',

  modules: ['@nuxt/ui'],

  css: ['~/assets/css/main.css'],

  // Dark ops-console: force dark mode (ui.colorMode only accepts a boolean in @nuxt/ui 4.x)
  colorMode: {
    preference: 'dark',
    fallback: 'dark'
  },

  // System mono font is used everywhere; @nuxt/fonts would otherwise need network access
  fonts: false,

  // Env: NUXT_PUBLIC_API_BASE overrides at runtime
  runtimeConfig: {
    public: {
      apiBase: 'http://localhost:8010'
    }
  },

  typescript: {
    strict: true
  },

  devtools: { enabled: false }
})
