// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2025-07-15',

  modules: ['@nuxt/ui'],

  css: ['~/assets/css/main.css'],

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
