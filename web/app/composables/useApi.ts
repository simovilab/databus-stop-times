import type {
  Source,
  RunInfo,
  RunMetrics,
  ResultRow,
  CreateRunRequest,
  CreateRunResponse
} from '~/types/api'

/**
 * Typed $fetch wrappers for the stoptimes val backend REST API.
 * All paths are relative to NUXT_PUBLIC_API_BASE (default http://localhost:8010).
 * Errors propagate as FetchError – callers should try/catch.
 */
export function useApi() {
  const config = useRuntimeConfig()
  const baseURL = config.public.apiBase as string

  async function getSources(): Promise<Source[]> {
    return $fetch<Source[]>('/api/sources', { baseURL })
  }

  async function getPredictors(): Promise<string[]> {
    return $fetch<string[]>('/api/predictors', { baseURL })
  }

  async function createRun(req: CreateRunRequest): Promise<CreateRunResponse> {
    return $fetch<CreateRunResponse>('/api/runs', {
      method: 'POST',
      body: req,
      baseURL
    })
  }

  async function getRuns(): Promise<RunInfo[]> {
    return $fetch<RunInfo[]>('/api/runs', { baseURL })
  }

  async function getRun(id: string): Promise<RunInfo> {
    return $fetch<RunInfo>(`/api/runs/${id}`, { baseURL })
  }

  async function getMetrics(id: string, withStats = false): Promise<RunMetrics> {
    return $fetch<RunMetrics>(`/api/runs/${id}/metrics`, {
      baseURL,
      query: withStats ? { stats: 1 } : undefined
    })
  }

  async function getResults(
    id: string,
    opts?: { predictor?: string; limit?: number }
  ): Promise<ResultRow[]> {
    return $fetch<ResultRow[]>(`/api/runs/${id}/results`, {
      baseURL,
      query: opts
    })
  }

  async function deleteRun(id: string): Promise<void> {
    await $fetch(`/api/runs/${id}`, { method: 'DELETE', baseURL })
  }

  return {
    getSources,
    getPredictors,
    createRun,
    getRuns,
    getRun,
    getMetrics,
    getResults,
    deleteRun
  }
}
