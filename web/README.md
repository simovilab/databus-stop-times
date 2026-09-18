# stoptimes val web UI

Nuxt 4 frontend for the `stoptimes val` (formerly `etaval`) ETA validation platform. It talks to
the `stoptimes.val.api` FastAPI backend only through the REST + WebSocket contract in
`app/types/api.ts` / `app/composables/useApi.ts` / `app/composables/useRunStream.ts` —
nothing here names an agency, a file path, or a CLI command outside this README, so it
needs no changes as the rest of the toolkit (data/train/serve) is built out in later
phases.

## Prerequisites

- Node.js v20+ (developed on v24)
- npm 10+ (developed on npm 11)
- The `stoptimes val` API backend (FastAPI), from this same uv workspace

## Quick start

### 1. Start the backend

From the repo root (the `stoptimes` uv workspace):

```bash
uv sync --group dev            # once, or `uv sync --extra val` from packages/cli
uv run --no-sync stoptimes val api
# → listens on http://localhost:8010 by default
# → set ETAVAL_API_PORT to override (see packages/val/src/stoptimes/val/api/config.py)
```

### 2. Install frontend dependencies

```bash
cd web
npm install
```

### 3. Run the dev server

```bash
npm run dev
# → http://localhost:3000
```

CORS on the backend already allows `http://localhost:3000`, so no proxy is needed.

### 4. Override the API base URL

The default API URL is `http://localhost:8010`.  
To point at a different host:

```bash
NUXT_PUBLIC_API_BASE=http://my-server:8010 npm run dev
```

Or create a `.env` file:

```env
NUXT_PUBLIC_API_BASE=http://my-server:8010
```

## Build for production

```bash
npm run build
npm run preview
```

## Project structure

```
web/
├── app/
│   ├── app.vue                  # Root component (UApp + top nav)
│   ├── types/
│   │   └── api.ts               # TypeScript interfaces matching backend contracts
│   ├── composables/
│   │   ├── useApi.ts            # Typed REST API wrappers ($fetch)
│   │   └── useRunStream.ts      # WebSocket composable for live run streaming
│   └── pages/
│       ├── index.vue            # CONSOLE — create + watch runs
│       ├── explorer.vue         # EXPLORER — cross-filtered analytics
│       ├── telemetry.vue        # TELEMETRY — system health + event log
│       └── runs/
│           ├── index.vue        # LEADERBOARD — run history
│           └── [id].vue         # Run detail (live + done + error views)
├── nuxt.config.ts
├── package.json
└── README.md
```

## Design language

The UI is a dark ops-console (dense, monospace, sharp corners, green accent).
Four views mirror the Textual TUI: CONSOLE, LEADERBOARD, EXPLORER, TELEMETRY.
Theme tokens live in `app/assets/css/main.css` (`@theme` block) and dark mode is
forced via the top-level `colorMode` key in `nuxt.config.ts`.

The `runs/[id].vue` done view wires `@unovis/vue` charts (horizon-bucket bars,
convergence lines, head-to-head scatter, error-distribution histograms) through
the analytics components under `app/components/analytics/`. Both `@unovis/vue`
and `@unovis/ts` are installed.
