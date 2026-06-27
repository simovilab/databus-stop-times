# etaval web UI

Nuxt 4 frontend for the etaval ETA validation platform.

## Prerequisites

- Node.js v20+ (developed on v24)
- npm 10+ (developed on npm 11)
- The etaval Python backend (FastAPI)

## Quick start

### 1. Start the backend

From the repo root (`docs/eta-val/`):

```bash
uv run --extra api etaval-api
# → listens on http://localhost:8010 by default
# → set ETAVAL_API_PORT to override
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
│       ├── index.vue            # New Run form (configure + start)
│       └── runs/
│           ├── index.vue        # History table
│           └── [id].vue         # Run detail (live + done + error views)
├── nuxt.config.ts
├── package.json
└── README.md
```

## Phase C2 (coming next)

The `runs/[id].vue` page contains a clearly-marked placeholder section
(`<!-- C2: analytics dashboard goes here -->`) where `@unovis/vue` charts
will be wired in: horizon-bucket bar charts, convergence lines,
head-to-head scatter plots, and error-distribution histograms.

Both `@unovis/vue` and `@unovis/ts` are already installed.
