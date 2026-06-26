# etaval — ETA Prediction Validation Tool

`etaval` measures **how good transit arrival predictions are**. For a live trip it
collects the arrival predictions published throughout the trip (from GTFS-RT
TripUpdates and from a built-in baseline model), independently detects when the
vehicle actually reaches each stop, and scores the predictions against that ground
truth — with emphasis on **how error evolves with prediction horizon**.

It is a **fully standalone tool**. It reads only **published GTFS-RT JSON feeds**
plus the GTFS static `.zip`, exactly as any agency serves them. Adding an agency is
just adding a YAML source config.

## Status

Headless vertical slice (build phases **P0–P6**): domain + storage + feed adapters +
spatial + predictors + ground-truth + scoring + engine + Typer CLI, with tests and
captured feed fixtures. The FastAPI/WebSocket API and Nuxt/MapLibre UI (P7–P8) are a
deliberate follow-up.

## Architecture

Three pluggable seams are the design; everything else is plumbing:

- **`Predictor`** — produces arrival predictions (`gtfsrt`, `baseline:constant_speed`).
- **`GroundTruthDetector`** — detects actual arrivals (`shape_distance` default,
  `position_radius`, `stop_status`).
- **`Storage`** — persists samples / arrivals / results (SQLite, Parquet export).

Per tick: poll VP + TU feeds → update active `TripInstance`s and position history →
ask every `Predictor` for ETAs to upcoming stops → run every `GroundTruthDetector`
(fires `ActualArrival` when a stop is reached) → `Scorer` joins samples to actuals
and emits `ValidationResult`s → persist all three.

## Install

```bash
uv sync
```

## Usage

```bash
# List configured sources
uv run etaval sources

# Run a live validation pass (network permitting), persist + export Parquet
uv run etaval run --source mbta \
  --predictor gtfsrt --predictor baseline:constant_speed \
  --duration 10m --out r.parquet

# Report per-predictor MAE + horizon breakdown + model comparison
uv run etaval report r.parquet
```

## Configured sources

- **`mbta`** — MBTA public GTFS-RT JSON + GTFS static.
- **`bucr`** — databus dev stack (base URL configurable, default `http://localhost:8000`).

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
```

## License

MIT
