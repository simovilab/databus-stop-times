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

- **`Predictor`** — produces arrival predictions (`gtfsrt`, `baseline:constant_speed`,
  and `gtfs_eta:{historical_mean,ewma,polyreg_distance,polyreg_time,xgboost}` — trained
  models from the sibling `gtfs-eta` repo, see below).
- **`GroundTruthDetector`** — detects actual arrivals (`shape_distance` default,
  `position_radius`, `stop_status`).
- **`Storage`** — persists samples / arrivals / results (SQLite, Parquet export).

### Validating gtfs-eta's trained models

`etaval/predictors/gtfs_eta_models.py` wraps `eta_service.estimator.estimate_stop_times`
from `../gtfs-eta` (wired as a `[tool.uv.sources]` path dependency, same pattern as
`gtfs-rt-pipeline` -> `gtfs-django`) so any model trained there can be scored the same
way as `gtfsrt`/`baseline:constant_speed`. `models.common.registry.ModelRegistry`'s
default path resolves via `gtfs-eta`'s own `core.config.get_config()` regardless of
etaval's cwd, so no `MODEL_REGISTRY_DIR` override is needed.

```bash
uv run --no-editable etaval run --source mbta \
  --predictor gtfsrt --predictor gtfs_eta:xgboost --predictor gtfs_eta:polyreg_time \
  --duration 10m --out r.parquet
uv run --no-editable etaval report r.parquet
```

**`--no-editable` is required** (see the environment note under Install) — without it
the `etaval` and `eta_prediction` console entrypoints fail with `ModuleNotFoundError`
on at least one dev machine.

Model selection is delegated to `models.common.registry.ModelRegistry.get_best_model`
(route-specific model preferred over global, ranked by `test_mae_seconds`) — nothing
here to configure beyond which `model_type`s to include. `historical_mean` and `ewma`
predict per (route, stop_sequence, hour) rather than from live kinematics, so scoring
them side-by-side with `polyreg_time`/`xgboost` is exactly how to judge whether they're
worth keeping vs. cutting.

Per tick: poll VP + TU feeds → update active `TripInstance`s and position history →
ask every `Predictor` for ETAs to upcoming stops → run every `GroundTruthDetector`
(fires `ActualArrival` when a stop is reached) → `Scorer` joins samples to actuals
and emits `ValidationResult`s → persist all three.

## Install

```bash
uv sync --no-editable
```

> **Known environment issue (Python 3.14 + uv-managed venv, confirmed on at least one
> dev machine):** `site.py`'s `.pth`-based editable-install activation silently never
> fires at interpreter startup — reproduced by invoking the venv's `python` binary
> directly with a stripped env, so it isn't a `uv run` wrapper quirk. Editable mode is
> uv's default for path/workspace sources, so a plain `uv sync` (or any `uv run`
> without `--no-editable`) re-reverts to the broken state and every `etaval`/
> `etaval-api` console-script invocation fails with `ModuleNotFoundError: No module
> named 'etaval'`. **Always pass `--no-editable`** to `uv sync`/`uv run` in this repo
> (or `export UV_NO_EDITABLE=1`) until this is root-caused. Note this means local edits
> to `etaval/` or `../gtfs-eta` need a re-sync (`uv sync --no-editable`) to take effect
> — there's no live-editable convenience while this workaround is in place.
>
> **Knock-on effect for `gtfs_eta:*` predictors:** once `eta_prediction` is physically
> copied in (rather than editable), `Path(__file__)`-based path resolution inside it —
> both `core.config._find_project_root()`'s walk-up-for-pyproject.toml and this repo's
> own `etaval/predictors/gtfs_eta_models.py` — can no longer locate gtfs-eta's real
> `models/trained/` (the installed copy's file location says nothing about where the
> source tree lives). `gtfs_eta_models.py` verifies its one in-source guess against a
> real `registry.json` and refuses to trust a wrong one, but that means **you must
> export `MODEL_REGISTRY_DIR` yourself** for the `gtfs_eta:*` predictors to find real
> trained models under this workaround:
> ```bash
> export MODEL_REGISTRY_DIR=/absolute/path/to/gtfs-eta/models/trained
> ```
> Without it, `gtfs_eta:*` predictors emit a `RuntimeWarning` at import time and every
> prediction comes back empty (`error: 'No trained models found for model_type'`) —
> not a crash, just silently no predictions, so watch for the warning.

## Usage

```bash
# List configured sources
uv run --no-editable etaval sources

# Run a live validation pass (network permitting), persist + export Parquet
uv run --no-editable etaval run --source mbta \
  --predictor gtfsrt --predictor baseline:constant_speed \
  --duration 10m --out r.parquet

# Report per-predictor MAE + horizon breakdown + model comparison
uv run --no-editable etaval report r.parquet
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
