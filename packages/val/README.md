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

Beyond the headless core (domain + storage + feed adapters + spatial + predictors +
ground-truth + scoring + engine + Typer CLI), etaval now ships:

- a **FastAPI REST + WebSocket API** (`etaval/api/`) for starting/monitoring/streaming
  live runs, with a process-wide telemetry endpoint (`GET /api/telemetry`,
  `GET /api/telemetry/events`) that surfaces tick health, feed-poll health, and
  swallowed predictor/detector error counts;
- a **Nuxt web UI** (`web/`) with four views — a live-run console, a runs list/detail
  page, a results explorer, and a telemetry dashboard;
- a **Textual TUI** (`etaval/tui/`) for monitoring runs from the terminal.

## Architecture

Three pluggable seams are the design; everything else is plumbing:

- **`Predictor`** — produces arrival predictions (`gtfsrt`, `baseline:constant_speed`,
  and `gtfs_eta:{historical_mean,polyreg_distance,polyreg_time,xgboost}` — trained
  models from the sibling `gtfs-eta` repo, see below). A predictor opts into receiving
  vehicle position history by declaring `accepts_position_history = True` (see
  `ConstantSpeedPredictor`); the engine dispatches on that attribute rather than
  inspecting the `predict` signature.
- **`GroundTruthDetector`** — detects actual arrivals (`shape_distance` default,
  `position_radius`, `stop_status`).
- **`Storage`** — persists samples / arrivals / results (SQLite, Parquet export).

Predictor and detector exceptions are swallowed per-tick (one flaky predictor must not
take down a run) but are never silent: `RunSummary.predictor_errors` /
`.detector_errors` (and the matching `TickEvent` cumulative snapshots) count
exceptions per predictor-id / detector-method, surfaced through the CLI run summary,
the REST API (`RunInfo.summary`), the WebSocket stream, and `/api/telemetry`.

### Validating gtfs-eta's trained models

`etaval/predictors/gtfs_eta_models.py` wraps `eta_service.estimator.estimate_stop_times`
from `../gtfs-eta` (wired as a `[tool.uv.sources]` path dependency, same pattern as
`gtfs-rt-pipeline` -> `gtfs-django`) so any model trained there can be scored the same
way as `gtfsrt`/`baseline:constant_speed`. Note `gtfs_eta:ewma` was retired upstream
and is not in `PREDICTOR_REGISTRY` — the live model families are `historical_mean`,
`polyreg_distance`, `polyreg_time`, and `xgboost`.

`GtfsEtaModelPredictor._resolve_model_registry_dir()` locates the trained-model
registry automatically in a packaged (non-editable) install: it tries, in order, an
explicit `MODEL_REGISTRY_DIR` env var, a sibling `../gtfs-eta` source checkout, the
installed `gtfs-eta` distribution's dist-info (`direct_url.json`), and finally the
installed top-level `models` package — each candidate is verified against a real
`registry.json` before being trusted. **You normally don't need to set
`MODEL_REGISTRY_DIR` yourself** — it's just an override for the rare case none of
those fallbacks locate a real registry (see below).

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
here to configure beyond which `model_type`s to include. `historical_mean` predicts
per (route, stop_sequence, hour) rather than from live kinematics, so scoring it
side-by-side with `polyreg_time`/`xgboost` is exactly how to judge whether it's worth
keeping vs. cutting.

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
> copied in (rather than editable), naive `Path(__file__)`-based path resolution can no
> longer locate gtfs-eta's real `models/trained/` (the installed copy's file location
> says nothing about where the source tree lives). `gtfs_eta_models.py` works around
> this with `GtfsEtaModelPredictor._resolve_model_registry_dir()` (see above), which
> checks a sibling source checkout, gtfs-eta's installed dist-info, and the installed
> `models` package in turn — this is auto-resolved in the common cases (source-tree dev
> setup, or a fully packaged install) with **no action needed**. Only if none of those
> fallbacks find a real `registry.json` do you need to override it explicitly:
> ```bash
> export MODEL_REGISTRY_DIR=/absolute/path/to/gtfs-eta/models/trained
> ```
> Without a resolvable registry, `gtfs_eta:*` predictors emit a `RuntimeWarning` at
> import time and every prediction comes back empty (`error: 'No trained models found
> for model_type'`) — not a crash, just silently no predictions, so watch for the
> warning.

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

## Walk-forward backtest and the 70/10/20 split

`etaval walk-forward` (`etaval/backtest.py`) splits an archived replay window
into contiguous, non-overlapping sub-windows — either an equal COUNT
(`--windows`) or a fixed WIDTH (`--width`) — and runs a fresh replay per
window with a shared `--seed`, reducing each window's ValidationResults to a
per-predictor MAE with a bootstrap confidence interval
(`etaval.stats.bootstrap.bootstrap_ci`). The output JSON carries per-window
MAE+CI, window boundaries, each window's fraction of the overall range, run
provenance, the seed, and a `split_note`.

**Alignment requirement with gtfs-eta's train/val/test split.** The
`gtfs_eta:*` predictors wrap models trained in the sibling `../gtfs-eta`
checkout, which reserves a canonical chronological train/validation/test
split of its own training data (`models/common/data.py`'s `temporal_split`,
which by default holds out the most-recent slice of the timeline as test —
etaval treats this as a 70/10/20-style train/val/test split and never
modifies gtfs-eta or re-derives that split itself). For a walk-forward run
here to be a *valid, leakage-free held-out evaluation* of any `gtfs_eta:*`
predictor, every backtest window's `[window_start, window_end]` must fall
entirely inside that model's reserved **test** slice — a window overlapping
the train/val slices would let the model be evaluated on data it was
directly or indirectly fitted on, inflating apparent accuracy. Predictors
with no training split (`gtfsrt`, `baseline:*`) have no such constraint.

It is the **caller's** responsibility to choose `--replay-start`/
`--replay-end` (or the `start`/`end` passed to `walk_forward()`) so they lie
within the trained model's reserved test window — `walk_forward()` records
this requirement in the result's `split_note` field, it does not verify it
(etaval has no visibility into which archived rows gtfs-eta's `temporal_split`
put in train vs. val vs. test at training time).

```bash
uv run --no-editable etaval walk-forward wf.json \
  --source mbta --replay-vp s3://bucket/mbta/vp --replay-gtfs mbta.zip \
  --replay-start 2024-06-01T00:00:00Z --replay-end 2024-06-08T00:00:00Z \
  --windows 7 --predictor gtfsrt --predictor baseline:constant_speed --seed 0
```

## Publication figures

`etaval figures <parquet> --out-dir <dir>` (`etaval/figures.py`) renders six
300 dpi PNG figures from a Parquet results file using a colorblind-safe
Okabe-Ito-derived palette (headless `Agg` backend, no seaborn):
`error_vs_horizon`, `error_distribution`, `convergence_curve`,
`per_route_mae`, `calibration_diagram`, and `cost_accuracy_pareto`. The
Pareto figure (inference latency vs. MAE) is only produced when `--summary
<...>.summary.json` is given — without it, a run has no recorded per-predictor
latency to plot, so it is skipped (returns `None`) rather than drawn empty.

```bash
uv run --no-editable etaval figures r.parquet --out-dir figs/ --summary r.parquet.summary.json
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
