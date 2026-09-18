# stoptimes val — ETA Prediction Validation Tool

`stoptimes val` (module `stoptimes.val`, formerly the standalone `etaval` tool)
measures **how good transit arrival predictions are**. For a live trip it collects
the arrival predictions published throughout the trip (from GTFS-RT TripUpdates and
from a built-in baseline model), independently detects when the vehicle actually
reaches each stop, and scores the predictions against that ground truth — with
emphasis on **how error evolves with prediction horizon**.

It is a **fully standalone tool** that validates *any* GTFS-RT TripUpdates
source — a shipped agency profile, an ad hoc third-party feed URL, a trained
`stoptimes` artifact, or your own serving pipeline (see "Predictors" and
"Validating any TripUpdates source" below) — against ground truth derived
from published feeds, live or replayed from an archive. It reads **GTFS-RT
VehiclePositions/TripUpdates as JSON or protobuf** (auto-detected per feed;
see "Feed formats") plus the GTFS static `.zip`, exactly as any agency serves
them. Adding an agency is just adding a YAML source config — nothing here is
agency-specific code.

## Status

Beyond the headless core (storage + predictors + ground-truth + scoring + engine +
Typer CLI — the domain models, polyline/spatial helpers, and the static + GTFS-RT
JSON feed readers now live in `stoptimes.core`, shared with the rest of the
toolkit; see `docs/MIGRATION_MAP.md`), `stoptimes val` ships:

- a **FastAPI REST + WebSocket API** (`stoptimes.val.api`) for starting/monitoring/
  streaming live runs, with a process-wide telemetry endpoint (`GET /api/telemetry`,
  `GET /api/telemetry/events`) that surfaces tick health, feed-poll health, and
  swallowed predictor/detector error counts — run it with `stoptimes val api`;
- a **Nuxt web UI** (`web/`) with four views — a live-run console, a runs list/detail
  page, a results explorer, and a telemetry dashboard;
- a **Textual TUI** (`stoptimes.val.tui`) for monitoring runs from the terminal —
  run it with `stoptimes val tui`.

## Architecture

Three pluggable seams are the design; everything else is plumbing:

- **`Predictor`** — produces arrival predictions. Built in: `gtfsrt` (replays
  whatever a feed publishes), `baseline:{constant_speed,schedule}`,
  `stoptimes:{historical_mean,polyreg_distance,polyreg_time,xgboost}` (trained
  models from `stoptimes-models`/`stoptimes-serve`, see below). Dynamic:
  `stoptimes:<artifact_id-or-scope>` (any artifact in a
  `core.contracts.artifact` registry, by exact id or by scope — e.g.
  `stoptimes:mbta` picks the best-scoped artifact per trip via
  `core.contracts.selection`) and `pipeline:<file.py>` (score your own
  `stoptimes.serve.pipeline` — see "Bring your own pipeline" below).
  `gtfs_eta:{historical_mean,polyreg_distance,polyreg_time,xgboost}` still
  work as **deprecated aliases** of the matching `stoptimes:*` id (a
  `DeprecationWarning` is emitted; existing run data/scripts keep working
  unchanged — `gtfs_eta:ewma` was retired upstream and was never re-added).
  A predictor opts into receiving vehicle position history by declaring
  `accepts_position_history = True` (see `ConstantSpeedPredictor`); the
  engine dispatches on that attribute rather than inspecting the `predict`
  signature.
- **`GroundTruthDetector`** — detects actual arrivals (`shape_distance` default,
  `position_radius`, `stop_status`).
- **`Storage`** — persists samples / arrivals / results (SQLite, Parquet export).

Predictor and detector exceptions are swallowed per-tick (one flaky predictor must not
take down a run) but are never silent: `RunSummary.predictor_errors` /
`.detector_errors` (and the matching `TickEvent` cumulative snapshots) count
exceptions per predictor-id / detector-method, surfaced through the CLI run summary,
the REST API (`RunInfo.summary`), the WebSocket stream, and `/api/telemetry`.

### Validating trained models (`stoptimes-serve[xgboost]`, the `models` extra)

`packages/val/src/stoptimes/val/predictors/gtfs_eta_models.py` wraps
`stoptimes.serve.estimator.estimate_stop_times` so any model trained with
`stoptimes-train`/`stoptimes-models` can be scored the same way as
`gtfsrt`/`baseline:constant_speed`; install it with `uv sync --extra models` (or,
inside this workspace, `uv sync --group dev`, which already pulls every package).
Note `gtfs_eta:ewma` was retired and is not in `PREDICTOR_REGISTRY` — the live
model families are `historical_mean`, `polyreg_distance`, `polyreg_time`, and
`xgboost`. The `gtfs_eta:*` predictor-id prefix is kept for backward compatibility
with existing run data and scripts even though the code now lives under
`stoptimes.*` — see `docs/MIGRATION_MAP.md`.

`GtfsEtaModelPredictor._resolve_model_registry_dir()` locates the trained-model
registry: it tries, in order, an explicit `MODEL_REGISTRY_DIR` env var, a few
legacy source-checkout / dist-info fallbacks left over from when `gtfs-eta` was a
separate sibling repository (harmless no-ops in this workspace), and the installed
`stoptimes-models` package — each candidate is verified against a real
`registry.json` before being trusted. **You normally don't need to set
`MODEL_REGISTRY_DIR` yourself** — it's just an override for the rare case none of
those fallbacks locate a real registry (see below).

```bash
uv run --no-sync stoptimes val run --source mbta \
  --predictor gtfsrt --predictor gtfs_eta:xgboost --predictor gtfs_eta:polyreg_time \
  --duration 10m --out r.parquet
uv run --no-sync stoptimes val report r.parquet
```

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

### Feed formats

`vehicle_positions_url`/`trip_updates_url` can each be **GTFS-RT JSON or
protobuf** — the default (`format: auto`) sniffs which one a response is
(Content-Type header, falling back to the body) per poll, so most sources
need no configuration at all. To pin a feed's format explicitly (skips the
sniff): add `vehicle_positions_format: pb` / `trip_updates_format: pb` (or
`json`) to the source's YAML, or pass `--vp-format pb --tu-format pb` to
`stoptimes val run` (CLI flag wins over the YAML hint). Protobuf parsing
needs the `pb` extra: `pip install 'stoptimes-core[pb]'` (already part of
this workspace's `uv sync --group dev`).

### Validating any TripUpdates source

A source doesn't need to be one of the shipped agency profiles
(`examples/agencies/`). Point `trip_updates_url` (and, for ground truth,
`vehicle_positions_url`) at *any* published GTFS-RT feed — someone else's
predictions, a feed you don't operate, a one-off test endpoint — and
`gtfsrt`/the ground-truth detectors work exactly as they do for `mbta`/`bucr`;
nothing in `val` branches on which agency it is. Either add a YAML file next
to the shipped ones (`--sources-dir` / `STOPTIMES_SOURCES` to point
elsewhere), or build a `stoptimes.core.domain.models.SourceConfig` directly
if you're driving `val` from Python. See
`packages/val/tests/test_any_tripupdates_source.py` for a worked example
against a source that was never added to `examples/agencies/`.

### Bring your own pipeline (`pipeline:<file.py>`)

`pipeline:<file.py>` scores a serving pipeline you wrote yourself, loaded via
`stoptimes.serve.pipeline.load_pipeline(path)` (T4's Phase 2 deliverable —
see `docs/notes/p2-t6.md` for the current state of that loader). Each
`predict()` call builds a `core.contracts.pipeline.PipelineContext` (this
trip, its upcoming stops, and the current vehicle position, in
`ctx.state`) and runs your pipeline; results come back either as the
pipeline's own return value or, for a pipeline whose `publish` stage hands
off to a real sink instead of returning anything, from
`ctx.state["predictions"]` if your `postprocess`/`publish` hook stashes them
there. `stoptimes.val.predictors.pipeline_adapter` is a small adapter `val`
owns, not part of the pipeline contract itself; it never imports
`stoptimes.serve` until the first prediction is actually requested, so `val`
stays importable whether or not `stoptimes-serve` is installed.

## Install

`stoptimes.val` is one member of the `stoptimes` uv workspace (`packages/val`). From
the repo root:

```bash
uv sync --group dev          # everything, for development (this workspace)
# or, installing just this part into another project:
pip install 'stoptimes-val[api]'      # + REST/WebSocket API
pip install 'stoptimes-val[models]'   # + scoring stoptimes-trained models
```

> Only if none of `GtfsEtaModelPredictor`'s registry-location fallbacks (see below)
> find a real `registry.json` do you need to set it explicitly:
> ```bash
> export MODEL_REGISTRY_DIR=/absolute/path/to/artifacts/models
> ```
> Without a resolvable registry, `gtfs_eta:*` predictors emit a `RuntimeWarning` at
> import time and every prediction comes back empty (`error: 'No trained models found
> for model_type'`) — not a crash, just silently no predictions, so watch for the
> warning.

## Usage

```bash
# List configured sources
uv run --no-sync stoptimes val sources

# Run a live validation pass (network permitting), persist + export Parquet
uv run --no-sync stoptimes val run --source mbta \
  --predictor gtfsrt --predictor baseline:constant_speed \
  --duration 10m --out r.parquet

# Report per-predictor MAE + horizon breakdown + model comparison
uv run --no-sync stoptimes val report r.parquet

# REST + WebSocket API (was the etaval-api console script)
uv run --no-sync stoptimes val api

# Terminal ops console (was the etaval-tui console script)
uv run --no-sync stoptimes val tui
```

## Replay from a recorded archive

`stoptimes val replay` scores predictions against an archived window instead
of a live feed — deterministic (same archive, window, and predictors in,
same scores out; no wall clock), which is what makes a run reproducible for
CI or for a paper's numbers:

```bash
uv run --no-sync stoptimes val replay \
  --source mbta --archive s3://transit/feeds/mbta/vehicle_positions \
  --archive-tu s3://transit/feeds/mbta/trip_updates_staging \
  --from 2026-06-01 --to 2026-06-08 \
  --gtfs mbta.zip --predictor gtfsrt --predictor baseline:constant_speed \
  --out r.parquet
```

`--archive`/`--archive-tu` accept a local dir, a single Parquet file, or an
`s3://` URI (the `s3` extra), in the exact Hive-partitioned layout
`stoptimes.record.storage.schema` (VehiclePositions) and `.tu_schema`
(TripUpdates) write — `val` reads those column names as a documented
contract, it does not import `stoptimes.record` (nothing depends on
`record` per `docs/PLAN.md` rule 1). `--archive-tu` is optional: predictors
that don't need published TripUpdates (`baseline:*`, `stoptimes:*`,
`pipeline:*`) score from `--archive` alone; only `gtfsrt` (which replays
published predictions) needs one. This is a friendlier front end over the
same replay engine `stoptimes val run --replay-vp ...` uses — both build the
same `ReplayRunDeps` (`stoptimes.val.engine.assembly.build_replay_run`).

## Walk-forward backtest and the 70/10/20 split

`stoptimes val walk-forward` (`packages/val/src/stoptimes/val/backtest.py`) splits an
archived replay window into contiguous, non-overlapping sub-windows — either an equal
COUNT (`--windows`) or a fixed WIDTH (`--width`) — and runs a fresh replay per window
with a shared `--seed`, reducing each window's ValidationResults to a per-predictor
MAE with a bootstrap confidence interval (`stoptimes.val.stats.bootstrap.bootstrap_ci`).
The output JSON carries per-window MAE+CI, window boundaries, each window's fraction of
the overall range, run provenance, the seed, and a `split_note`.

**Alignment requirement with the training train/val/test split.** The `gtfs_eta:*`
predictors wrap models trained with `stoptimes-train`/`stoptimes-models`, which
reserves a canonical chronological train/validation/test split of its training data
(`models/common/data.py`'s `temporal_split`, which by default holds out the
most-recent slice of the timeline as test — `stoptimes val` treats this as a
70/10/20-style train/val/test split and never modifies training or re-derives that
split itself). For a walk-forward run here to be a *valid, leakage-free held-out
evaluation* of any `gtfs_eta:*` predictor, every backtest window's `[window_start,
window_end]` must fall entirely inside that model's reserved **test** slice — a
window overlapping the train/val slices would let the model be evaluated on data it
was directly or indirectly fitted on, inflating apparent accuracy. Predictors with
no training split (`gtfsrt`, `baseline:*`) have no such constraint.

It is the **caller's** responsibility to choose `--replay-start`/
`--replay-end` (or the `start`/`end` passed to `walk_forward()`) so they lie
within the trained model's reserved test window — `walk_forward()` records
this requirement in the result's `split_note` field, it does not verify it
(`stoptimes val` has no visibility into which archived rows the training
`temporal_split` put in train vs. val vs. test at training time).

```bash
uv run --no-sync stoptimes val walk-forward wf.json \
  --source mbta --replay-vp s3://bucket/mbta/vp --replay-gtfs mbta.zip \
  --replay-start 2024-06-01T00:00:00Z --replay-end 2024-06-08T00:00:00Z \
  --windows 7 --predictor gtfsrt --predictor baseline:constant_speed --seed 0
```

## Publication figures

`stoptimes val figures <parquet> --out-dir <dir>`
(`packages/val/src/stoptimes/val/figures.py`) renders six 300 dpi PNG figures from a
Parquet results file using a colorblind-safe Okabe-Ito-derived palette (headless
`Agg` backend, no seaborn): `error_vs_horizon`, `error_distribution`,
`convergence_curve`, `per_route_mae`, `calibration_diagram`, and
`cost_accuracy_pareto`. The Pareto figure (inference latency vs. MAE) is only
produced when `--summary <...>.summary.json` is given — without it, a run has no
recorded per-predictor latency to plot, so it is skipped (returns `None`) rather
than drawn empty.

```bash
uv run --no-sync stoptimes val figures r.parquet --out-dir figs/ --summary r.parquet.summary.json
```

## Configured sources

- **`mbta`** — MBTA public GTFS-RT JSON + GTFS static.
- **`bucr`** — databus dev stack (base URL configurable, default `http://localhost:8000`).

## Development

From the repo root (this workspace):

```bash
uv sync --group dev
uv run --no-sync python -m pytest packages/val -q -p no:cacheprovider
uv run --no-sync ruff check packages/val
```

## License

MIT
