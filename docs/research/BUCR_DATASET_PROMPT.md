# TASK: Build a BUCR ETA training dataset from raw navsat traces, then train the models

> Implementation prompt for building BUCR (navsat AVL) ETA models by reusing this repo's
> MBTA pipeline. Hand to an implementing agent or execute directly. Grounded in the exact
> modules and S3 paths so nothing gets re-derived. Drafted 2026-08-24.

## Ground truth — do NOT re-derive
- This repo's ETA pipeline already works for MBTA, whose GTFS-RT hands over
  `trip_id`/`route_id`/`stop_id`/`current_stop_sequence` per position. The training
  target (`time_to_arrival_seconds`) is computed purely from observed positions vs.
  stop coordinates (first VP within 50 m = arrival) — the offset schedule never
  supplies truth, so a wrong timetable does NOT poison labels.
- BUCR (navsat AVL) has none of those structural columns. Raw columns in
  `s3://transit/feeds/bucr/navsat/...`: `plate_number` (6 vehicles), `cr_datetime`
  (local CR string), `cr_datetime_utc`, `ingested_at_utc`, `lat`, `lon`, `speed_kmh`,
  `odometer_km` (cumulative), `estado` ∈ {movimiento, detenido}, `lugar`. Natural key
  `(plate_number, cr_datetime)`. ~44 compacted rehearsal days available.
- BUCR static GTFS = 2 routes, 22 stops, ~896 shape points. Source it from the weekly
  S3 snapshot `s3://transit/feeds/bucr/gtfs_static/<ISO date>.zip`. If that prefix is
  empty, set `BUCR_GTFS_STATIC_URL=https://feeds.simovi.org/bucr/schedule/gtfs.zip`,
  run the `snapshot_static_gtfs` task once, then read the snapshot. Do NOT use the toy
  `gtfs/fixtures/example.json`.
- Map-matching already exists and is installed in `.venv` — REUSE, do not reimplement:
  `from etaval.spatial.polyline import build_polyline, project_point_to_polyline,
  assign_stops_monotonic, upcoming_stops`.
- Agency temporal config already present: `core.config.AGENCY_TEMPORAL_DEFAULTS["bucr"]`
  = America/Costa_Rica / region CR. Extract temporal features with `agency="bucr"`.
- Formulation: **segment-based** (roadmap Phase 3.1) — one row per (trip instance ×
  stop-to-stop segment), target = observed traversal seconds. Stop-level ETA is derived
  by summing predicted segments.

## Goal
A BUCR segment training dataset + trained model families that beat the offset published
schedule on held-out days by a defensible margin. Reasonable, not perfect.

## Steps

1. **navsat → canonical VP adapter** (`feature_engineering/navsat_adapter.py`).
   Pure, deterministic, unit-tested field mapping. Emit the canonical VP frame *minus*
   the structural columns: `vehicle_id`=plate_number, `ts`=cr_datetime_utc (UTC),
   `lat`, `lon`, `speed`=speed_kmh/3.6 (m/s), `current_status` (detenido→STOPPED_AT,
   movimiento→IN_TRANSIT_TO); retain `odometer_km`.

2. **Trace cleaning** (roadmap 2.5). Drop stale device fixes where
   `|cr_datetime_utc − ingested_at_utc|` exceeds a chosen, recorded threshold; drop
   null/zero coords. Report the drop-rate. No smoothing yet.

3. **Trip / route / stop inference** (`feature_engineering/bucr_trip_inference.py`) —
   the core work. Per (plate × service-day):
   - a. Build a polyline per candidate route+direction from the static shapes
     (`build_polyline`).
   - b. Project the trace onto each candidate (`project_point_to_polyline`); score by
     mean cross-track error + monotonic-progress fraction; assign the best
     route+direction. Park ambiguous traces.
   - c. Segment into discrete **trip instances** where along-shape progress resets
     (end-of-shape → layover → restart) or a long time/space gap occurs.
   - d. Within each instance assign `current_stop_sequence` + `stop_id` via
     `assign_stops_monotonic` (loop-back-safe). Synthesize a stable `trip_id` =
     f(plate, route, direction, instance-start).
   - e. Reject points whose cross-track error exceeds a recorded threshold — the
     "bus went somewhere else" anomaly.
   Output: the canonical 12-column frame `rt_pipeline.storage.write_vehicle_positions`
   expects, now with `route_id`/`trip_id`/`stop_id`/`current_stop_sequence` filled.

4. **Trip-inference quality report** (roadmap 2.4) — non-negotiable; there are no
   ground-truth trip labels. Emit per-route match rate, ambiguous count, dropped-trace
   count, trips/day, and a spot-check table of N inferred trips vs. the timetable.
   Hand-label one day as a validation set if match rate is in doubt.

5. **Segment dataset builder** (`feature_engineering/segment_dataset_builder.py`,
   roadmap 3.1). One row per (trip instance × traversed segment); target = observed
   seconds between arrival at stop k and stop k+1 (reuse the 50 m arrival detection in
   `dataset_builder.py`). Features: segment length along shape, entry kinematics
   (speed, is_at_stop), temporal features (`agency="bucr"`), stops-remaining, plus the
   shape/cyclical features the existing schema uses — must satisfy the model feature
   contract in `models/common/data.py::ETADataset.FEATURE_GROUPS`. Also emit the
   stop-level dataset from the *same* inferred frame via
   `build_vp_training_dataset(agency="bucr")` as a cheap ablation.

6. **Train + evaluate.** Run `models/train_all_models.py` on the segment dataset
   (historical_mean, ewma, polyreg_distance, polyreg_time, xgb). Derive stop-level ETA
   by summing predicted segment times. Baseline = the offset published schedule. Report
   MAE/RMSE per horizon, segment- and stop-level, using the walk-forward backtester
   `models/evaluation/roll_validate.py`. Calendar-boundary splits grouped by `trip_id`
   — no `trip_id` in two splits.

## Constraints
- Reuse `etaval.spatial.polyline`, the existing arrival-detection / temporal / model
  code. New code = small, single-responsibility modules only.
- Deterministic and recorded: all thresholds, seed, git SHA, dataset hash, exact
  feature list, split boundaries — this is a citable artifact.
- Run on the ~44 compacted rehearsal days first; the pipeline must re-run unchanged on
  the 90-day corpus later.

## Acceptance
- A BUCR segment dataset with documented schema + a defensible trip-inference match rate.
- All model families train; summed-segment stop-level ETA beats the offset-schedule
  baseline on held-out days.
- Leakage probe passes (shuffled labels → metrics collapse to baseline); no `trip_id`
  spans two splits; fixed-seed re-run is byte-identical.
