# p4-f4b-serve notes

## What changed

* `serve/live_session.py` (new, small; not in the track's path list but the shared piece of serve and val): `LiveSession`
  = run-scoped `TripStateStore` + `TripContext`/`ForwardMatcher` per (trip, service date) + artifact assets/tz/region.
  Used by `serve.pipeline` and by `StoptimesArtifactPredictor`, so both compute features one way.
* `serve/pipeline.py`: `SERVING_INPUTS`; `default_ingest` polls VehiclePositions when `config["live_session"]` is set;
  `default_match`/`default_features` have a live branch (state update, 30 s staleness guard, `assemble_live_features`);
  `default_predict` makes one batched call for live rows (falls back to per-row on failure). Non-live paths unchanged.
* `cli serve`: builds the session once (outlives every tick); refuses a live artifact with no static feed.
* `val/predictors/artifact.py`: live artifacts are scored by a per-run session (`accepts_position_history`,
  `attach_static_loader`, optional `now=`).
* Tests: serve +24 (127+2s), cli +2 (101), val +10 (495+1s). Existing test changed:
  `test_model_loading::test_live_features_are_refused_by_default...` now asserts the opposite (by design).

## Rules (definitions, not tuning)

* Staleness: skip a vehicle whose newest valid ping is > 30 s older than t (training bound).
* Service date: VP `trip.start_date` if valid YYYYMMDD; else the agency-local date in {ping day-1, day, day+1} whose
  anchored schedule window is closest to the ping (no day boundary). On the 14 test days it agreed with the study's
  service date for every instance (`gate/diag`, mismatch 0).
* val predictor clock: t = newest ping epoch seen across all trips (the engine does not pass the tick time to
  `predict`); `made_at` = t. In Gate A the study's grid instant is passed as `now=`.

## Problems found (not fixed; not mine)

* `val` replay is slow: ~0.8x real time for a 5-minute window, all of it needing >18 h per service day. Of the 230 s,
  ~175 s is the live predictor (38 ms/call: feature assembly + one xgboost call per trip per tick; the engine calls
  every active trip every tick). The engine's O(history) re-projection per tick adds to it. Gate B therefore uses a
  25-minute window per test day.
* `val` TU replay reader: `arrival_time = NaT` becomes int64-min epoch (garbage `predicted_arrival_epoch` of -9.2e9,
  |error| 1e10 s) for the `gtfsrt` predictor, which wrecks any mean; the Gate B report drops those rows.
* The study's `trip_updates_staging` archive mixes INTEGER and VARCHAR `*_schedule_relationship` files, so
  `--archive-tu` on the raw day directory fails ("Failed to parse 'CANCELED' ... int32"). Gate B reads a normalised
  copy (`gate/normalize_tu.py`).
* `GtfsStaticLoader.trip_geometry/trip_schedule` filter the whole stop_times per trip (F4a already noted).
* `core.features` APIs untouched. `StoptimesArtifactPredictor` for non-live artifacts is unchanged.

## Gate results (test days 2026-10-21..11-03, scored once, no tuning)

Outputs: `/private/tmp/claude-501/-Users-dotj-Desktop-SIMOVI-git-no-sync-databus-stop-times/8e06a208-5ebe-48f9-a314-c094112577b1/scratchpad/f4/gate/` (`gateA/SUMMARY.md`, `gateB_report.txt`, scripts `gate_a_predict.py`, `gate_b_run.sh`, `gate_b_report.py`).

**Gate A** (package val predictor path, ping stream fed in order, study scorer + paired set, 2,859,642 rows, 4,562 trips,
1000 cluster bootstraps). xgb-live 95.6 s [93.0, 98.1] vs study servable-plus 95.5 s [92.9, 98.0] (same rows) vs MBTA 118.1 s
[114.8, 121.3]. By horizon (min): [0,2) 22.4 / 28.7; [2,5) 41.0 / 48.8; [5,10) 64.2 / 76.0; [10,20) 101.8 / 123.5;
[20,30) 156.7 / 198.1; 30+ 263.1 / 337.6 (ours / MBTA). Win rate vs MBTA 61.1 %. Gate target (within ~2 s) met: +0.1 s.
Predictions: 12 k instants/day, ~9-12 min/day single process, 5-6 days in parallel.

**Gate B** (`stoptimes val replay`, `gtfsrt` vs `stoptimes:xgb-live`, window 20:00-20:25Z of each of the 14 days, NOT whole days;
paired on trip-stops that both predictors have). 3,452 trip-stops, 290 trips. MAE: gtfsrt 80.9 s [70.5, 93.5]; xgb-live 67.8 s
[62.4, 73.2] (trip-cluster bootstrap). By horizon: [0,2) 28.6 vs 31.8; [2,5) 56.6 vs 48.6; [5,10) 87.9 vs 69.6; [10,20) 133.9 vs 102.7;
[20,30) 182.9 vs 140.8. xgb-live better on 11 of 14 days. Caveats: the replay scores every tick (xgb-live 1.76 M results vs 245 k
for the change-only TU archive), so the two predictors are paired by trip-stop, not by made-at instant; the replay scorer's own
ground truth (shape-distance first approach) differs from the study's arr_interp (study notes: ~4.8 s earlier); the window is a
PM peak, so absolute numbers are not comparable with Gate A. At [0,2) min xgb-live is worse than gtfsrt here, unlike Gate A
(22.4 vs 28.7); likely the ground-truth/timing difference plus age-0 pings (below), not investigated further.
Wall time per window (6 parallel procs, 25 min of feed): 20-58 min (`walltimes.txt`), i.e. about 1x real time or slower; 11-01..03 re-run
after a path fix (the archive uses unpadded month=11/day=1).

**Age-0 note.** In replay t = newest ping (ping age ~0-6 s); training used grid instants with age 0-30 s (p50 3 s). Not tuned.

