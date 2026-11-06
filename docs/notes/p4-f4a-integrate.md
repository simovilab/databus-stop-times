# p4-f4a-integrate notes (for F4b: serving, val, test-day gate)

## What exists now

| Piece | Where |
|---|---|
| Feature assembly (serving path, one `t`) | `core.features.live.assemble_live_features(ctx, assets, state, t, features=LIVE_FEATURES)` |
| Feature assembly (training path, many `t`, vectorised, bit-identical) | `core.features.live.assemble_live_batch(ctx, assets, state, times)` |
| Trip context from static | `core.features.live.build_trip_context(static, trip_id, route_id, service_date, tz, holiday_region=...)` -> `(TripContext, ForwardMatcher)` |
| Registry entries with input tags | `core.features.registry`: `Feature.inputs / .assets`, `missing_inputs(...)`, tag `live` |
| Artifact assets | `ArtifactMetadata.assets / .timezone / .holiday_region` (schema 2, v1 loads); `write_asset`, `read_asset_bytes` |
| Asset loading and refusal | `serve.model_loading.load_artifact_assets`, `check_artifact_assets`, `ArtifactAssetError`; `check_feature_producibility(..., available_inputs=)` |
| Training frame | `train.live_frame` (re-exported from `train.frame`): `prepare_trip_records`, `build_live_dataset`, `build_day_frame`, `write_live_assets` |
| CLI | `stoptimes data frame --live --train-days A:B --val-days C:D [--rows-per-day N]`; `stoptimes train run` detects `<frame>.assets/` and trains the live artifact |
| Experiment | `examples/experiments/xgb-live.yaml` |

## How F4b calls the assembly per tick

```
ctx, matcher = build_trip_context(loader, trip_id, route_id, service_date, tz, holiday_region=...)   # cache per (trip, service date)
state = update_trip_state(store.get(ctx.trip_key), Ping(epoch, lat, lon), polyline=matcher,
                          stop_progress_m=ctx.stop_progress_m, trip_key=ctx.trip_key)
store.put(state)
rows = assemble_live_features(ctx, assets, state, t, features=model.required_features)   # one row per downstream stop
y = model.predict(rows[model.required_features])          # seconds from t to arrival
```
`assets = load_artifact_assets(meta, registry)`; `tz` and `holiday_region` come from `meta.timezone / meta.holiday_region`
(not from the serving source config) so the temporal features and `minute_of_day` are what the model was trained with.
`rows` also carries `trip_id, route_id, service_date, stop_id, stop_idx, t, ping_ts`.

Definitions fixed here (all tested): downstream stops are `stop_progress > progress` (`searchsorted(..., 'right')`),
`stops_ahead = j - j0` (0 = next stop), the vehicle state is the latest *valid* forward-matched ping with epoch <= `t`
(`ping_ts` is that ping), `delay_last` only uses arrivals whose `detect_epoch <= ping_ts`.

## For F4b to do (not done here)

1. `serve/pipeline.py`: define `SERVING_INPUTS = frozenset({"schedule", "trip_state", "identity", "clock"})` once the
   pipeline really supplies them. `check_feature_producibility` reads that name with `getattr(..., frozenset())`, so until
   then a live artifact is refused with "needs static schedule; needs per-trip state ..." (verified), and
   `STOPTIMES_ALLOW_UNPRODUCIBLE_FEATURES` / `allow_unproducible_features=True` is the explicit bypass for a custom predictor.
   `val`'s artifact predictor can pass `available_inputs=` directly.
2. `load_artifact_model` now also calls `check_artifact_assets` (hash of every declared asset, and that every asset a feature
   reads is declared). This is not overridable by design.
3. The model is a pickled `xgboost.XGBRegressor` (`ModelFormat.PICKLE`, so the existing drift check covers xgboost and sklearn).
   Pass a DataFrame whose columns are `required_features` in that order (xgboost checks names and order).
   `required_features` is set on the pickled object and mirrored in `artifact.json`.
4. Staleness: training instants require a valid ping at most 30 s old (`LiveFrameParams.max_ping_age_s`). Serving should apply
   the same guard (or the model extrapolates `sched_rem_t - sched_rem`).
5. Ghost trips: `prepare_trip_records` does not know the calendar. The study dropped instances whose `trip_id` was not active
   that day (about 17 % of July instances, 0.1 % in the fall); my real-data driver applied the same instance list from
   `gt/arrivals_by_day`. A package-side fix is to read `calendar.txt`/`calendar_dates.txt` in `GtfsStaticLoader` (not owned here).

## Deviations from the study, and why

* `route_i` (int route id) -> `route_code` (index in the artifact's sorted route vocabulary; unseen -> NaN). Agency-agnostic.
* Distance and `stops_ahead` come from F3's forward-only matcher (the study used the package's global nearest projection and
  `searchsorted(..., 'left')`). Differences vs study frames on joined rows: `distance_to_stop` identical for 74-77 % of rows,
  p95 |diff| 22-31 m; `stops_ahead` identical for 89-92 %, p95 diff 1.
* `prior_rem`: identical to the study for ~80 % of rows, p95 |diff| about 1.1-1.4 s (my out-of-fold fold assignment is
  `day index mod 6` over the 83 training days in date order; the study's fold map differs).
* Prediction instants are the study's 60 s epoch grid with a valid ping at most 30 s old, *not* every ping: with per-ping
  instants `sched_rem_t - sched_rem` would equal `-delay_now` exactly (age 0) and the model would learn a relation serving
  ticks (age 0-30 s) never satisfy. `delay_now, sched_rem, sched_rem_t, delay_last` and the target match the study's rows
  exactly on joined (trip, t, stop) keys (|diff| = 0 for all; same NaN rate of `delay_last`).
* Training uses the `xgboost` sklearn API with early stopping on the validation days (study: `xgb.train` +
  QuantileDMatrix, same objective and params). Rounds cap 4000 (study's final fit ended at 3998).
* The generic `ETADataset.clean_data()` filters (distance >= 10 m, target <= 2 h) are NOT applied on the live path;
  rows only need `target > 0`.

## Gaps and things worth knowing

* `train run` cannot express calendar splits in `Experiment.split` (contract is not mine). On a live frame the frame's own
  `split` column is the split and the experiment's `split:` block is ignored (documented in `xgb-live.yaml`).
  `test_*` metrics in the live artifact are the *validation* days (kept for `models list` ranking; also `val_mae_seconds`).
* `data frame --live` uses one static feed for all days. A multi-feed archive (the study's per-day feed versions) is driven
  through the Python API (`prepare_trip_records` per day + `build_day_frame`), which is what the real run did.
* `GtfsStaticLoader.trip_geometry/trip_schedule` filter the whole `stop_times` frame per call (O(rows)); fine for filtered
  feeds, slow for a full national feed. Prepare-time is ~30 s/day on the 3-route data.
* pandas 3 returns `datetime64[s]` from some readers: `ts.astype("int64")` is then seconds, not ns. `prepare_trip_records`
  coerces via `coerce_utc_ns` first.
* Test days (2026-10-21..11-03) were never read, built or scored. The study's research frames under `feat/pool83` contain
  test-day files; only train/val days were opened (the comparison script refuses test days).
