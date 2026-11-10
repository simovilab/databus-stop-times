# databus integration: TripUpdate predictions from stoptimes

Status: **design, direction agreed (O3)**. Refreshed 2026-11-09 against databus `main` @
`1be2da1` and `feat/ops-dashboard` @ `39a479d`, and this repo @ `a988f73`. Audience: whoever
wires databus to serve the bUCR live model. The cutover steps are PLAN.md §8 **G2**.

The databus-side items in §4 are written so they can be pasted into a databus issue
unchanged.

## 1. Decision, and how we got here

The model now runs in a **separate prediction-service container (O3)**:

- The service loads the model from MLflow (`ml.simovi.org`, `stoptimes-bucr@production`,
  shadowed by `@staging`).
- It reads the same VehiclePositions / trip state that databus already has.
- It writes stop-time predictions that databus folds into its TripUpdates.

**databus never talks to MLflow and never imports ML code.**

History, kept short (the full option table is in git history of this file, before
2026-11-09):

| Option | What it was | Fate |
|---|---|---|
| O1 | swap databus's `gtfs_eta` import for `stoptimes.serve.estimator`, in-process | **dead for the live model.** The live model needs per-trip state and 19 features; `estimate_stop_times` is per-ping and stateless. It would also need stoptimes and xgboost on databus's Python 3.14 |
| O2 | stoptimes pipeline running inside `realtime-engine` | superseded 2026-09-19 (all of O1's costs) |
| **O3** | prediction-service container next to databus | **chosen** |
| O4 | databus calls MLflow / imports mlflow itself | **rejected.** It would make databus rebuild the 19 live features and per-trip state (the training/serving duplication again), and the model would be invisible to `val`, drift checks and selection |

MLflow is the model source of O3, not a separate path. The external server means databus's
compose needs no `mlflow` service ([MLFLOW_INTEGRATION.md](MLFLOW_INTEGRATION.md) §0.3).

## 2. What is true today

### databus (verified in source)

The ETA producer **is on `main`** now. This file previously said otherwise.

```
NavSat HTTP ─(fetch_positions, beat 10 s)─▶ MQTT transit/vehicle/<id>/position ◀─ simulator / OBE
        realtime-engine: process_position_update(run_id, vehicle_id)      realtime_engine/tasks.py:255
          1 produce_stop_status        → run:<id>:vehicle_stop_status (STOPPED_AT ≤20 m / INCOMING_AT ≤50 m)
          2 detect_from_telemetry      → RUN_COMPLETED at the terminal
          3 produce_stop_times         → SET run:<id>:stop_time_updates EX 60       tasks.py:293 (main), :386 (ops-dashboard)
          4 position-leaf detection    → RunStarted etc.
scheduler (beat 15 s) ─▶ schedule-engine: build_vehicle_positions / build_trip_updates
          builders.build_trip_update_entity: for run in runs:in_progress → GET run:<id>:stop_time_updates
          → backend/feed/files/{vehicle_positions,trip_updates}.{json,pb}
orchestrator: /feed/realtime/{vehicle_positions,trip_updates}.{json,pb}, /feed/schedule/feed.zip
```

**The producer** is `runs/domain/progression/stop_times.py`:

- It lazily imports `gtfs_eta.eta_service.estimator.estimate_stop_times` with
  `ETA_MAX_STOPS=3`.
- Uncertainty is a fixed `ETA_DEFAULT_UNCERTAINTY_S=120`, and `STOP_TIME_UPDATES_TTL_S=60`.
- The model is whatever `MODEL_REGISTRY_DIR` holds. In the repo that is
  `backend/eta_models/`: a **synthetic** `polyreg_distance_global_baseline_v0`, fit on
  constant-speed samples, not real telemetry.
- So today's bUCR TripUpdates are a constant-speed guess for the next 3 stops.

**The Redis contract** is `runs/domain/telemetry/stop_time_updates.py` (stdlib-only).
`run:<run_id>:stop_time_updates` is a JSON **string**: a list of entries with exactly
`stop_sequence`, `stop_id`, `arrival_time`, `departure_time` (POSIX s) and `uncertainty`
(s). `from_redis` drops malformed entries. The builder sorts and dedups by
`stop_sequence`, and a missing or empty key gives a TripUpdate with no `stop_time_update`.

The builder also guarantees:

1. It iterates `runs:in_progress` only.
2. It never runs a model.
3. A missing projection leads to an honest empty list.

**Keys O3 needs**, from `runs/domain/telemetry/keys.py` plus hard-coded ones:

- `runs:in_progress` (set)
- `run:<id>` (hash: `vehicle`, trip fields)
- `run:<id>:trip`
- `vehicle:<id>:position` (hash with `timestamp`, epoch s)
- `run:<id>:vehicle_stop_status`
- `trip:<trip_id>:current_run`, `vehicle:<id>:current_run`

**Feed shape.**

- The JSON is the builder dict (snake_case, `json.dumps`), and the `.pb` is the same dict
  through `ParseDict`.
- The VP `trip` has `trip_id`, `route_id`, `direction_id` and `schedule_relationship`.
  `start_date`/`start_time` exist only on `feat/ops-dashboard` (`32c0158`, not on `main`).
- `vehicle.timestamp` is the ping time.

**Other branch differences.** `feat/ops-dashboard` adds an `ops` app (`/api/ops/`) that
also reads `stop_time_updates_key`, feed validation in `schedule_engine/validation.py`, and
the shared-HTTP-source dedupe (`39a479d`).

### stoptimes (verified in source)

- **Live inference:** `stoptimes.serve.live_session.LiveSession`. It is run-scoped: it
  holds a `TripStateStore`, is fed pings in arrival order, and skips trips whose newest
  ping is more than 30 s older than the feed clock. The artifact's sidecar assets carry the
  time zone and holiday region.
- **`stoptimes serve`** already builds a `LiveSession` for live artifacts
  (`packages/cli/src/stoptimes/cli/commands/serve.py:18`).
  - Input: the source's VehiclePositions URL.
  - Output: GTFS-RT TripUpdates to `--sink file:|http(s)://|mqtt://` in `--format pb|json`,
    with `--registry mlflow+<uri>`.
  - It has **no** Redis/databus sink yet.
- **`val`** scores the same `LiveSession` path (`mlflow:<model>@<alias>`), so a `val` result
  is a direct preview of O3's quality.
  - See [guides/bucr-live-test.md](guides/bucr-live-test.md).
  - The `serving:` block in `examples/agencies/bucr.yaml` pins the August **segment**
    models. It is used only by the legacy `estimate_stop_times` path (`stoptimes:*` /
    `gtfs_eta:*` predictors, old O1).
  - It has no effect on the live model or on O3.

## 3. O3, concretely

```
databus Redis (state) ──read──┐                       ┌── or: GET /feed/realtime/vehicle_positions.pb
                              ▼                       ▼
            prediction-engine  (stoptimes, own image, own Python)
              model: MLflow stoptimes-bucr@production (resolved at start, re-checked every few min)
              static: the GTFS zip databus publishes (/feed/schedule/feed.zip), reloaded on change
              loop every N s: pings → LiveSession → predictions per (run, stop)
                              │ SET run:<id>:stop_time_updates[:shadow] EX 60  (+ :meta, heartbeat)
                              ▼
            schedule-engine builders.build_trip_update_entity  (unchanged) → trip_updates.pb
```

### Interfaces

- **Input.** Either input works; pick one (open question 1).
  - **(a) databus's own published VehiclePositions** (`/feed/realtime/vehicle_positions.pb`,
    15 s). This is zero coupling to the Redis schema, and it is exactly what `val` consumes.
    It costs up to 15 s of extra latency, and the service sees only In Progress runs.
  - **(b) Redis directly:** `runs:in_progress` → `run:<id>` (vehicle, trip) →
    `vehicle:<id>:position`. This gives per-ping freshness, but the service then has to
    track `keys.py`.
  - Either way, a ping is `(trip_id, start_date, vehicle_id, epoch, lat, lon)`, plus
    optional `vehicle_stop_status` for loop safety.
- **Run id.** The service writes per run. From a VP entity, take `vehicle.id`, then
  `vehicle:<id>:current_run`, or use `trip:<trip_id>:current_run`.
- **Output.** The existing five-field list, unchanged, so `schedule-engine` needs no code
  change:

  ```json
  [{"stop_sequence": 8, "stop_id": "bUCR_0_11", "arrival_time": 1790272777,
    "departure_time": 1790272777, "uncertainty": 60}]
  ```

  - One entry per remaining stop the model predicts. That is more than today's 3, so
    decide the cap (open question 3).
  - `stop_sequence` must be the exact `stop_times.txt` value.
  - `uncertainty` can come from horizon-bucket MAE until quantile models exist.
- **Freshness and audit.** Add a sibling key `run:<id>:stop_time_updates:meta` =
  `{source_ts, computed_at, model: "stoptimes-bucr/<version>"}`. Keep it out of the
  five-field list.
- **Health.** `prediction-engine:heartbeat` (EX ≈ 3× interval) with run, error and
  skipped counts.
- **Where it plugs in.** The consumer is `schedule_engine/builders.py::build_trip_update_entity`
  (the `stop_time_updates.from_redis(raw)` read). It is untouched. The only databus code
  change is removing step 3 (`produce_stop_times`) from `process_position_update` at
  cutover, because there must never be two writers to one key.

### Rollout (each step needs the user's go)

| Step | What | Rollback |
|---|---|---|
| G2b shadow | service writes `run:<id>:stop_time_updates:shadow`; production still comes from `produce_stop_times`. Compare with `val` live, and by replaying a recorded day | stop the container |
| G2c cutover | service writes the real key; the same databus deploy removes step 3 | stop the service and revert the deploy; keys expire in 60 s |

Old step G2a (O1) is dropped. G4 (artifact library versions) moves inside the service
image, because databus's versions no longer matter.

## 4. What databus needs to change

- [ ] A `prediction-engine` compose service (dev and prod):
  - on the network with `state`;
  - environment `REDIS_HOST=state`, `REDIS_PASSWORD`, `STOPTIMES_MLFLOW`, a **read-only**
    MLflow credential, and the key suffix (`:shadow` at G2b, empty at G2c);
  - outbound HTTPS to `ml.simovi.org`;
  - no host ports.
- [ ] Ops dashboard (`feat/ops-dashboard`): show the heartbeat and `:meta` age.
- [ ] At G2c: remove the `produce_stop_times` call (`realtime_engine/tasks.py:293` on
  `main`) in the same deploy. Delete `stop_times.py`, `backend/eta_models/` and the
  `gtfs-eta` clone in `docker-entrypoint.sh` once G2c has been stable.
- [ ] ARCHITECTURE.md: add a "Prediction engine" role.
  - It reads the state snapshot on an interval.
  - It writes only the derived `run:<id>:stop_time_updates` (+ `:meta`) with a TTL, plus a
    heartbeat.
  - It does not modify operational state, build feeds or serve APIs.
  - The State section's rule becomes: operational state is written by `realtime-engine` and
    the lifecycle actions; derived prediction projections are written by `prediction-engine`.
- [ ] Tests: the builder is unchanged. Optionally add a test that skips predictions whose
  `:meta.source_ts` is too old.

## 5. What stoptimes needs to build

- [ ] A databus sink/ingest for `stoptimes serve`:
  - a `--pipeline` file that does input (a) or (b) above;
  - publishing per run to Redis in the five-field format, with `:meta` and heartbeat;
  - keeping the last good value when a run has no predictions;
  - no Django imports.
  - The pipeline file owns databus's key names. Keep it in sync with `keys.py`, or vendor
    `keys.py` and `stop_time_updates.py` (stdlib-only).
- [ ] A `serve` Docker image, with MLflow alias re-check and static-GTFS reload.
- [ ] Loop safety: bUCR stops on loop sections. Feed `vehicle_stop_status` /
  `current_stop_sequence` into matching so passed stops are never predicted.

## 6. What tomorrow's live test does and doesn't tell us

The test is [guides/bucr-live-test.md](guides/bucr-live-test.md): real NavSat pings, a
local databus, and `val` with `mlflow:stoptimes-bucr@staging`.

**It tells us:**

- the model's accuracy on unseen days, with real GPS, scored the way O3 would serve it
  (the same `LiveSession`);
- how it compares with the timetable and with what databus publishes today (`gtfsrt`);
- whether databus's VehiclePositions carry everything the live features need: `trip_id`,
  ping timestamps fresh within 30 s, and `start_date`.

**It doesn't tell us:**

- **O3's own latency and freshness.** `val` polls the 15 s feed; O3 would also add its
  cycle and Redis round trip.
- **How databus *serves* predictions.** Nothing is written to Redis, and `trip_updates.pb`
  still carries the synthetic model.
- **Behaviour under operator-matched trips at scale.** Tomorrow's runs are started by hand
  with hand-chosen `trip_id`s. In production that matching is the operator's (or the
  simulator's), and mismatches will be more frequent.
- **Robustness to databus edge cases:**
  - runs completed early at the terminal: ≤ 20 m `STOPPED_AT` can drop the terminal arrival
    from scoring;
  - No Signal / Cancelled runs;
  - restarts.
- **Long-term drift.** The calendar ends 2026-11-27, and the next term needs a new feed and
  likely a retrain.

## 7. Open questions

1. **Input (a) or (b)?** (a) databus's published VP feed (simplest, up to 15 s staler) or
   (b) Redis directly (fresher, coupled to `keys.py`).
2. **Cycle interval.** 5 s was the proposal. The bUCR NavSat cadence and the 10 s
   `fetch_positions` beat set the useful floor.
3. **How many stops to publish.** Today it is 3. The model predicts all remaining stops,
   but is weak beyond 20 min. Options: cap by horizon (e.g. 20 min), or publish everything
   with a larger `uncertainty`.
4. **Empty TripUpdates.** GTFS-RT expects at least one `stop_time_update` unless the trip
   is canceled. What does infobus do with an empty one? (A pre-existing issue.)
5. **Where the pipeline file lives** (databus `backend/prediction/`, or stoptimes), and
   whether the two helper modules are vendored.
6. **Credentials.** Who owns the read-only MLflow credential for the VPS, and how is it
   delivered (env file vs. Docker secret)?
