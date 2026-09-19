# databus integration: TripUpdate predictions from stoptimes

Status: **design, agreed direction** · written 2026-09-19 · cutover steps are PLAN.md §8 **G2** ·
audience: databus maintainers and whoever runs the G2 cutover.

This document lists the ways databus can get its TripUpdate arrival predictions from stoptimes, what each
costs, and exactly what databus has to change for the chosen one. It lives in the stoptimes repo because
that is where the plan lives. The databus-side items (§5) are written so they can be copied into a databus
issue or ADR unchanged.

databus facts below were read from databus commit `15dcfe1` on `feat/eta-xgboost-model` /
`feat/fetch-telemetry`. **The ETA code is not on databus `main`** (nor on `feat/ops-dashboard`), so
re-check paths against whatever branch is deployed before G2a.

---

## 1. How databus predicts today

```
Vehicle ─MQTT─▶ realtime-engine (Celery task per position, ~1–5 s)
                  realtime_engine/tasks.py:293 → produce_stop_times(run_id, vehicle_id)
                    reads Redis: run:<id>, vehicle:<id>:position, stop status; shape geometry via ORM (cached)
                    projects the vehicle onto the shape → shape_distance_to_stop per upcoming stop
                    gtfs_eta.eta_service.estimator.estimate_stop_times(vehicle, upcoming_stops, route_id, trip_id,
                                                                       prefer_route_model=True, max_stops=3, shape)
                    SET run:<id>:stop_time_updates  (JSON list, EX 60)
scheduler (15 s) ─▶ schedule-engine
                  builders.py: for run_id in runs:in_progress → GET run:<id>:stop_time_updates → trip_updates.pb
```

**The Redis contract.** `run:<id>:stop_time_updates` holds a JSON list. Each entry has exactly five fields,
`stop_sequence`, `stop_id`, `arrival_time`, `departure_time` (POSIX seconds) and `uncertainty` (s). It is
encoded and decoded by `runs/domain/telemetry/stop_time_updates.py` (`to_redis` / `from_redis`). Today
`arrival_time == departure_time` and `uncertainty` is always `ETA_DEFAULT_UNCERTAINTY_S` (120).

**Behaviour any option must preserve:**

1. **No predictions → no write.** `produce_stop_times` leaves the last good value in place.
2. **60 s TTL.** It is the only thing that removes the key. Nothing deletes it when a run ends.
3. **The builder iterates `runs:in_progress` only**, so a key left over from an ended run never reaches the feed.
4. **A missing or empty key** produces a TripUpdate entity with no `stop_time_update` entries (the
   builder's "honest skip").
5. **`schedule-engine` never runs a model.** It only reads the projection (ARCHITECTURE.md: the publisher
   role doesn't interpret telemetry).

Also relevant: the model call currently runs **inside the telemetry task**, so model latency or a model
failing to load delays position ingestion. databus runs **Python 3.14**. stoptimes development is pinned to
**3.12** because of torch (LSTM only).

## 2. Options at a glance

| | Where the model runs | Changes to databus | In databus's venv | Telemetry task waits on the model | New model without a databus deploy | Extra infrastructure |
|---|---|---|---|---|---|---|
| **O1 Library, in-process** (G2a) | realtime-engine | 1 import + dependency pin | stoptimes-serve + model libs | yes (as today) | yes, if the registry is shared | none |
| **O2 Pipeline, in-process** (former G2b, superseded) | realtime-engine | pipeline stages inside databus | same as O1 | yes | yes | none |
| **O3 Prediction service** (target, G2b–G2c) | its own container | pipeline file, compose service, remove the producer, ARCHITECTURE.md | **nothing** | **no** | yes | one container |
| O3 + MLflow in-process (G2d) | the service | none beyond O3 | nothing | no | yes, by moving an alias | MLflow tracking server + store |
| O3 + MLflow scoring server | a model server behind the service | none beyond O3 | nothing | no | redeploy the model server | MLflow server per model |
| **O4 databus calls MLflow directly** | MLflow | rebuild the feature code in databus | mlflow client | yes | yes | MLflow |

**Recommendation:** O1 first, because it is a one-line reversible cutover. Then O3, run in shadow mode
before it takes over. MLflow is an optional model source for O3, not a separate path. O2 is superseded. O4 is
rejected.

## 3. The options

### O1 · Library, in-process (G2a)

`stop_times.py` swaps `gtfs_eta.eta_service.estimator` for `stoptimes.serve.estimator` (same
`estimate_stop_times` signature; in Phase 3 it became a thin adapter over the pipeline) and
`gtfs_eta.feature_engineering.spatial.ShapePolyline` for `stoptimes.core.spatial.geo.ShapePolyline`.

- **Benefits.** Smallest possible change, rollback is a revert, and databus's own shape projection keeps
  supplying `shape_distance_to_stop` exactly as now.
- **Trade-offs.**
  - The model still runs in the telemetry task.
  - stoptimes-serve and the model libraries have to install and import on **Python 3.14**.
    Check the scikit-learn/xgboost wheels before scheduling it. The LSTM needs torch and is out of
    reach on 3.14.
  - The artifact must load under databus's library versions: **G4 comes before G2a** (U7 refuses a
    drifted artifact).
- **databus needs:** §5.1.

### O2 · Pipeline, in-process (superseded)

databus supplies `ingest`/`publish` overrides and runs the pipeline inside `realtime-engine`. It removes the
duplicated feature code just as O3 does, but keeps every O1 cost: shared venv, Python 3.14, the model in the
telemetry path. Kept here only to record why it was dropped (2026-09-19).

### O3 · Prediction service (target)

A `stoptimes serve` container runs next to databus on the same Docker network. It loops every few seconds,
reads active runs from databus's Redis, predicts them all in one batch, and writes
`run:<id>:stop_time_updates`. databus only reads the key, as it already does.

```
realtime-engine ──writes──▶ Redis (runs:in_progress, run:<id>, vehicle:<id>:position, stop status)
                                  │ read every N s
                  prediction-engine = stoptimes serve --pipeline databus_pipeline.py:pipeline
                    ingest (databus) → match → features → predict → postprocess → publish (databus)
                                  │ write (EX 60)
                             run:<id>:stop_time_updates ──read──▶ schedule-engine (unchanged)
```

- **Benefits.**
  - Telemetry ingestion no longer waits on the model.
  - The model's dependencies and Python version are fully separate from `backend_venv`.
  - One batched predict call per cycle, so heavier models (LSTM) become realistic.
  - Models are promoted and rolled back without a databus deploy.
  - Feature computation is the same code as training (U6), not a second copy in databus.
  - `schedule-engine`, the Redis contract and the feed are unchanged.
- **Trade-offs.**
  - **Freshness.** Predictions are up to one cycle older than today's per-position ones. A 5 s cycle
    against a 15 s feed build is acceptable. Record the source position's timestamp so staleness is
    visible (§5.2).
  - **Architecture.** A second writer to Redis, which ARCHITECTURE.md currently reserves for
    `realtime-engine`. It needs an amended mandate (§5.2). The precedent is `schedule-engine`, which
    already runs on a timer over a Redis snapshot.
  - **Static GTFS consistency.** The service needs shapes and stop times. It must use the same feed
    version databus publishes, or `trip_id`s and stop sequences drift.
  - **Operations.** One more container to deploy and monitor. When it is down, keys expire within 60 s and
    the feed carries TripUpdates without stop time updates. That is the same degradation as "no model
    today", and it should be visible on the ops dashboard.
  - **Loop routes.** databus's `STOPPED_AT` / `current_stop_sequence` filtering and bUCR's loop-back-safe
    shape distance must be carried into the service. The default `match` stage projects onto the shape
    only, so pass stop status through `ingest` and use it in a databus `match` override.
- **databus needs:** §5.2.

#### Model source for O3

The service's model can come from three places. Changing between them is configuration only.

| Source | How | Benefits | Trade-offs |
|---|---|---|---|
| **Filesystem registry** (default) | `--registry /models`, a volume; `models promote` / rollback (U7) | no extra services; what G2b ships with | promotion means writing to the volume on the host |
| **MLflow, in-process** | `STOPTIMES_REGISTRY=mlflow+http://mlflow:5000`; pins map to aliases (`@route-22`) | promotion and rollback by moving an alias, with UI and audit trail; no network call per prediction | needs an MLflow server and store; the service re-checks aliases every few minutes; the U7 drift refusal still applies |
| **MLflow scoring server** | `mlflow models serve` container; the service calls it via `HttpModel` with `request_format: mlflow` | model runtime isolated even from stoptimes | one server per model (per-route choice needs a router model); a network call per cycle; an alias change means restarting the server |

Details: [MLFLOW_INTEGRATION.md](MLFLOW_INTEGRATION.md) (I2, I4, I6). Whichever is chosen, it is invisible to
databus.

### O4 · databus calls MLflow directly (rejected)

databus would have to rebuild the model's inputs itself: shape progress, kinematic and temporal features.
That brings back the training/serving duplication G2b exists to remove, and puts the MLflow client in
databus's venv. The model would also be invisible to stoptimes' selection policy, drift checks and `val`.
Rejected.

---

## 4. Rollout

Also in PLAN.md §8 G2. Each step needs the user's explicit go.

| Step | What happens | Rollback |
|---|---|---|
| **G4** | Re-export the served artifacts under databus's library versions, or set `STOPTIMES_ALLOW_VERSION_DRIFT` as a logged exception | — |
| **G2a** | O1 import swap | revert the databus commit |
| **G2b** | O3 service in **shadow**: writes `run:<id>:stop_time_updates:shadow`; production still comes from G2a. Compare live and by replaying one recorded bUCR day through `val` with both | stop the container |
| **G2c** | Service writes the real key; the same databus deploy removes `produce_stop_times` | stop the service and revert the databus deploy (back to G2a); keys expire in 60 s |
| **G2d** (optional) | Switch the service's registry to MLflow; run the MLFLOW_INTEGRATION.md §6 comparison | point the registry back at the volume |

## 5. What databus needs to change

### 5.1 For G2a (library, in-process)

- [ ] `backend/pyproject.toml`: replace the `backend/gtfs-eta` symlink with a pinned `stoptimes-serve`
      dependency (git ref until PyPI), with `[xgboost]` if the served family needs it. Verify it resolves
      on Python 3.14.
- [ ] `runs/domain/progression/stop_times.py`: two imports (estimator module, `ShapePolyline`). Keep
      calling via the module attribute so the existing test patching still works.
- [ ] Environment: `MODEL_REGISTRY_DIR` → `STOPTIMES_MODEL_REGISTRY_DIR`; `ETA_*` names remain readable
      (deprecated). Registry volume mounted into `realtime-engine`.
- [ ] Before deploy: load the served artifact under databus's versions (G4).
- [ ] Tests: existing `stop_times` / `test_mqtt_ingestion` suites pass with the new patch target.

### 5.2 For G2b–G2c (prediction service)

**Compose service** (dev and prod), on the same network as `state`:

```yaml
prediction-engine:
  image: <stoptimes-serve image>        # built from this repo (G2 prerequisite)
  command: >
    stoptimes serve --source bucr --static /gtfs/current.zip
      --pipeline /databus/databus_pipeline.py:pipeline
      --registry /models --interval 5
  environment:
    REDIS_HOST: state
    REDIS_PASSWORD: ${REDIS_PASSWORD}
    DATABUS_PREDICTIONS_KEY_SUFFIX: ":shadow"   # G2b; empty at G2c
  volumes:
    - ./prediction/databus_pipeline.py:/databus/databus_pipeline.py:ro
    - models:/models
    - gtfs:/gtfs:ro
  depends_on: [state]
```

**The pipeline file, owned by databus** (it encodes databus's Redis schema, so it belongs with `keys.py`,
not in stoptimes). Sketch:

```python
from stoptimes.serve.pipeline import build_default_pipeline
from stoptimes.core.contracts.data import Observation, ObservationKind

pipeline = build_default_pipeline()

@pipeline.stage("ingest")
def ingest(_payload, ctx) -> list[Observation]:
    # For each run in runs:in_progress: read run:<id>, vehicle:<id>:position and stop status.
    # Emit one vehicle_position Observation per run; stash stop status in
    # ctx.state["stop_status"][trip_id] for the match override.
    ...

@pipeline.stage("match")
def match(observations, ctx):
    # Default map-matching, then drop stops already served (STOPPED_AT / current_stop_sequence),
    # keep at most max_stops, and apply bUCR's loop-back-safe shape distance.
    ...

@pipeline.stage("publish")
def publish(predictions, ctx) -> None:   # list[StopTimePrediction]
    # Per run: stop_time_updates.to_redis([...5 fields...]) → SET run:<id>:stop_time_updates<suffix> EX 60.
    # Skip the write when a run has no predictions (keep last good). Also SET a heartbeat key.
    ...
```

It needs a Redis client and databus's `keys` / `stop_time_updates` helpers. Either vendor those two small
modules into the file's directory, or publish them as a tiny package. The service must not import Django.

**Contract and state:**

- [ ] Freshness: record the timestamp of the position each prediction came from. Adding a sixth field
      breaks `stop_time_updates.to_redis`'s five-field contract, so prefer a sibling key
      `run:<id>:stop_time_updates:meta` = `{source_ts, computed_at, artifact_id}`. `schedule-engine` may
      then skip predictions older than a threshold. Optional at G2b, recommended by G2c.
- [ ] Heartbeat: `prediction-engine:heartbeat` (EX ~3× interval) with the last cycle's run count and error
      count, shown on the ops dashboard.
- [ ] Static GTFS: the service loads the same GTFS Schedule databus publishes (shared volume, or the URL
      schedule-engine serves) and reloads when it changes. It never reads the Django ORM.

**Code removal at G2c:**

- [ ] `realtime_engine/tasks.py`: remove the `produce_stop_times` call in the same deploy that switches the
      service to the real key. Never two writers to one key. Keep `stop_times.py` until G2c has been stable
      for a while, then delete it with its tests.

**ARCHITECTURE.md amendment** (proposed text):

> ### Prediction engine
> **Role:** Arrival-time prediction for active runs.
> **Responsibilities:** read the operational state snapshot on a fixed interval; write the derived
> `run:<id>:stop_time_updates` projection (and its `:meta`) with a TTL; publish a heartbeat.
> **Does NOT:** interpret raw telemetry, modify operational state, build or publish GTFS-RT feeds, serve APIs.
> **Implemented by:** `stoptimes serve` with a databus-owned pipeline file.

Also update the State (Redis) section: the "written by realtime-engine only" rule becomes "operational
state is written by realtime-engine only; derived prediction projections by prediction-engine".

**Tests, in databus:**

- [ ] The pipeline file against a fake Redis: active runs in → keys out in the exact `from_redis` format;
      ended runs are ignored; no predictions → no write.
- [ ] A recorded bUCR day replayed through the service, diffed against G2a's output (the shadow comparison
      done off-VPS first).
- [ ] `schedule-engine` builder unchanged, with an optional test for the `:meta` staleness skip.

### 5.3 For G2d (MLflow, optional)

Nothing in databus beyond an `mlflow` service in compose (or an external one). The prediction service's
environment changes to `STOPTIMES_REGISTRY=mlflow+http://mlflow:5000`. MLflow can use databus's Postgres (a
separate database) as its backend store, and a volume or S3 for artifacts.

## 6. What stoptimes provides

These are the G2 prerequisites in PLAN.md §8:
- a `serve` Docker image;
- reloading the static GTFS when databus publishes a new version;
- the heartbeat and `:meta` support;
- documentation of the prediction-service use;
- the U6 pipeline, the U7 registry lifecycle and drift refusal, already merged.

## 7. Open questions

- **Cycle interval.** 5 s is the starting proposal. The right value depends on bUCR's position rate and the
  15 s feed cadence.
- **Empty TripUpdates.** A TripUpdate with no `stop_time_update` is how databus handles "no prediction"
  today. Check it against the GTFS-RT spec (it expects at least one unless the trip is canceled) and against
  what infobus does with it. This is a pre-existing question, not one O3 introduces.
- **Where the pipeline file lives in databus** (`backend/prediction/`?), and whether the two helper modules
  are vendored or packaged.
- **`uncertainty`.** It is a constant 120 s today. The service could fill it from quantile models later, and
  the contract already has the field.
