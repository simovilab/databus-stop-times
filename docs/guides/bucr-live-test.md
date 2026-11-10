# Guide: live-test the bUCR model on a local databus

A runbook for one day of production-like validation. You run the databus dev stack, start
runs for the real bUCR buses as they depart, and databus emits GTFS-RT with real
`trip_id`s built from their NavSat pings. `stoptimes val` then scores the frozen live
model (`mlflow:stoptimes-bucr@staging`) against arrivals detected from those same pings.

bUCR has no agency TripUpdates the way MBTA does. Your baselines are the timetable
(`baseline:schedule`) and what databus publishes today (`gtfsrt`). The segment-history
baseline from CV has no `val` predictor; see [§5](#5-start-the-validation-run).

Verified on 2026-11-09 against this repo (`feat/bucr-inferred-trips` @ `a988f73`) and the
local databus checkout (`feat/ops-dashboard` @ `39a479d`; `main` is `1be2da1`). Commands
come from `--help` and the source. **Q:** marks an open question for you.

!!! warning "Before you start"
    - **The model isn't published yet.** On 2026-11-09 the only registered model on
      `ml.simovi.org` was `stoptimes-mbta` v1 `@staging`. Publish `stoptimes-bucr` with
      `@staging` first; until then `mlflow:stoptimes-bucr@staging` fails at run start.
      The frozen artifact is `~/stoptimes-registry/xgb-live-bucr-final`. Publish it from the
      repo root. The upload takes several minutes, so run it in the background:
      ```bash
      nohup uv run --no-sync stoptimes models publish xgb-live-bucr-final \
        --registry ~/stoptimes-registry --to mlflow+https://ml.simovi.org \
        --alias staging --json > /tmp/bucr-publish.log 2>&1 &
      # done when the log shows "stoptimes-bucr", version 1, aliases ["staging"]
      ```
    - **The bUCR calendar ends 2026-11-27.** Service id `entresemana` runs weekdays
      20260810–20261127, with 31 Aug, 15 Sep, 12 Oct and 1 Dec removed. On any other day
      the model's schedule features have no service and the test is meaningless.

## What the model is, in one paragraph

XGBoost on 19 live features, computed from the trip state of VehiclePositions that carry a
GTFS `trip_id`. It was trained on bUCR NavSat pings matched to the timetable
([session notes](../research/SESSION_2026-11-09_BUCR_INFERRED_TRIPS.md)). Its config is
`examples/experiments/xgb-live-bucr-final.yaml`: refit on all 62 weekdays from 10 Aug to
6 Nov.

Rolling-origin CV, pooled MAE:

| Predictor | Pooled MAE |
|---|---|
| Model | **52.1 s** (fold std 4.2 s, range 46.4–60.5) |
| Timetable | 93.3 s |
| Segment history | 57.9 s |

It is weak beyond 20 minutes (about 240 s MAE there). In `val` the live model scores
through the same `LiveSession` that `stoptimes serve` uses. A trip whose newest ping is more
than 30 s older than the newest ping in the feed gets no prediction (`MAX_PING_AGE_S`,
`packages/serve/src/stoptimes/serve/live_session.py:60`).

## 0. Prerequisites

| What | How | Check |
|---|---|---|
| Workspace deps | `uv sync --group dev` from the repo root (per `web/README.md`) | `uv run --no-sync stoptimes val --help` |
| MLflow server | `export STOPTIMES_MLFLOW=https://ml.simovi.org` | the model resolves (below) |
| MLflow credentials | `~/.mlflow/credentials` (already present, mode 600), or `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` | never echo them |
| Tracking (optional) | `export STOPTIMES_TRACKING=https://ml.simovi.org` logs each finished run to experiment `stoptimes-val/bucr` | the run link appears in API telemetry |
| Sources | run everything **from the repo root**, so `./examples/agencies/bucr.yaml` is found, or set `STOPTIMES_SOURCES` | `uv run --no-sync stoptimes val sources` lists `bucr` |
| Node 20+ | for the web UI | `node -v` |

Check that the model resolves without starting a run. This prints only names and versions,
never credentials:

```bash
STOPTIMES_MLFLOW=https://ml.simovi.org uv run --no-sync python -c "
from stoptimes.core.registry import mlflow_registry
for v in mlflow_registry().versions(): print(v.name, v.version, list(v.aliases), v.artifact_id)"
# expect a line: stoptimes-bucr <n> ['staging'] <artifact_id>
```

`$STOPTIMES_MLFLOW` and `$STOPTIMES_TRACKING` are read when the process starts:

- **The API.** `RunManager` builds its tracker in `__init__`
  (`packages/val/src/stoptimes/val/api/runs.py:79`), so export both **before** starting the
  API. Restart it after any change, and after pulling `a988f73` (the fix that moves
  `/metrics` off the event loop).
- **The CLI.** `--track <uri>` overrides `$STOPTIMES_TRACKING`, and `--track ''` turns
  tracking off.

## 1. Bring up databus dev

Follow databus's own instructions: `README.md` → *Getting Started*, and `HOWTO.md` in the
databus repo (`~/Desktop/SIMOVI/git.no_sync/databus`). In short:

```bash
cd ~/Desktop/SIMOVI/git.no_sync/databus
cp .env.example .env      # first time only; edit values
./scripts/dev.sh          # builds, starts, waits for healthy (1-2 min first run)
```

- **Q: which databus branch?** The checkout is on `feat/ops-dashboard`, 45 commits ahead
  of `main`. Only that branch has `start_date`/`start_time` in the TripDescriptor (`32c0158`)
  and the shared HTTP-source dedupe (`39a479d`). Without `start_date` the service date is
  derived from the ping time, which works but is less exact. Use `feat/ops-dashboard`
  unless you have a reason not to.
- **Ports.** The orchestrator is on `:8000`, Redis on `:6379`, MQTT on `:1883`, and
  **databus's Nuxt `user-interface` on `:3000`**. That collides with the val web UI's
  default, which is handled in step 5.
- **GTFS.** At startup the entrypoint loads `publishers.json` and runs
  `bootstrap_schedule` from `https://feeds.simovi.org/bucr/schedule/gtfs.zip`, the feed
  the model was trained on. It exports `backend/feed/files/gtfs.zip` only if that file is
  absent. The local one is dated 2026-09-07 and has the right calendar (122 trips,
  `entresemana` to 20261127, `bUCR_0_11` = Odontología). If the upstream feed changes,
  delete the old zip so it gets re-exported, or the `trip_id`s can drift.
- **The demo fleet.** `demo_fleet.json` loads only when `DEBUG` is on. It contains the six
  real NavSat buses (`299-1014`, `299-1015`, `299-604`, `299-921`, `299-922`, `299-987`),
  each with an HTTP `Sensor` "GPS <plate>".

!!! danger "The fixture's NavSat URLs are placeholders"
    Every sensor's `source_http_url` in `backend/operations/fixtures/demo_fleet.json` is
    `https://navsat.example.com/REPLACE_WITH_…`. Set the real NavSat URL on each Sensor in
    your local database (Django admin at `http://localhost:8000/admin`). Otherwise
    `fetch_positions` fetches nothing and no real pings ever reach databus. The URL embeds a
    vendor key, so keep it out of git and out of logs.

    **Q:** where does the real URL come from (the VPS DB, `.env`)?

How real pings flow:

1. Celery beat runs `fetch_positions` every 10 s.
2. It polls only ACTIVE `http`/`both` sensors whose vehicle has `vehicle:<id>:current_run`
   set, which means a run must exist first.
3. It republishes each position to `transit/vehicle/<id>/position` on MQTT.
4. `realtime-engine` map-matches the position.
5. `schedule-engine` rebuilds the feeds every **15 s**.

Don't send simulated MQTT for these six vehicles.

## 2. Start runs as the real buses depart

Your simulator (`~/Desktop/SIMOVI/git.no_sync/databus-sim`, branch `feat/django-simulator`)
drives databus's run lifecycle over REST. Start it with `./scripts/dev.sh` or
`docker compose up --build`; its UI is at `http://localhost:8080`. Its `.env` must point at
local databus, not the VPS profile.

What the source shows:

- **The roster is hard-coded** to simulated units `unit-01`…`unit-06`
  (`simulator_app/domain/fleet.py:44`).
- **The schedule** (`simulator_app/data/schedule.yaml`) requires `vehicle_id` to be "in
  the fleet roster".
- **`NAVSAT_URL`** in the simulator only draws the real buses as a **map overlay**. It
  does not create runs for them.

**Q:** how does your simulator start a run for a *real* vehicle (e.g. `299-922`)? Does the
Operator tab's "Request Run" accept any databus vehicle, or do you have a local change?

If not, the databus REST calls below do the same thing. They are verified against
`backend/api/urls.py` and `serializers.py`. Local dev has no auth on these views, and the
operator must already exist (**Q:** which operator id do you use locally?).

```bash
DB=http://localhost:8000
# 1. create the run (Initialized). trip_id/shape_id/direction_id from the local gtfs.zip trips.txt
curl -s -X POST $DB/api/create-run/ -H 'Content-Type: application/json' -d '{
  "vehicle_id": "299-922", "operator_id": "<operator>", "route_id": "bUCR",
  "trip_id": "desde_educacion_a_odontologia_sin_milla_entresemana_11:45",
  "direction_id": 0, "shape_id": "desde_educacion_a_odontologia_sin_milla",
  "schedule_relationship": "SCHEDULED"}'
# 2. confirm it (operator action) -> Confirmed; pings then move it to Tracking -> In Progress
curl -s -X POST $DB/api/runs/<run_id>/update/ -H 'Content-Type: application/json' \
  -d '{"event": "run_confirmed_by_operator", "details": {}}'
curl -s $DB/api/runs/<run_id>/state/
```

Choosing the trip id is the step that matters. The run's `trip_id` must be the timetable
trip that this bus is actually running at this departure. A wrong match is scored as if the
bus were very early or very late, and the model's schedule-delay features go wrong with it.
Pick the trip whose scheduled departure (`HH:MM` suffix) and variant (origin, *con/sin
milla*) match what the bus is doing. Training rejected about 3% of legs as unmatched or
contested; live, you are the matcher.

Only runs in `runs:in_progress` appear in the feed. A run completes on its own when the bus
is `STOPPED_AT` (≤ 20 m) the terminal stop. It goes to No Signal after 60 s without pings
and is Cancelled after 600 s.

## 3. Confirm the feed carries trip ids and positions

`examples/agencies/bucr.yaml` points at the right paths. databus serves **both** JSON and
protobuf (`backend/feed/urls.py`, same on `main` and `feat/ops-dashboard`):

| Path | Content |
|---|---|
| `/feed/schedule/feed.zip` | GTFS Schedule (404 until exported) |
| `/feed/realtime/vehicle_positions.json` / `.pb` | VehiclePositions |
| `/feed/realtime/trip_updates.json` / `.pb` | TripUpdates |

The JSON is the builder's snake_case dict (`json.dumps`, not `MessageToJson`). That is the
shape `stoptimes.core.realtime.gtfsrt_json` reads. `val`'s `--vp-format/--tu-format`
default `auto` sniffs each response, so JSON (the yaml default) is fine. The `.pb` paths
need the `pb` extra (`gtfs-realtime-bindings`).

```bash
curl -sI http://localhost:8000/feed/schedule/feed.zip | head -1          # 200
curl -s http://localhost:8000/feed/realtime/vehicle_positions.json | python3 -m json.tool | head -40
# One line per vehicle: trip_id, position, age of the ping, stop status
curl -s http://localhost:8000/feed/realtime/vehicle_positions.json | python3 -c "
import json,sys,time; f=json.load(sys.stdin); now=time.time()
print('header age', round(now-f['header']['timestamp']),'s; entities', len(f['entity']))
for e in f['entity']:
    v=e['vehicle']; t=v.get('trip',{}); p=v.get('position',{})
    print(e['id'], t.get('trip_id'), t.get('start_date'), p.get('latitude'), p.get('longitude'),
          'ping age', round(now-v.get('timestamp',0)),'s', v.get('current_status'), v.get('stop_id'))"
```

Check each of these:

- **Every entity has `trip.trip_id`.** `val` silently drops entities without one.
- **The `trip_id` is in `trips.txt` of `feed.zip`.** `val` skips trips that aren't in the
  static feed.
- **Ping age stays under about 30 s.** The header refreshes every 15 s and the vehicle
  `timestamp` is the NavSat ping time. An older ping gets no model prediction (the 30 s
  bound).
- **`start_date` is today** (on `feat/ops-dashboard`).
- **`current_status`/`stop_id` advance** as the bus moves.

`trip_updates.json` should have one entity per run, with `stop_time_update`s for the
**next 3 stops at most** (`ETA_MAX_STOPS=3`, uncertainty fixed at 120 s). An empty
`stop_time_update` list is databus's "no prediction".

!!! note "If `vehicle_positions.json` returns 500"
    The view opens the file without an existence check. It errors until `schedule-engine`
    has built the feed once, about 15 s after startup.

## 4. Start the val API and the web UI

```bash
# terminal A: API (repo root). Port 8010 by default (ETAVAL_API_PORT); data in ./.etaval_data
export STOPTIMES_MLFLOW=https://ml.simovi.org
export STOPTIMES_TRACKING=https://ml.simovi.org     # optional
export ETAVAL_CORS_ORIGINS=http://localhost:3001    # databus's UI already owns :3000
uv run --no-sync stoptimes val api

# terminal B: UI on 3001
cd web && npm install && npm run dev -- --port 3001     # -> http://localhost:3001
```

The UI talks to `http://localhost:8010` (`NUXT_PUBLIC_API_BASE` overrides it). The CORS
default is `http://localhost:3000`, which is databus's UI when the stack is up. Hence port
3001 and the explicit `ETAVAL_CORS_ORIGINS`.

Known slow spots after `a988f73`. They don't block other pages any more:

- `GET /api/predictors` takes about 24 s while the MLflow listing is cold (60 s cache; the
  API warms it at startup and drops `mlflow:` ids if MLflow takes over 30 s).
- `GET /api/runs/{id}/metrics?stats=1` takes about 14 s on a large run.

## 5. Start the validation run

The predictor ids come from `packages/val/src/stoptimes/val/engine/assembly.py`.

**Fixed `PREDICTOR_REGISTRY`:**

- `gtfsrt`
- `baseline:constant_speed`
- `baseline:schedule`
- `stoptimes:{historical_mean,polyreg_distance,polyreg_time,xgboost}`
- deprecated `gtfs_eta:*` aliases of the line above

**Dynamic prefixes:** `stoptimes:<artifact_id>`, `pipeline:<file.py>`, and
`mlflow:<model>@<alias>` or `mlflow:<model>/<version>`.

**`DEFAULT_PREDICTOR_IDS`:** `["gtfsrt", "baseline:constant_speed"]`. These are used only
when you name none.

Use these four:

| Id | What it is |
|---|---|
| `mlflow:stoptimes-bucr@staging` | the model. The alias resolves to one version **at run start**; later alias moves don't affect the run |
| `baseline:schedule` | timetable, shifted by the current offset at the vehicle's progress. The live counterpart of CV's `sched_rem` (93.3 s) |
| `baseline:constant_speed` | recent along-shape speed |
| `gtfsrt` | what databus publishes today: its gtfs-eta producer with the **synthetic** `polyreg_distance_global_baseline_v0` (`backend/eta_models/`), next 3 stops only |

`gtfsrt` is scored only on horizons up to three stops, so compare it within the horizon
buckets, not on overall MAE.

There is **no segment-history predictor** in `val`. CV's `prior_rem` (57.9 s) lives only in
`packages/train/.../live_cv.py`. Don't substitute `stoptimes:historical_mean`, for two
reasons:

- It is the legacy August segment-model path (`GtfsEtaModelPredictor` →
  `estimate_stop_times`).
- With `STOPTIMES_AGENCY=bucr` the yaml's `serving:` pin overrides the requested family,
  so it would actually serve the pinned August xgboost segments model under a misleading
  id.

Use the CV number as the reference, or add a `val` predictor later.

**CLI** (headless; all flags from `stoptimes val run --help`):

```bash
mkdir -p runs
uv run --no-sync stoptimes val run --source bucr \
  --predictor mlflow:stoptimes-bucr@staging \
  --predictor baseline:schedule \
  --predictor baseline:constant_speed \
  --predictor gtfsrt \
  --duration 3h \
  --db runs/bucr-live-$(date +%F).db \
  --out runs/bucr-live-$(date +%F).parquet
```

`--db` keeps the SQLite file; the default is in-memory. You need it for `--sensitivity`
later. `--route bUCR` is unnecessary because there is one route. The API's `POST /api/runs`
takes `{source, predictors, routes, duration, seed}` and has no format flags, so the yaml
and `auto` apply.

**UI:** open `http://localhost:3001`, pick source `bucr`, tick the four predictors (wait for
the `mlflow:` ids to appear), set the duration (`3h`), then start. With
`STOPTIMES_TRACKING` set, the finished run (done, or ended early with an error) is logged in
the background to `stoptimes-val/bucr`. Metrics are keyed per predictor, e.g.
`mlflow_stoptimes-bucr_staging.lt2m.mae_s`, and the resolved model version is a tag
(`packages/val/src/stoptimes/val/tracking.py`).

Equivalent curl:

```bash
curl -s -X POST http://localhost:8010/api/runs -H 'Content-Type: application/json' -d '{
  "source":"bucr","duration":"3h",
  "predictors":["mlflow:stoptimes-bucr@staging","baseline:schedule","baseline:constant_speed","gtfsrt"]}'
```

`bucr.yaml` polls every 5 s, but databus rebuilds the feed every 15 s. The tracker appends
every polled record, so each real ping is seen about 3 times and each predictor emits about
3 identical samples per ping. This weights all predictors the same and doesn't bias the
comparison, but sample counts are about 3× the ping count.

## 6. Ground truth: how arrivals are detected

`build_detectors` (`engine/assembly.py:299`) runs three detectors on every tick. The
**scored** one is `shape_distance`, which is hard-coded as `primary_method` in the CLI
(`cli/main.py:348,448`) and the API (`api_verbs.py:144`). The other two are saved for
`--sensitivity`.

| Method | Rule | bucr.yaml param |
|---|---|---|
| `shape_distance` (scored) | the stop is reached when a ping's along-shape progress is within `shape_threshold_m` of the stop's progress, on either side | `shape_threshold_m: 25` |
| `position_radius` | first ping within `arrival_radius_m` great-circle of the stop | `arrival_radius_m: 30`, `dwell_s: 0` |
| `stop_status` | first `STOPPED_AT` with a matching `current_stop_sequence` (databus: ≤ 20 m) | — |

### The Odontología risk (stop `bUCR_0_11`)

Buses park about 6 m (along the shape) short of Odontología's progress point. In training,
a crossing rule (progress ≥ stop) fired only when the bus left, about 30 min late. The fix
there was `live_frame.with_radius_arrivals` with `--arrival-radius-m 30`
(`packages/train/src/stoptimes/train/live_frame.py:258`).

**What val does.** Val's ground truth is **not crossing-based**, so the
30-minutes-late failure does not happen in `val run`/`replay`/the API.

- `ShapeDistanceDetector.detect` (`packages/val/src/stoptimes/val/groundtruth/shape_distance.py:63-83`)
  takes the closest-approach sample in `|progress − stop_progress|` and accepts it if
  that distance is ≤ 25 m. A bus 6 m short qualifies.
- The engine runs it on every tick over the history so far and keeps only the **first**
  arrival per (trip, stop, method) (`engine/runtime.py:550-565`).
- So in practice the arrival is the first ping within 25 m along the shape, the same
  idea as training's 30 m radius but slightly tighter.

**What can still go wrong:**

1. **Closest approach over a full history.** `detect()` itself picks the closest sample,
   not the first, so it is only safe while it is called tick by tick. Any batch use over a
   complete trip history would pick an arbitrary ping from the layover, up to 30 min late.
   No such call exists today. Also, if a tick brings several new in-threshold pings at
   once (after a feed stall), the closest one wins rather than the first. Suggested fix,
   not applied: return the **first** sample with
   `|Δprogress| ≤ shape_threshold_m` instead of `min(...)` at `shape_distance.py:63-69`,
   so that ground truth doesn't depend on how often `detect()` is called.
2. **The terminal arrival can be missing.** databus completes the run on the first
   `STOPPED_AT` (≤ 20 m) at the terminal and removes it from `runs:in_progress`. The feed
   rebuilds every 15 s, so the in-threshold ping may never reach `val`. Odontología would
   then have **no** arrival and its predictions would go unscored (dropped, not wrong).
   Check the arrival count for `bUCR_0_11` (below).
3. **The two radii differ** (25 m along-shape scored, versus 30 m in training). This is
   small in practice. `stoptimes val report <parquet> --sensitivity --db <db>` shows how
   MAE moves when `position_radius` (30 m, great-circle) is the truth instead.

Separately: the history is projected point by point with `project_point_to_polyline`
(`engine/runtime.py:459-466`), not monotonically, even though the detector's docstring
says otherwise. On bUCR's shapes that pass close to themselves, a ping can project onto a
later stretch and fire an early arrival for a later stop. Watch for negative errors at one
stop that are much larger than at its neighbours.

Check the Odontología arrivals in the run database (the `--db` file, or
`./.etaval_data/runs/<run_id>.sqlite` for API runs):

```bash
sqlite3 runs/bucr-live-$(date +%F).db \
  "select method, count(*) from actual_arrivals where stop_id='bUCR_0_11' group by method;"
```

The tables (`storage/sqlite.py`) are `prediction_samples`, `actual_arrivals` and
`validation_results`. Expect about one arrival per inbound trip per method. Many fewer `shape_distance`
arrivals than trips means risk 2 is happening.

## 7. Reading the results

```bash
uv run --no-sync stoptimes val report runs/bucr-live-$(date +%F).parquet
uv run --no-sync stoptimes val report runs/bucr-live-$(date +%F).parquet \
  --sensitivity --db runs/bucr-live-$(date +%F).db
uv run --no-sync stoptimes val figures --help    # 300 dpi PNGs of the same breakdown
```

The report has four parts:

1. Per predictor: MAE, RMSE, MAPE, bias and count.
2. MAE by horizon bucket.
3. Head-to-head win rate.
4. With `--sensitivity`: MAE under each detector.

In the UI, the run page shows the predictor comparison, *MAE by horizon bucket*, error
distribution, convergence, head-to-head and, after the run, `?stats=1` CIs, significance and
calibration.

`val`'s horizon buckets are `<2m`, `2-5m`, `5-10m` and `>10m`
(`scoring/metrics.py:33`). There is **no 20-minute bucket**, so the model's weak 20+ min
range is folded into `>10m`. To see it, filter the parquet on `horizon_s > 1200`.

What "good" looks like. These are leak-free CV numbers to compare against; a live day is
noisier:

| Bucket | Model in CV | Timetable in CV |
|---|---|---|
| < 2 min | ~19 s | ~44 s |
| 2–5 min | ~37 s | ~78 s |
| 5–10 min | ~56 s | ~116 s |
| 10–20 min | ~83 s | ~130 s |
| 20+ min | ~200–240 s | ~300 s |

The per-bucket figures come from the earlier `xgb-live-bucr` validation split. Only the
pooled numbers are from the final CV.

- **Pooled.** CV gave 52.1 s for the model and 93.3 s for the timetable. A live MAE within
  roughly ±10 s of 52 s, with `baseline:schedule` near 90 s, means it holds up. Fold MAE
  ranged 46–61 s.
- **Bias.** It should be near zero (CV +8 s). A large constant bias usually means a wrong
  `trip_id` match or service-date problem, not the model.
- **One day is small.** About 100 trips a day across six buses, less for a partial day.
  Check the `?stats=1` CIs before drawing conclusions.

Expected weak spots:

- horizons beyond 20 min
- pre-departure predictions at an origin terminal: a run already In Progress while the bus
  sits at Educación/Artes/Odontología
- the depot stretch (Sección de Transportes, about 580 m before Odontología), where a bus
  can stop out of service
- trips you matched to the wrong timetable entry
- any stretch where a bus's pings go stale for more than 30 s (no model prediction, while
  the baselines still predict, so counts differ)

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `mlflow:stoptimes-bucr@staging` errors at start: "needs an MLflow server" | `STOPTIMES_MLFLOW` not exported in the process that runs the predictor | export it and restart the API |
| …"unknown" ref, or no `mlflow:` ids in the UI | model or alias not published yet; or MLflow listing over 30 s | run the `versions()` check in §0; wait a minute and reload (60 s cache) |
| 401/403 from MLflow | credentials | check `~/.mlflow/credentials` (don't print it) |
| UI loads but every API call fails (CORS) | UI on 3001, API still allowing only 3000 | `ETAVAL_CORS_ORIGINS=http://localhost:3001`, restart the API |
| UI opens databus's frontend | both on 3000 | `npm run dev -- --port 3001` |
| `vehicle_positions.json` 500 | feed file not built yet | wait 15 s for `schedule-engine` |
| feed has 0 entities | no run is In Progress; bus hasn't left yet; run Cancelled after 600 s without pings | check `GET /api/runs/<id>/state/` |
| run stuck in Confirmed/Tracking, no pings | sensor URL is still the `navsat.example.com` placeholder; sensor not ACTIVE; `vehicle:<id>:current_run` not set | fix the Sensor in admin; check `realtime-engine` logs (`docker compose -f compose.dev.yml logs -f realtime-engine`) |
| model has far fewer samples than `baseline:schedule` | pings older than 30 s are skipped | check ping age (§3); NavSat cadence |
| trips silently missing in `val` | `trip_id` not in `feed.zip`, or entity lacks `trip_id` | re-export the zip; check the run's `trip_id` |
| no arrivals at Odontología | risk 2 in §6 | check the `actual_arrivals` count; compare with `--sensitivity` |
| run page "Loading…" for ~14 s | `?stats=1` on a big run | expected after `a988f73`; other pages stay responsive |
| nothing logged to MLflow | `STOPTIMES_TRACKING` unset when the API started | export it and restart the API; for the CLI use `--track https://ml.simovi.org` |
| `gtfsrt` MAE looks great | it predicts only the next ≤ 3 stops | compare it per horizon bucket |

## Open questions for you

- How does the simulator start runs for real vehicles (§2)? And which operator id do you
  use locally?
- Where does the real NavSat sensor URL come from for the local DB?
- Which databus branch will you run (`feat/ops-dashboard` recommended)?
- What is the NavSat ping cadence per bus? If it is often over 30 s, the model will skip
  predictions.
