# Guide: live features (schedule delay, last observed delay, path prior)

A model that only knows "distance to the stop, stops ahead, time of day" cannot see that the
bus is already 6 minutes late. The **live** features can. In the 2026-11 MBTA study the 19-feature
`xgb-live` model reached 95.5 s MAE against 118.1 s for MBTA's own TripUpdates on 14 held-out days,
while the best static-only model scored 211.8 s.

## What an agency needs

| Input | Required | Used for |
|---|---|---|
| GTFS static (`.zip`: stops, trips, stop_times, shapes) | yes | schedule, stop positions along the shape |
| GTFS-RT **VehiclePositions** whose `trip_id`s match the static feed | yes | vehicle state; also the training signal |
| `trip.start_date` in the VehiclePositions | no | service date; otherwise derived (below) |
| GTFS-RT TripUpdates | no | only a *comparison baseline* in `val`; never a model input |

Time zone comes from `agency.txt` (config as fallback); nothing agency-specific is hard-coded.

## 1. Build a frame and train

Record VehiclePositions for a few weeks (7 days already beat MBTA in the study; the gain stops near
30 days), then:

```bash
stoptimes data frame --live --observations ./archive --static ./gtfs.zip --source myagency \
  --train-days 2026-07-01:2026-10-13 --val-days 2026-10-14:2026-10-20 \
  --rows-per-day 30000 --out datasets/live.parquet
stoptimes train run examples/experiments/xgb-live.yaml --dataset-dir datasets
```

The frame computes arrivals from the pings, builds the features with the same code serving uses, and
writes sidecar assets (segment statistics, route vocabulary) next to the model. The artifact records
its own `timezone` / `holiday_region`; serving and validation use those, not the source's.

## 2. Serve

```bash
stoptimes serve --source myagency --sources-dir ./agencies --registry ./registry \
  --artifact xgb-live --static ./gtfs.zip --sink file:./trip_updates.json --interval 15
```

`serve` builds one **live session** at start-up and keeps it for the whole run: per-vehicle ping
history (in-memory `TripStateStore`), and a trip context per (trip, service date). Every poll it
ingests VehiclePositions, assembles the features at the poll instant, and calls the model **once** for
all vehicles. A vehicle whose newest valid ping is more than 30 s old (the bound used in training) is
skipped. The artifact loads without any override variable; without a static feed or a
`vehicle_positions` feed `serve` refuses with the reason.

Service date: `trip.start_date` when it is a valid `YYYYMMDD`; otherwise, of the agency-local dates
(ping day -1, 0, +1) the one whose scheduled window for the trip is closest to the ping (no fixed
day boundary; a 25:30 trip seen at 00:40 belongs to yesterday).

## 3. Validate against the agency's TripUpdates

Replay recorded days (deterministic) with both predictors over identical trips:

```bash
export STOPTIMES_MODEL_REGISTRY_DIR=./registry
stoptimes val replay --source myagency --sources-dir ./agencies \
  --archive ./archive/vehicle_positions --archive-tu ./archive/trip_updates \
  --gtfs ./gtfs.zip --from 2026-10-21T08:00:00Z --to 2026-10-21T12:00:00Z \
  --predictor gtfsrt --predictor stoptimes:xgb-live --out ./scores.parquet
stoptimes val report ./scores.parquet
```

Live, against the real feeds (the source's `gtfs_static_url` supplies the static feed; agency files are
found through `STOPTIMES_SOURCES`):

```bash
stoptimes val run --source myagency \
  --predictor gtfsrt --predictor stoptimes:xgb-live --duration 1h
```

`stoptimes:xgb-live` keeps its own per-run trip state, fed from the ping stream the engine hands it; it
never reads the validator's ground-truth detectors. Build one predictor per run.

## Caveats

* Cold start: a vehicle first seen mid-trip is predicted from the schedule delay and the path prior;
  `delay_last` stays missing until it crosses a stop.
* Replay speed: `val replay` re-projects each trip's whole history every tick; see the Gate B timing in
  `docs/notes/p4-f4b-serve.md` before replaying more than a few hours at a time.
