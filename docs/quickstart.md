# Quickstart: I have a GTFS zip and a realtime URL

This page takes you from nothing to a scored prediction using only `stoptimes` commands.
Every command below was actually run while writing this page, in a clean checkout, and the
output shown is real (lightly trimmed for length) — not representative or made up.

The chain is the same one `tests/e2e/test_end_to_end.py` checks on every commit:

```
data add  →  data arrivals  →  data frame  →  train run  →  serve  →  val replay
```

## Install

```bash
pip install 'stoptimes[core,record,train,serve,val]'
```

Install only the parts you need — `pip install 'stoptimes[serve]'` alone never pulls in
training dependencies (xgboost, torch). Every command below prints an install hint instead
of a traceback if you're missing a part.

## 0. Try it in under a minute (no network, no real agency needed)

You don't need a live feed to see the whole toolkit work once. This step builds a tiny
four-stop, four-trip GTFS feed and a matching "recorded" vehicle-position archive — the same
kind of synthetic fixture the project's own end-to-end test uses — entirely on your machine.
Save this as `make_fixtures.py`:

```python
import zipfile
from pathlib import Path
import pandas as pd

SOURCE, SERVICE_DAY = "demo", "20260917"

def build_static_zip(path):
    stops = ["stop_id,stop_name,stop_lat,stop_lon"]
    stops += [f"s{i+1},Stop {i+1},0.0,{i*0.01:.5f}" for i in range(4)]
    trips = ["route_id,service_id,trip_id,direction_id,shape_id"]
    stop_times = ["trip_id,arrival_time,departure_time,stop_id,stop_sequence,shape_dist_traveled"]
    for t in range(4):
        trip_id = f"t{t+1}"
        trips.append(f"r1,svc,{trip_id},0,sh1")
        for i in range(4):
            clock = f"{8+t:02d}:{i*2:02d}:00"
            stop_times.append(f"{trip_id},{clock},{clock},s{i+1},{i+1},{i*1113}")
    shapes = ["shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence,shape_dist_traveled"]
    shapes += [f"sh1,0.0,{i*0.01:.5f},{i+1},{i*1113}" for i in range(4)]
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("stops.txt", "\n".join(stops) + "\n")
        zf.writestr("trips.txt", "\n".join(trips) + "\n")
        zf.writestr("stop_times.txt", "\n".join(stop_times) + "\n")
        zf.writestr("shapes.txt", "\n".join(shapes) + "\n")
        zf.writestr("routes.txt", "route_id,route_short_name,route_long_name\nr1,1,Line One\n")

def build_archive(root):
    rows = []
    for t in range(4):
        start = pd.Timestamp(f"2026-09-17T{8+t:02d}:00:00Z")
        for step in range(25):
            rows.append({
                "source_id": SOURCE, "kind": "vehicle_position", "trip_id": f"t{t+1}",
                "route_id": "r1", "vehicle_id": f"v{t+1}", "start_date": SERVICE_DAY,
                "ts": start + pd.Timedelta(seconds=15 * step),
                "lat": 0.0, "lon": round(step * 0.00125, 6), "bearing": 90.0,
            })
    partition = root / "year=2026" / "month=09" / "day=17" / "route_id=r1"
    partition.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(partition / "part-0.parquet")

def build_agency_yaml(directory):
    directory.mkdir(exist_ok=True, parents=True)
    (directory / f"{SOURCE}.yaml").write_text(
        f"name: {SOURCE}\ngtfs_static_url: gtfs.zip\n"
        "vehicle_positions_url: https://example.invalid/vp.json\n"
        "trip_updates_url: https://example.invalid/tu.json\n"
        "poll_interval_s: 15\ntimezone: UTC\n"
        "arrival_radius_m: 60.0\ndwell_s: 0.0\nshape_threshold_m: 50.0\n",
    )

if __name__ == "__main__":
    here = Path(__file__).parent
    build_static_zip(here / "gtfs.zip")
    build_archive(here / "archive")
    build_agency_yaml(here / "agencies")
    print("fixtures written")
```

```console
$ python make_fixtures.py
fixtures written
```

This gives you `gtfs.zip` (a real, tiny GTFS static feed), `archive/` (vehicle positions in
the same partitioned Parquet layout `stoptimes record` writes), and `agencies/demo.yaml` (a
[`Source`](concepts/observations.md) description — see [Contracts §1](CONTRACTS.md#1-source-feed-and-agency-descriptions)).

If you already have a real GTFS zip and a recorded/exported vehicle-position table, skip to
step 2 — everything from here on is identical either way.

## 1. Register the recorded positions

```console
$ stoptimes data add positions ./archive --source demo --kind observations
✓ added dataset 'positions' (observations, 100 rows)

$ stoptimes data list
                               stoptimes datasets
┏━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ name      ┃ source ┃ kind         ┃ rows ┃ date range      ┃ location        ┃
┡━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ positions │ demo   │ observations │ 100  │ 2026-09-17 ...  │ .../archive     │
└───────────┴────────┴──────────────┴──────┴─────────────────┴─────────────────┘
```

`data add` works the same way for a live feed URL, an `s3://` archive, or a mapped AVL CSV —
see [Record a feed](guides/record-a-feed.md).

## 2. Turn positions into ground-truth arrivals

Map-matches vehicle positions against the static feed to produce one
[`Arrival`](concepts/arrivals.md) row per (trip, service day, stop):

```console
$ stoptimes data arrivals \
    --observations positions --static ./gtfs.zip --source demo \
    --sources-dir ./agencies --out ./arrivals.parquet --register arrivals --json
{
  "source": "demo", "observations": 100, "arrivals": 16,
  "by_method": {"map_matched": 16},
  "static_tables": ["stops", "trips", "stop_times", "shapes", "routes"],
  "out": ".../arrivals.parquet", "registered": "arrivals"
}
```

## 3. Build the training frame

Joins observations, arrivals, and static schedule into one row per (vehicle position,
upcoming stop) — see [The training frame](concepts/training-frame.md):

```console
$ stoptimes data frame \
    --observations positions --arrivals arrivals --static ./gtfs.zip --source demo \
    --sources-dir ./agencies --out ./frame.parquet --register frame --json
{
  "source": "demo", "observations": 100, "arrivals": 16,
  "rows": 252, "labeled_rows": 252,
  "static_tables": ["stops", "trips", "stop_times", "shapes", "routes"],
  "out": ".../frame.parquet", "registered": "frame"
}
```

## 4. Train a model

Write an `experiment.yaml` (the [`experiment` contract](CONTRACTS.md#3-experiment-experimentyaml)):

```yaml
name: quickstart-historical-mean
data:
  datasets: [frame]
target: arrival_delay_s
model:
  family: historical_mean
  params: {}
features: [distance_to_stop, stops_ahead, hour]
split:
  kind: time
  train_fraction: 0.7
  test_fraction: 0.3
seed: 7
outputs:
  registry_dir: ./registry
  tags: [quickstart]
baselines: []
```

```console
$ stoptimes train run experiment.yaml --json
{"artifact_id": "quickstart-historical-mean", "family": "historical_mean",
 "metrics": {"test_mae_seconds": 81.04, "test_mae_minutes": 1.35,
 "test_rmse_seconds": 95.82, "test_r2": -0.0, "test_within_120s": 0.79, ...}}

$ stoptimes models list --registry ./registry
                                    models: ./registry
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ artifact id         ┃ family          ┃ scope  ┃ mae_s ┃ created             ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ quickstart-histori… │ historical_mean │ global │ 81.0  │ 2026-09-19T04:52:… │
└─────────────────────┴─────────────────┴────────┴───────┴─────────────────────┘
```

`train run` also writes `card.md` next to `artifact.json` — a human-readable model card
(data window, features, scores by horizon). See `stoptimes models card
quickstart-historical-mean --registry ./registry`.

A toy 100-row synthetic feed trains in about a second; this is a `historical_mean` baseline
family on purpose — it's fast and needs no tuning, which keeps this page honest about what
was actually run. See [Train a model](guides/train-a-model.md) for real feature/family
choices, and [Baselines](concepts/baselines.md) for why a baseline ships with every report.

## 5. Serve predictions from that model

Serving needs two things beyond the model: a way to get observations in (`ingest`) and a
sink to publish to. This example overrides `ingest` to read one JSON snapshot instead of
polling a live URL, and fills in two feature columns a real deployment's `match` stage
(map-matching against the static feed) would normally compute — see the note in
[Serve predictions](guides/serve-predictions.md) about why that's still a manual step today.

```python
# my_pipeline.py
import json
from pathlib import Path
import pandas as pd
from stoptimes.core.contracts.data import Observation, ObservationKind
from stoptimes.serve.pipeline import build_default_pipeline, default_features

pipeline = build_default_pipeline()

@pipeline.stage("ingest")
def ingest_from_file(_payload, ctx):
    feed = json.loads(Path("incoming_tu.json").read_text())
    observations = []
    for entity in feed["entity"]:
        tu = entity["trip_update"]
        trip = tu["trip"]
        for stu in tu["stop_time_update"]:
            observations.append(Observation(
                kind=ObservationKind.TRIP_UPDATE, source_id=ctx.source.id,
                trip_id=trip["trip_id"], route_id=trip.get("route_id"),
                direction_id=trip.get("direction_id"), start_date=trip.get("start_date"),
                vehicle_id=(tu.get("vehicle") or {}).get("id"),
                ts=pd.Timestamp(tu["timestamp"], unit="s", tz="UTC"),
                stop_id=stu["stop_id"], stop_sequence=stu["stop_sequence"],
                arrival_time=pd.Timestamp(stu["arrival"]["time"], unit="s", tz="UTC"),
            ))
    return observations

@pipeline.stage("features")
def features_with_distance(matched, ctx):
    frame = default_features(matched, ctx)
    frame["distance_to_stop"] = 300.0
    frame["stops_ahead"] = frame["stop_sequence"].astype(float)
    return frame
```

```console
$ stoptimes serve \
    --source demo --sources-dir ./agencies --registry ./registry \
    --model quickstart-historical-mean --pipeline ./my_pipeline.py:pipeline \
    --sink file:./trip_updates.json --format json --once --json
{"source": "demo", "artifact_id": "quickstart-historical-mean",
 "generated_at": "2026-09-19T04:53:22Z", "trip_update_count": 4,
 "errors": [], "published": true}
```

`trip_updates.json` is now a spec-valid GTFS-RT TripUpdates feed with your model's
predictions. `--once` produces one feed and exits (used above and by the project's own gate
test); drop it and add `--interval 15` to run continuously like a real publisher.

## 6. Score it against what actually happened

```console
$ stoptimes val replay \
    --source demo --sources-dir ./agencies --archive ./archive --gtfs ./gtfs.zip \
    --from 2026-09-17 --to 2026-09-18 --predictor baseline:schedule --out ./scores.parquet
Replaying: source='demo', archive='./archive', ..., predictors=['baseline:schedule'], ...
Done. Ticks=100 Samples=176 Arrivals=32 Results=176
Exported 176 results → ./scores.parquet

$ stoptimes val report ./scores.parquet
                         Per-Predictor Metrics
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┓
┃ Predictor         ┃ MAE (s) ┃ RMSE (s) ┃ MAPE (%) ┃ Bias (s) ┃ Count ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━┩
│ baseline:schedule │     0.0 │      0.0 │      0.0 │      0.0 │   176 │
└───────────────────┴─────────┴──────────┴──────────┴──────────┴───────┘
```

(`baseline:schedule` scores 0 error here only because this toy feed's synthetic ground truth
was built to exactly match the schedule — a real feed will show real error. Point `val
replay` at your trained artifact with `--predictor stoptimes:quickstart-historical-mean` to
score your own model the same way; see [Validate any feed](guides/validate-a-feed.md).)

**Note on first-run speed:** the first invocation of any `stoptimes val <command>` in a
fresh interpreter pulls in a heavy import chain (scikit-learn, matplotlib) and can take
20-60 seconds before it prints anything — this is a real, known cost (see
`docs/notes/p3-u2.md`'s follow-up), not a hang. `stoptimes val --help` itself is fast; only
running a real subcommand pays the cost.

## Using your own feed instead of the synthetic one

Everything above works identically with real data — swap in your own GTFS zip's path/URL
for `--static`, and instead of the pre-built `archive/`, record one:

```bash
stoptimes record start --source myagency --sources-dir ./agencies --duration 3600 \
  --status-dir ./status
```

polls `myagency.yaml`'s configured feeds for an hour and writes to its configured archive.
See [Record a feed](guides/record-a-feed.md) for the full agency-file shape and for running
it unattended for weeks. Everything from step 1 onward is exactly the commands above, with
your own `--source`/paths in place of `demo`/`./archive`/`./gtfs.zip`.

## What just happened

You went from raw vehicle positions and a GTFS zip to a trained model, a served GTFS-RT
feed, and a validation score — the same six commands whether the model is a one-line
baseline or a tuned model family, and whether the feed comes from a toy fixture or a real
agency. Next: [Concepts](concepts/observations.md) for why each piece exists, or the
[guides](guides/record-a-feed.md) for the job you actually have.
