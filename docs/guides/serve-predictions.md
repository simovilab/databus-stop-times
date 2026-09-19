# Guide: serve predictions

Read [Selection policy: two things called "serving"](../concepts/selection-policy.md#two-things-called-serving)
first. This guide covers the standalone publisher (`stoptimes serve`). If you already have
your own feed builder — the way SIMOVI's `databus` does — you want the **prediction
library** underneath it instead: call `stoptimes.serve.pipeline` (or
`stoptimes.serve.estimator.estimate_stop_times`, the signature databus calls today) directly
from your own polling/publishing loop, and skip this command entirely. Everything below,
other than the `--sink`/`--interval`/`--format` publishing flags, is exactly what that
library does under the hood — reading this page still tells you how matching, features, and
model selection work either way.

## The standalone publisher

```bash
stoptimes serve \
  --source myagency --sources-dir ./agencies --registry ./registry \
  --model <artifact_id> --static ./gtfs.zip \
  --sink file:./trip_updates.json --format json --once --json
```

Verified end to end in the [quickstart](../quickstart.md#5-serve-predictions-from-that-model):
poll (or, with `--once`, produce a single feed and exit) → match → run the model → publish.
`--sink` accepts `file:PATH`, `http(s)://...`, or `mqtt://host/topic`; `--format pb` emits
protobuf instead of JSON. `--interval N` loops indefinitely instead of running once, which is
the shape for actually running this as a small agency's publisher rather than a one-shot
demo.

`--model` accepts a registry artifact id, or a BYO spec — see below.

## Map-matching: `--static`

`--static` points the default `match` stage at a GTFS static feed (`.zip` or URL); it
defaults to the source's own `gtfs_static_url` if you omit it, so most of the time you don't
need to pass it explicitly at all. With geometry available, `match` projects a vehicle
position onto its trip's shape and emits the real upcoming stops with a genuine
along-shape `distance_to_stop` and 0-indexed `stops_ahead`; a TripUpdate observation (which
already names a stop) gets the same two features attached from the shape when geometry is
available. Verified by running the quickstart's pipeline (below) with only `ingest`
overridden and no manual feature columns — `distance_to_stop`/`stops_ahead` came from
`match` automatically:

```bash
stoptimes serve --source demo --sources-dir ./agencies --registry ./registry \
  --model quickstart-historical-mean --static ./gtfs.zip \
  --pipeline ./my_pipeline.py:pipeline --sink file:./trip_updates.json --once --json
```

```json
{"source": "demo", "artifact_id": "quickstart-historical-mean", "generated_at": "...", "trip_update_count": 4, "errors": [], "published": true}
```

(`my_pipeline.py` here only overrides `ingest`, reading one JSON snapshot instead of polling
— no `features` override needed anymore, unlike the pre-U6 version of this page.)

If the static feed can't be read at all — a bad path, a URL that doesn't resolve — `serve`
warns loudly on stderr rather than degrading silently, and continues without map-matching:

```text
⚠ could not read the static feed 'gtfs.zip' ([Errno 2] No such file or directory: 'gtfs.zip');
serving without map-matching — only stop-bearing feeds (TripUpdates) will produce predictions
```

Without geometry, a `vehicle_position` observation can't be turned into "which stop, how
far" at all (it's recorded as a per-item `match` error, not a crash); a `trip_update`
observation still serves, just without the two spatial features, since it already names its
stop.

## Overriding a pipeline stage {#bring-your-own-model}

Serving runs through six stages — `ingest → match → features → predict → postprocess →
publish` (see [Contracts §5](../CONTRACTS.md#5-pipeline-serving-stages)) — and `--pipeline
module:attr` or `--pipeline path/to/file.py:attr` lets you override any subset of them
without rewriting the server, or the other five. The quickstart's `my_pipeline.py` overrides
only `ingest` (read a JSON snapshot instead of polling); `match`, `features`, `predict`,
`postprocess`, and `publish` all run unmodified from
`stoptimes.serve.pipeline.build_default_pipeline()`.

## Bring your own model

`--model pickle:PATH`, `--model onnx:PATH`, `--model callable:module:function`, or `--model
http(s)://URL` load a model that never went through `stoptimes train` at all, through the
same [`ModelRef`](../CONTRACTS.md#4-model-the-model-interface)/`ModelLoader` machinery as a
registry artifact. `stoptimes.core` itself never imports `pickle`-unsafe code, `onnxruntime`,
or an HTTP client for this — only the matching loader inside `serve` does, which is why
`stoptimes[serve]` installs without pulling training dependencies.

A registry artifact loaded this way is also subject to the version-drift refusal described in
[Train a model: registry lifecycle](train-a-model.md#registry-lifecycle) — a stale artifact
is refused, not silently deserialized, before serving ever sees it.

## What this means for databus, concretely

`databus` owns its own vehicle feed and its own Redis `stop_time_updates` projection, so it
never runs the command above. What it calls instead is the same matching/feature/predict
logic through `stoptimes.serve.estimator.estimate_stop_times` (the pre-pipeline entry point,
rebuilt on the pipeline internals in U6 with its public signature unchanged) or, going
forward, a `Pipeline` with its own `ingest` (its own vehicles, no polling) and `publish`
(write the Redis projection) stages, inheriting `match`/`features`/`predict`/`postprocess`
from the default pipeline documented on this page — see
[the SIMOVI case study](../case-study/simovi.md) and `docs/PLAN.md` §8 G2b. The publisher
above and databus's integration are two different callers of the identical prediction code;
neither is a special case of the other.
