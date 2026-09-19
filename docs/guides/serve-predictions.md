# Guide: serve predictions

Read [Selection policy: two things called "serving"](../concepts/selection-policy.md#two-things-called-serving)
first — this guide covers the standalone publisher; if you already have your own feed
builder and just need predictions, you want the prediction library, not this command.

## The standalone publisher

```bash
stoptimes serve \
  --source myagency --sources-dir ./agencies --registry ./registry \
  --model <artifact_id> \
  --sink file:./trip_updates.json --format json --once --json
```

Verified end to end in the [quickstart](../quickstart.md#5-serve-predictions-from-that-model):
poll (or, with `--once`, produce a single feed and exit) → run the model → publish. `--sink`
accepts `file:PATH`, `http(s)://...`, or `mqtt://host/topic`; `--format pb` emits protobuf
instead of JSON. `--interval N` loops indefinitely instead of running once, which is the
shape for actually running this as a small agency's publisher rather than a one-shot demo.

`--model` accepts a registry artifact id, or a BYO spec — see below.

## Overriding a pipeline stage {#bring-your-own-model}

Serving runs through six stages — `ingest → match → features → predict → postprocess →
publish` (see [Contracts §5](../CONTRACTS.md#5-pipeline-serving-stages)) — and `--pipeline
module:attr` or `--pipeline path/to/file.py:attr` lets you override any subset of them
without rewriting the server. The quickstart's `my_pipeline.py` overrides `ingest` (read a
JSON snapshot instead of polling) and `features` (supply two columns by hand); everything
else — `match`, `predict`, `postprocess`, `publish` — runs unmodified from
`stoptimes.serve.pipeline.build_default_pipeline()`.

## Bring your own model

`--model pickle:PATH`, `--model onnx:PATH`, `--model callable:module:function`, or `--model
http(s)://URL` load a model that never went through `stoptimes train` at all, through the
same [`ModelRef`](../CONTRACTS.md#4-model-the-model-interface)/`ModelLoader` machinery as a
registry artifact. `stoptimes.core` itself never imports `pickle`-unsafe code, `onnxruntime`,
or an HTTP client for this — only the matching loader inside `serve` does, which is why
`stoptimes[serve]` installs without pulling training dependencies.

## What's thin here, and why

`serve.estimator` — the code path databus calls in production today — predates the pipeline
contract above and hasn't been rebuilt on it yet; that rebuild, plus real vehicle-position
map-matching in the default `match` stage (via `core.arrivals`/`core.segments`, instead of
the hand-supplied distance features this page's quickstart example still needs), is track
**U6**, running in parallel with this docs pass. Once U6 lands:

- the `match` stage will do real map-matching by default, and the quickstart's manual
  `distance_to_stop`/`stops_ahead` override should no longer be necessary for a pipeline that
  keeps the default `match` stage;
- per-route/per-trip [selection](../concepts/selection-policy.md) behavior in the default
  `predict` stage may have more to say than this page currently does.

This page will be revised once U6's actual behavior is confirmed rather than guessed at now.
Everything above this section — the CLI flags, `--pipeline`, `--model`, and the BYO loaders —
is current contract, not expected to change.
