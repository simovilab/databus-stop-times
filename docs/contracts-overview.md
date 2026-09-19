# Contracts

`stoptimes.core.contracts` is the set of frozen interfaces every part of the toolkit builds
against instead of against each other's internals: pydantic models, `typing.Protocol`s,
enums, and small pure functions — no agency names, no pyarrow/protobuf/xgboost/torch
imports. This is what lets `data`, `train`, `serve`, `record`, and `val` be developed (and
extended by you) independently: a data source, a model family, or a pipeline stage only ever
needs to satisfy a contract, never to know about the others' implementations.

**The full spec lives in [`docs/CONTRACTS.md`](CONTRACTS.md)** — this page orients you to it
rather than duplicating it; the definitive text (including the sign-off history and open
questions) is there, owned separately from this site.

## The seven contracts, and why each exists

| Contract | Answers | Who reads it |
|---|---|---|
| [`source`](CONTRACTS.md#1-source-feed-and-agency-descriptions) | What is this feed, and where does it come from? | readers, recorder, arrivals, val, serve |
| [`data`](CONTRACTS.md#2-data-canonical-data-tables) | What shape are observations and arrivals, exactly? | readers, arrivals, training, recorder, val |
| [`experiment`](CONTRACTS.md#3-experiment-experimentyaml) | What fully describes one trained artifact? | train, the TUI |
| [`model`](CONTRACTS.md#4-model-the-model-interface) | What must a model implement to be trainable/servable? | train, serve, BYO models |
| [`pipeline`](CONTRACTS.md#5-pipeline-serving-stages) | What are the stages of serving, and how do you override one? | serve, val's pipeline predictor |
| [`prediction`](CONTRACTS.md#6-prediction-serving-output) | What does serving output, exactly, and how does it map to GTFS-RT? | serve, val |
| [`artifact`](CONTRACTS.md#7-artifact-model-artifacts-on-disk) | How is a trained model stored on disk? | train, models, serve |
| [`selection`](CONTRACTS.md#7b-selection-which-artifact-serves-a-trip) | Which artifact serves a given trip? | serve, train reports, val |

See [Concepts](concepts/observations.md) for the "why" behind the ideas these contracts
encode (observations, arrivals, the training frame, artifacts and scope, selection,
horizons, baselines), and the [CLI reference](reference/cli.md) for the commands that read
and write them.

## Extending the toolkit through contracts, not forks

- **A custom model family**: implement the `Model` protocol, or wrap something else behind a
  `ModelRef` (`pickle:`, `onnx:`, `callable:`, `http(s)://`) — see
  [Train a model](guides/train-a-model.md).
- **A custom pipeline stage**: override any of `ingest`/`match`/`features`/`predict`/
  `postprocess`/`publish` — see [Serve predictions](guides/serve-predictions.md).
- **A custom data source**: point `data.datasets` at any prepared table satisfying the `data`
  contract's schemas — training never strictly requires GTFS-RT, only a schedule for the
  parts that genuinely need one. See [The training frame](concepts/training-frame.md).

!!! note "A filename note for anyone editing this site"
    This page is named `contracts-overview.md`, not `contracts.md`, because this checkout
    lives on a case-insensitive filesystem (macOS/APFS default) where `contracts.md` and
    `CONTRACTS.md` are the same file on disk — a second file that only differs by case would
    silently overwrite (or be overwritten by) `docs/CONTRACTS.md`, which this track must not
    edit. See `docs/notes/p3-u5.md`.
