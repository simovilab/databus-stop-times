# Guide: train a model

For the modeler user stories: "describe an experiment in one file and reproduce it later,"
"compare families over one dataset with errors broken out by horizon," "plug in a custom
model without forking."

## Describe the experiment

An `experiment.yaml` is the whole description of one trained artifact — see
[Artifacts and scope](../concepts/artifacts-and-scope.md) for why it's one model per file,
not a sweep:

```yaml
name: my-experiment
data:
  datasets: [frame]        # a name registered with `data add`/`data frame`, or sources+dates
target: arrival_delay_s
model:
  family: xgb               # free-form string, resolved against the model registry
  params:
    max_depth: 5
features: [distance_to_stop, stops_ahead, hour]
split:
  kind: time
  train_fraction: 0.7
  test_fraction: 0.3
seed: 42
outputs:
  registry_dir: ./registry
  tags: [my-tag]
```

`stoptimes train run experiment.yaml --dry-run` prints the fully resolved plan (verified
against the [quickstart](../quickstart.md#4-train-a-model)'s example) without touching data —
use it to sanity-check paths and split fractions before a real run.

## Hyperparameter search, not just a fixed value

`ParamValue` accepts a plain scalar, or a `ParamRange` (low/high/step/log) or `ParamChoices`
(a discrete list) for tuning — see [Contracts §3](../CONTRACTS.md#3-experiment-experimentyaml).
Building a range/choices value today means writing it into the YAML directly or through
`stoptimes.train.experiment`'s Python helpers (`make_param_range`/`make_param_choices`); the
TUI's compact hyperparameter field (below) only accepts scalars as of this writing.

## Build one interactively instead

```bash
stoptimes train ui
```

opens a Textual builder: pick a target, model family (introspected from the real model
classes' constructors, so its listed parameters and defaults can't drift from what actually
trains), features (from the [feature registry](../concepts/training-frame.md), with
descriptions), and split strategy, and it writes an `experiment.yaml` — never anything the
YAML itself can't express, per the project's "UIs write config; the CLI runs config" rule.
Needs the `textual` extra installed; `train ui --help` gives an install hint if it isn't.

## Compare families

Train several experiments (one file each, per the one-model-per-experiment design) against
the same `data.datasets`, then compare their `scores_by_horizon` via `stoptimes models card
<id>` for each, or `stoptimes val report` if you've scored them all through validation — see
[Horizons](../concepts/horizons.md) for why the comparison is broken out by horizon rather
than one blended number.

## Plug in your own model

A family doesn't have to be one of the built-in ones. A
[`ModelRef`](../CONTRACTS.md#4-model-the-model-interface) — `pickle:PATH`,
`onnx:PATH`, `callable:module:function`, or an `http(s)://` URL — lets you bring a model that
was never trained through `stoptimes train` at all and still serve it through the same
selection and pipeline machinery. See [Serve predictions](serve-predictions.md#bring-your-own-model).

## List, inspect, and promote trained artifacts

```bash
stoptimes models list --registry ./registry
stoptimes models card <artifact_id> --registry ./registry
```

Both were run against the quickstart's artifact and produce the table/card shown on that
page. `models list --json` and `models card --json` give the same information as a flat
dict/list.

### Registry lifecycle — thin, pending U7

`stoptimes models promote <artifact_id> --as <pin> --source <agency>` pins an artifact into
an agency's `serving.selection.pins`, and `stoptimes models migrate` brings a legacy
`gtfs-eta` registry into this layout. Both commands exist and run today, but the surrounding
lifecycle work — refusing to load an artifact whose library versions have drifted (
`version_mismatches` is computed already; nothing acts on it yet) and a tested
promote/rollback workflow — is the subject of track U7, landing separately from this docs
pass. This section will be filled in with the finished command-level behavior once U7 lands;
until then, treat `promote` as "pins an id into a YAML file" and nothing more.
