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

### Registry lifecycle

Two separate questions, both answered here for an operator who needs to know what happens at
2am: *will a stale model load?* and *how do I move a pin forward or back?* Full spec:
[Contracts §7](../CONTRACTS.md#loading-refuses-on-load-critical-version-drift-u7) (loading)
and [§7c](../CONTRACTS.md#7c-promote-rollback-u7) (promote/rollback) — this section is the
operator-facing walkthrough, not a restatement.

#### A stale artifact is refused, not silently loaded

If a library that actually deserializes the model (`sklearn`/`xgboost` for a pickle artifact,
`onnxruntime` for onnx, `torch` for torchscript — not every tracked library; a numpy or
pandas point-release is only ever a warning) has drifted between when an artifact was built
and what's installed on the host trying to load it now, loading **refuses**:

```console
$ stoptimes serve --source demo --sources-dir ./agencies --registry ./registry \
    --model quickstart-historical-mean --pipeline ./my_pipeline.py:pipeline \
    --sink file:./trip_updates.json --once --json
...
ArtifactVersionDriftError: artifact 'quickstart-historical-mean' was built with different
library versions than are installed here, for a library that deserializes the model:
xgboost: built with 1.0.0, installed 3.4.1. Re-export the artifact under the versions
installed on this host (or the deploy target's versions, if this is a deploy), or, if you
know this artifact is still good, override with STOPTIMES_ALLOW_VERSION_DRIFT=all (or a
comma-separated list of artifact ids including 'quickstart-historical-mean') -- this is
logged loudly, never the default.
```

(reproduced against a copy of the quickstart's artifact with its recorded `xgboost` version
edited to force a drift — the real message and exit code `stoptimes serve` produces today).
**As shipped, this surfaces as an uncaught Python traceback from `stoptimes serve`/anything
else calling `load_artifact_model`, not a clean one-line CLI error** — `serve.py` only
catches `NotImplementedError` around this call today, not `ArtifactVersionDriftError`. At
2am the signal to look for is `ArtifactVersionDriftError` near the bottom of the traceback,
not a `✗ ...` message.

An artifact with no recorded library versions at all (anything pre-Phase-1) always loads —
absence of provenance is never treated as evidence of drift.

**The override**, when you know the artifact is still good: `STOPTIMES_ALLOW_VERSION_DRIFT`
as an environment variable — `all`/`1`/`true`/`*` for every artifact, or a comma-separated
list of specific artifact ids. It is never a config default and never silent:

```console
$ STOPTIMES_ALLOW_VERSION_DRIFT=all stoptimes serve ... --once --json
2026-09-19 05:25:34 [W] [model_loading] OVERRIDE: loading artifact despite load-critical
library drift | artifact_id=quickstart-historical-mean drift=xgboost: built with 1.0.0,
installed 3.4.1
{"source": "demo", "artifact_id": "quickstart-historical-mean", ..., "published": true}
```

both lines verified for real against a deliberately drifted copy of the quickstart's
artifact. A caller embedding this in code (an offline re-scoring tool, say) can pass
`load_artifact_model(meta, registry, allow_version_drift=True)` instead of setting the
environment variable.

#### Promote, and get back if you need to

```bash
stoptimes models promote <artifact_id> --as global --source myagency --registry ./registry
```

pins an artifact into the agency's `serving.selection.pins` (the same pins
[selection](../concepts/selection-policy.md) reads at serve time) and appends to a sidecar
history file next to the agency's profile (`<source>.promotions.json`) — this history is
what makes rollback possible. Verified against the quickstart's registry:

```console
$ stoptimes models promote quickstart-historical-mean --as global --source demo \
    --registry ./registry --sources-dir ./agencies --json
{"artifact_id": "quickstart-historical-mean", "pin": "global", "source": "demo",
 "profile_path": ".../agencies/demo.yaml", "history_path": ".../agencies/demo.promotions.json",
 "promotions_for_pin": 1}
```

`stoptimes models pins --source myagency` shows what's live right now, and how many
promotions are on record per pin (so you can tell at a glance whether `rollback` has
somewhere to go):

```console
$ stoptimes models pins --source demo --sources-dir ./agencies --json
[{"pin": "global", "artifact_id": "quickstart-historical-mean", "promotions": 1}]
```

`stoptimes models rollback --as global --source myagency` restores the pin to whatever was
live immediately before the *most recent* promotion. Run with only one promotion on record
(nothing came before it), it refuses cleanly rather than guessing or clearing the pin:

```console
$ stoptimes models rollback --as global --source demo --registry ./registry --sources-dir ./agencies --json
✗ no earlier promotion recorded for pin 'global' in .../agencies/demo.promotions.json
  (1 entry) -- nothing to roll back to
```

Promoted a second artifact, `rollback` correctly restored the first:

```console
$ stoptimes models promote quickstart-historical-mean-v2 --as global --source demo --registry ./registry --sources-dir ./agencies --json
{"artifact_id": "quickstart-historical-mean-v2", "pin": "global", ..., "promotions_for_pin": 2}
$ stoptimes models rollback --as global --source demo --registry ./registry --sources-dir ./agencies --json
{"pin": "global", "source": "demo", "restored_artifact_id": "quickstart-historical-mean", ...}
```

If the artifact `rollback` restores has since been deleted from the registry, it still
restores the pointer and warns on stderr rather than refusing — a pin naming a missing
artifact degrades to metric-ranked selection at serve time (the same rule
[selection](../concepts/selection-policy.md) always follows), so this is a safe outcome, not
a broken one. There is no separate "redo": promoting again after a rollback is how you move
forward.

`stoptimes models migrate <legacy_dir> <registry_dir>` brings a legacy `gtfs-eta`
`{key}.pkl`/`{key}_meta.json` registry into this layout in one pass; unaffected by any of the
above.
