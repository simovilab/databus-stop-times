# Artifacts and scope

A trained model lands on disk as an **artifact**: a directory holding `artifact.json`
(metadata — family, target, features, params, data window, provenance, scope, metrics,
scores by horizon), `model.<ext>` (the fitted model), and an optional `card.md` (a
human-readable model card). `stoptimes train run` writes one artifact per experiment;
`stoptimes models list`/`card` read them back.

```
<registry>/<artifact_id>/
  artifact.json
  model.<ext>
  card.md
```

## Why one experiment produces exactly one artifact

An alternative design would let one `experiment.yaml` list several models and sweep them.
The toolkit deliberately keeps it to one model per experiment file, so "what produced this
specific artifact" is never ambiguous — every artifact traces to exactly one experiment.
Comparing several families over the same dataset becomes a loop over several experiment
files (or a future `study` file that expands into them), not a wider single file. See
[Contracts, open question 2](../CONTRACTS.md#open-questions-for-sign-off).

## Why provenance is a first-class field, not an afterthought

A user story explicitly asks for artifacts that "carry enough provenance to identify [them]
six months later." `artifact.json`'s `provenance` and `data` (a `DataWindow`) fields exist so
an artifact answers, on its own, what data window it trained on, what library versions built
it, and what git state produced it — without anyone needing to remember or reconstruct
context later. `version_mismatches` compares an artifact's recorded library versions against
the environment trying to load it now.

## Scope: what an artifact is allowed to serve

A registry usually holds several artifacts for one agency — a global model, per-route
models, sometimes per-trip ones. `ArtifactMetadata.scope` (a `ModelScope`) says what an
artifact may serve:

```
ModelScope
  level: global | route | direction | trip
  source_id: str | None      # None = serves any agency
  route_ids, direction_id, trip_ids
```

This exists because ranking-by-metric alone has no agency awareness — it's exactly why
bUCR's production deployment needed a hard-coded pin before this contract existed. Scope
turns "which artifacts are even eligible for this trip" into a declared, checkable fact
instead of an implicit assumption baked into serving code. See [Selection
policy](selection-policy.md) for how an artifact is actually chosen among the eligible ones,
and [Model lifecycle](../guides/train-a-model.md#registry-lifecycle-thin-pending-u7) for the
part of this still landing in track U7.

Full shape: [Contracts §7](../CONTRACTS.md#7-artifact-model-artifacts-on-disk).
