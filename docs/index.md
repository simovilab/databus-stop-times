# stoptimes

**stoptimes** is a toolkit for arrival-time prediction on GTFS data: record GTFS-Realtime
feeds, train models on them, serve GTFS-RT TripUpdates, and validate any prediction
source against what actually happened.

It is built to work for anyone with GTFS data — a transit agency, a researcher, a hobbyist
with a feed URL. [SIMOVI](case-study/simovi.md) (bUCR and MBTA) is the first user, not the
design target: nothing in the packages is agency-specific, and bUCR/MBTA live entirely as
configuration.

## Where to start

- **New here?** [Quickstart](quickstart.md) takes a GTFS zip and a realtime feed to a scored
  prediction using only copy-pasteable commands, end to end.
- **Want the mental model first?** [Concepts](concepts/observations.md) explains the handful
  of ideas (observations, arrivals, the training frame, artifacts, selection, horizons,
  baselines) that everything else is built from.
- **Have a specific job?** The [guides](guides/record-a-feed.md) are organized by what you're
  trying to do: record a feed, diagnose it, train a model, serve predictions, or validate any
  feed.
- **Need exact flags?** The [CLI reference](reference/cli.md) is generated from the `stoptimes`
  command itself, so it can't drift from what `--help` prints.
- **Wiring it into your own system?** [Contracts](contracts-overview.md) explains the frozen interfaces
  (`stoptimes.core.contracts`) that let you plug in a model, a pipeline stage, or a data
  source without forking anything, and links to the full spec.

## The parts

| Part | What it gets you | Package |
|---|---|---|
| Data | Load GTFS static + realtime data in common forms; inspect coverage | `stoptimes-core` |
| Record | Archive a live GTFS-RT feed, compacted, local or S3 | `stoptimes-record` |
| Diagnose | A feed-quality report before you invest in training | `stoptimes-core` |
| Train | Describe an experiment in one file; get a versioned artifact + model card | `stoptimes-train`, `stoptimes-models` |
| Serve | A prediction library, plus a standalone publisher for agencies without one — see [What "serving" means](concepts/selection-policy.md#two-things-called-serving) | `stoptimes-serve` |
| Validate | Score any TripUpdates source against what actually happened | `stoptimes-val` |

Install only what you need: `pip install 'stoptimes[serve]'` never pulls training
dependencies. One command ties it together: `stoptimes {data,record,train,models,serve,val,diagnose}`.

## Status

This site documents Phase 3 of the toolkit's build-out. Two pages are deliberately thin
right now and marked as such: [Serve predictions](guides/serve-predictions.md) (serving is
being rebuilt on the pipeline, with real map-matching, in track U6) and the model-lifecycle
parts of [Train a model](guides/train-a-model.md) (registry lifecycle — refusing stale
artifacts, promote/rollback — is being finished in track U7). Everything else on this site
was run in this checkout while writing it; see the [CLI reference](reference/cli.md) for
what's generated versus hand-written.
