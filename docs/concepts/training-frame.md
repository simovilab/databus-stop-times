# The training frame

The training frame is the table a model family actually fits: one row per (vehicle
position, upcoming stop), with the target (arrival delay) attached where a matching
`Arrival` exists. `stoptimes data frame` builds it by joining `Observation`s (usually
vehicle positions), `Arrival`s (the target), and the static feed (schedule and shape
context) — see the columns produced in the [quickstart](../quickstart.md#3-build-the-training-frame).

## Why this is a separate step from "the arrivals"

Arrivals answer "what happened at each stop." A model needs something different: "given
where the vehicle is *right now*, relative to an upcoming stop, how late will it be." That
requires expanding each vehicle-position sample into one row per stop still ahead of it
(bounded by `--max-stops-ahead`), attaching schedule context (distance to stop, stops ahead,
hour of day, and other features from the [feature registry](../guides/train-a-model.md)),
and only then attaching the label. Collapsing this into the arrivals step would mean
re-deriving it slightly differently for every model family; building it once means training
and any future family (or a validator wanting the same features for diagnosis) all read the
same frame.

## Why training never strictly needs GTFS-RT

`data.datasets` (in an `experiment.yaml`) can point at *any* prepared dataset — an AVL CSV
export, an arrivals table someone handed you, a frame built entirely outside `stoptimes`. A
GTFS static feed is only required for the specific things that genuinely need a schedule:
deriving arrivals by map-matching, schedule-based features and baselines, and emitting
GTFS-RT TripUpdates (which are addressed by `trip_id`/`stop_id` from a feed the consumer also
has). See [Contracts, "Does training require GTFS?"](../CONTRACTS.md#does-training-require-gtfs).

## Rows that never got an arrival

`--keep-unlabeled` keeps rows with no matching arrival (target left `NaN`) instead of
dropping them, for cases where you want to inspect coverage or feed the frame to something
other than a supervised regressor. The default drops them, since a family accepting a
`DataFrame`/`ndarray` pair (the [model contract](../CONTRACTS.md#4-model-the-model-interface)) generally can't use an
unlabeled row directly.

## What happens next

`train run experiment.yaml` reads a registered frame by name (`data.datasets: [frame]`),
resolves the `features:` list against the feature registry, splits it (`time`/`trip`/`kfold`
— see [`SplitStrategy`](../CONTRACTS.md#3-experiment-experimentyaml)), fits the chosen model
family plus every baseline, and writes an [artifact](artifacts-and-scope.md) with scores
broken out by [horizon](horizons.md). See [Train a model](../guides/train-a-model.md).
