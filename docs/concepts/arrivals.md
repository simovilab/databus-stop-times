# Arrivals

An `Arrival` is one actual stop arrival — the ground truth everything else is measured
against. It has a `method` (`reported`, `map_matched`, `interpolated`, or
`schedule_fallback` — never used as ground truth) and, for derived methods, a `confidence`.

## Why this exists as its own thing, not just "a TripUpdate you trust"

A published TripUpdate is a *prediction*, even the operator's own. Training a model on
predictions instead of on what actually happened would let a model learn to reproduce
existing bias rather than correct it, and it would make validation circular (scoring a
predictor against another predictor's opinion). `Arrival` is deliberately the one place in
the system that means "this really happened," derived independently by map-matching vehicle
positions against the static schedule (`stoptimes data arrivals`) rather than read off a
realtime prediction feed.

## Why keyed by stop *sequence*, not stop ID

```
Arrival: source_id, trip_id, start_date, route_id, vehicle_id, stop_id, stop_sequence, ...
```

A loop route visits the same `stop_id` twice in one trip. Keying by `(trip_id, start_date,
stop_id)` would silently merge two real, independent arrivals into one. Keying by
`stop_sequence` instead — which is what Phase 2 actually implemented, correcting an earlier
draft of [Contracts §2](../CONTRACTS.md#arrival) that said "trip+stop+day" — keeps both
visits as separate evidence.

## How derivation works, briefly

`stoptimes data arrivals` map-matches each vehicle-position observation against the static
feed's shape and stop sequence, using the `Source`'s
[`GroundTruthDetector`](../CONTRACTS.md#1-source-feed-and-agency-descriptions) thresholds
(`arrival_radius_m`, `dwell_s`, `shape_threshold_m`) to decide when a vehicle counts as
"arrived" at a stop. This is the same map-matching machinery
(`stoptimes.core.arrivals`/`stoptimes.core.segments`) that a bUCR-style trip-less raw AVL
feed needs a trip-inference step in front of — see the [SIMOVI case study](../case-study/simovi.md).

## Where arrivals are used

- **Training** — the target column in the [training frame](training-frame.md) comes from
  joining observations to their nearest `Arrival`.
- **Validation** — `val replay` and `val run` score any predictor's output against archived
  `Arrival`s, never against another predictor's output (this is what makes "score *any*
  TripUpdates feed against what actually happened," one of the [user
  stories](../PLAN.md#71-user-stories-the-acceptance-list), possible without access to how a
  feed was produced).

Full field list: [Contracts §2](../CONTRACTS.md#arrival).
