# Case study: SIMOVI (bUCR and MBTA)

SIMOVI is the first user of this toolkit, not its design target — see
[`docs/PLAN.md` §1, rule 4](../PLAN.md#1-product-in-one-page): "no agency-specific code in
`packages/`. bUCR and MBTA live in `examples/` as configs." Everything below is
configuration; nothing here describes internal infrastructure, hostnames, or credentials.

## Two agencies, one code path

Both agencies are exactly one `Source` YAML each — `examples/agencies/bucr.yaml` and
`examples/agencies/mbta.yaml`. Neither file, nor anything that reads it, contains an
`if agency == "bucr"` anywhere in `packages/`.

**MBTA** — a large public agency with a standard GTFS-RT JSON feed and no agency-specific
serving pin:

```yaml
name: mbta
gtfs_static_url: https://cdn.mbta.com/MBTA_GTFS.zip
vehicle_positions_url: https://cdn.mbta.com/realtime/VehiclePositions.json
trip_updates_url: https://cdn.mbta.com/realtime/TripUpdates.json
poll_interval_s: 15
timezone: America/New_York
holiday_region: US_MA
arrival_radius_m: 40.0
dwell_s: 0.0
shape_threshold_m: 30.0
serving:
  primary_model: null
  fallback_model: null
```

With no pin set, [selection](../concepts/selection-policy.md) falls through to ranking by
metric among eligible artifacts — MBTA doesn't need a manual override.

**bUCR** — SIMOVI's own small agency, served through `databus` (SIMOVI's backend) rather
than through `stoptimes serve`'s standalone publisher, and pinned to a specific trained
artifact with a fallback:

```yaml
name: bucr
gtfs_static_url: http://localhost:8000/feed/schedule/feed.zip
vehicle_positions_url: http://localhost:8000/feed/realtime/vehicle_positions.json
trip_updates_url: http://localhost:8000/feed/realtime/trip_updates.json
poll_interval_s: 5
timezone: America/Costa_Rica
holiday_region: CR
arrival_radius_m: 30.0
dwell_s: 0.0
shape_threshold_m: 25.0
serving:
  primary_model: xgboost_bucr_segments_corpus_spatial-temporal_global_..._max_depth=5_...
  fallback_model: polyreg_distance_bucr_segments_corpus_distance_global_..._degree=2_...
```

(URLs above point at a local dev stack, as the shipped example file documents — a real
deployment supplies its own.) The pin exists because bUCR's ranked-by-metric artifact set
isn't yet rich enough to trust unpinned selection for production traffic; MBTA's `null`
pins show that this is a per-agency choice the contract supports, not a requirement.

## How databus actually uses this toolkit

Per [Selection policy: two things called "serving"](../concepts/selection-policy.md#two-things-called-serving),
databus is the proof that the prediction-library/publisher seam is in the right place:
databus owns its own GTFS-RT feed already, so it calls the prediction library directly and
never runs `stoptimes serve`'s standalone publisher. The cutover from databus's own
prediction code to this toolkit's library is planned as two deploys, not one
([`docs/PLAN.md` §8, G2](../PLAN.md#8-phase-4-generality-proof-and-simovi-cutover-sequential)):
first a one-line import swap with the same function signature (so a rollback is a revert),
then — once that's proven live — databus supplies its own `ingest`/`publish` pipeline
overrides and inherits `match`/`features`/`predict`/`postprocess` from the shared pipeline,
which is what finally makes training and serving compute features the same way instead of
agreeing only by inspection.

## What's still agency-specific, on purpose, for now

bUCR's raw AVL feed has no trip IDs — unlike MBTA's standard GTFS-RT, it needs roughly 1,240
lines of hand-tuned trip-inference logic to become `Observation`s at all. That code is not
yet a configurable, reusable step; it's `docs/PLAN.md`'s own G1 ("bUCR trip inference...
deserves its own track") acceptance item for the phase that follows this one, and the reason
the toolkit doesn't yet claim bUCR-style trip-less AVL as a generally solved input format,
even though everything else about bUCR — timezone, holidays, ground-truth thresholds,
serving pins — is configuration exactly like MBTA's.
