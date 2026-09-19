# Observations

An `Observation` is one realtime record: a vehicle-position sample, or one stop-time
prediction read from a TripUpdate feed. `kind` tells you which (`vehicle_position` or
`trip_update`).

## Why one shape for two very different things

Vehicle positions and trip updates come from different GTFS-RT message types with almost no
overlapping fields. Keeping them as one discriminated `Observation` type — rather than two
unrelated classes — means every reader, the recorder, and every downstream consumer
(map-matching, feature building, replay) can share one code path for "read some
observations, dispatch on `kind` where the fields actually differ." Two full parallel type
hierarchies would have meant two of everything: two readers, two partition schemes, two
validators.

## Why source-agnostic

`Observation` never mentions "MBTA" or "bUCR." It's addressed by `source_id`, `feed_name`,
`trip_id`/`route_id`/`stop_id` — identifiers any agency's GTFS already has. This is what lets
`stoptimes.core.readers` turn a GTFS-RT protobuf feed, a GTFS-RT JSON feed, an archived
Parquet partition, or a raw AVL CSV (with a column-mapping file) into the *same* table. A
model or a validator written against `Observation` never needs to know which of those four
things it came from.

## Timestamps

Every timestamp column is `datetime64[ns, UTC]`, pinned exactly to that resolution (not just
"UTC-aware") — `coerce_utc_ns` is the one function that enforces it, and
`validate_frame` treats a merely-close dtype (e.g. pandas' `us` inference on newer pandas) as
a defect, not a pass. This exists because Phase 0 was bitten by exactly that silent
downgrade: code assumed `ns`, pandas 2.x/3.x sometimes infers `us`, and comparisons or joins
that looked fine quietly returned wrong answers. `ingested_at` (when the reader/recorder saw
the record) is kept separate from `ts` (the record's own timestamp) because the gap between
them is itself a signal — see [staleness in Diagnose a feed](../guides/diagnose-a-feed.md).

## Where they come from, where they go

- **T1's readers** (`stoptimes.core.readers`) produce `Observation`s from any of the four
  formats above.
- **T5's recorder** (`stoptimes record start`) writes them to a Hive-partitioned Parquet
  archive (`year=/month=/day=/route_id=` for vehicle positions, `year=/month=/day=` for trip
  updates) — agency-agnostic partitioning, unlike the pre-migration archive layout, which
  baked an agency name into the path.
- **T2's arrival derivation** (`stoptimes data arrivals`) consumes vehicle-position
  observations to produce [Arrivals](arrivals.md).
- **T6's validation** replays archived observations to score any prediction source against
  them.

Full field list and the two partition schemes: [Contracts §2](../CONTRACTS.md#2-data-canonical-data-tables).
