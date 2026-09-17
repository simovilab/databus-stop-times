"""Contract: canonical data tables.

PLACEHOLDER — Phase 1, track C1 (docs/PLAN.md). Must cover:
- StaticFeed: a loaded GTFS zip (stops, trips, stop_times, shapes, calendars).
- Observation: one realtime record (vehicle position or trip update), source-agnostic.
- Arrival: one actual (derived or reported) stop arrival, with how it was derived.
- Column names, dtypes, timezone rules and the parquet layout for each table.
Existing starting points: stoptimes.core.domain.models, stoptimes.core.gtfs.static,
stoptimes.core.realtime.gtfsrt_json, stoptimes.record.storage.schema / tu_schema.
"""
