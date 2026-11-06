# p4-f2-assets notes (for F4 integration)

- Sidecar assets: `SegmentStats.save(path) -> sha256` / `SegmentStats.load(path)` (single file, own binary format,
  no pickle, versioned) and `RouteVocabulary.to_dict()/from_dict()` (JSON-safe). Artifact contract must carry both
  plus the agency `tz` and record the stats sha256. Not edited here (contracts are F4's).
- Training rows must use `OutOfFoldStats.for_day(service_date)`; serving and validation use `OutOfFoldStats.full`
  (save only `full`).
- `prior_remaining` handles ONE vehicle state (many targets). For a training frame, call it per (trip, instant) or
  vectorise later; for a bulk path `SegmentStats.segment_ids` + `lookup_ids` avoid per-call key hashing.
- Hour/daytype use the UTC offset at local noon of the service date (study semantics), constant for the day.
- `RouteVocabulary.encode` gives NaN for unseen ids; the model must be trained with missing allowed (xgboost is).
- No new dependencies. Segment input needs `stop_progress_m` and a per-instance `trip_key` (must not repeat across dates
  for different trips; `service_date` is also compared).
