# F3 trip state notes (for F4 integration)

- Module: `stoptimes.core.features.trip_state`. Pass a `ForwardMatcher(polyline)` (built once per shape) as
  `polyline=` in `update_trip_state` / `arrivals_from_pings`; a raw `list[PolylinePoint]` also works but rebuilds
  numpy arrays on every ping.
- Serving flow per poll: `state = store.get(key)`; `state = update_trip_state(state, ping, ...)`; `store.put(state)`;
  call `store.evict(now)` periodically. `delay_last(state, sched_epochs, state.latest_progress_m(), ping.epoch)`.
- Training: `arrivals_from_pings(...)` returns the same `TripState`; `.arrival_arrays()` gives numpy `(arrival,
  detect, gap)`. To simulate a cold start, feed only the pings from the chosen start onward.
- `delay_last` expects `sched_epochs` aligned with `stop_progress_m` (F1 output); NaN schedule entries yield NaN.
- Online semantics: a ping with epoch <= the last accepted epoch is ignored (duplicate/out-of-order); batch applies the
  same rule in input order, so feed both in arrival order, not pre-sorted.
- Per-trip state keeps every ping (tuple); fine for a trip (~1e3 pings). A Redis store (G2b) should serialise
  `TripState` fields; they are plain tuples/floats.
- Needs elsewhere: none.
