# p4-f1-schedule notes

- No cross-track edits. `features/registry.py` untouched; F4 should register `delay_now`, `sched_rem`,
  `sched_rem_t` (input tag `schedule`) and call `schedule_features(...)` per tick.
- Per tick F4 needs: `loader.trip_schedule(trip_id)` -> `scheduled_epochs(sched, trip_start_date, tz)` with
  `tz = resolve_timezone(loader.agency_timezone(), config.timezone)`. `start_date` must be the trip's *service date*
  (YYYYMMDD from the VehiclePosition `trip.start_date`, or derived by F4), not the calendar date of the ping.
- `stop_progress_m` is the per-trip stop progress along the shape (from spatial code); lengths must match
  `TripSchedule.stop_ids`. Stops skipped by `trip_geometry` (unknown stop_id) would misalign; F4 should assert.
- Missing schedule values are NaN and propagate to `sched_rem*`; `delay_now` skips NaN stops when interpolating.
  Leading/trailing blank times (illegal in GTFS) stay NaN and are not flagged `interpolated`.
- Circularity: `gtfs/static.py` imports `features/schedule.py` (pure, no static import). Keep it that way.
