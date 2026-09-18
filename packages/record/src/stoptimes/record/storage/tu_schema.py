"""S3 Hive-partitioned Parquet layout + schema for MBTA TripUpdates.

TripUpdates are archived for one reason: replaying a window of history so a
served model can be scored against the agency's own published predictions
(``etaval`` replay runs) instead of against whatever 30-minute live window
happened to be available. That purpose sets every design choice here:

* **Route-scoped at poll time.** The whole feed is ~22x the size of the VP
  feed per poll (measured: 94.5 kB vs 4.3 kB). Archiving only the routes the
  corpus actually models is the difference between ~20 GB and ~250 MB over a
  90-day window, and it is the only lever that big -- so it is applied at the
  poller, before anything is spooled.
* **Horizon-capped.** Predictions further out than an hour are dominated by
  schedule-plus-historical fallbacks, not realtime, and are ~5% of rows.
* **Run-length encoded at flush.** A prediction repeated unchanged across
  polls is stored once; see ``tu_spool.compact_repeated_predictions``.
* **Staging only.** There is no curated re-partitioning step like the VP
  pipeline's, because nothing reads these by route in production -- replay
  reads a date range once.

Layout::

    s3://transit/feeds/mbta/trip_updates_staging/
        year=YYYY/month=M/day=D/<uuid>.parquet

``ts`` is the TripUpdate's own ``timestamp`` (falling back to the feed header
timestamp) -- the instant the prediction was in effect. That, not the
predicted arrival time, is the partition key and the replay clock.
"""

from __future__ import annotations

BUCKET = "transit"
BASE_PREFIX = "feeds/mbta/trip_updates_staging"
PARTITION_COLUMNS = ("year", "month", "day")

# --- Parameterized layout (Phase 2, T5) ------------------------------------
# See ``schema.py``'s equivalent note: ``BASE_PREFIX`` stays a frozen literal
# for back-compat; new agency-generic code uses this template, which defaults
# to exactly today's MBTA layout.
STAGING_PREFIX_TEMPLATE = "feeds/{source_id}/trip_updates_staging"


def staging_prefix(source_id: str = "mbta", template: str = STAGING_PREFIX_TEMPLATE) -> str:
    """TripUpdates staging prefix for ``source_id``.

    ``staging_prefix("mbta") == BASE_PREFIX`` -- existing archives keep
    working unchanged.
    """
    return template.format(source_id=source_id)

# Columns persisted inside each Parquet file. Deliberately narrow: no
# scheduled-time echoes, no occupancy, no stop_time_properties -- every column
# here is either part of the natural key, part of the prediction, or needed to
# tell a realtime prediction apart from a schedule fallback (``uncertainty``).
DATA_COLUMNS: tuple[str, ...] = (
    "feed_name",
    "route_id",
    "trip_id",
    "start_date",
    "start_time",
    "direction_id",
    "vehicle_id",
    "trip_schedule_relationship",
    "revenue",
    "stop_id",
    "stop_sequence",
    "arrival_time",
    "arrival_delay",
    "arrival_uncertainty",
    "departure_time",
    "departure_delay",
    "departure_uncertainty",
    "stu_schedule_relationship",
    "ts",
    "header_ts",
    "ingested_at",
)

COLUMN_TYPES: dict[str, str] = {
    "feed_name": "VARCHAR",
    "route_id": "VARCHAR",
    "trip_id": "VARCHAR",
    "start_date": "VARCHAR",
    "start_time": "VARCHAR",
    "direction_id": "INTEGER",
    "vehicle_id": "VARCHAR",
    "trip_schedule_relationship": "VARCHAR",
    "revenue": "BOOLEAN",
    "stop_id": "VARCHAR",
    "stop_sequence": "INTEGER",
    "arrival_time": "TIMESTAMP",
    "arrival_delay": "INTEGER",
    "arrival_uncertainty": "INTEGER",
    "departure_time": "TIMESTAMP",
    "departure_delay": "INTEGER",
    "departure_uncertainty": "INTEGER",
    "stu_schedule_relationship": "VARCHAR",
    "ts": "TIMESTAMP",
    "header_ts": "TIMESTAMP",
    "ingested_at": "TIMESTAMP",
}

# One prediction is identified by which stop of which trip instance it is
# about, as of when. ``start_date`` is what separates today's run of a trip
# from tomorrow's; ``ts`` is the "as of".
NATURAL_KEY: tuple[str, ...] = (
    "feed_name",
    "trip_id",
    "start_date",
    "stop_id",
    "stop_sequence",
    "ts",
)

# The identity of a prediction *series* -- everything in the natural key
# except the "as of" instant. Run-length encoding groups on this.
SERIES_KEY: tuple[str, ...] = (
    "feed_name",
    "trip_id",
    "start_date",
    "stop_id",
    "stop_sequence",
)

# What counts as "the same observation". Two consecutive rows of a series that
# agree on every one of these carry no new information and the later one is
# dropped at flush time.
#
# `vehicle_id` is in here even though it is not part of the prediction: a
# vehicle swap on a trip whose predicted times happen not to move is a real
# event, and it is what a join from this archive to the VehiclePositions
# archive keys on -- compacting it away would make that join silently miss
# the reassignment. Same reasoning for `direction_id`/`route_id`, which cost
# nothing to compare because they almost never change mid-trip.
VALUE_COLUMNS: tuple[str, ...] = (
    "arrival_time",
    "arrival_delay",
    "arrival_uncertainty",
    "departure_time",
    "departure_delay",
    "departure_uncertainty",
    "stu_schedule_relationship",
    "trip_schedule_relationship",
    "vehicle_id",
    "direction_id",
    "route_id",
    "revenue",
)


def s3_base_uri(bucket: str = BUCKET, prefix: str = BASE_PREFIX) -> str:
    """Return the canonical ``s3://`` base URI for the dataset."""
    return f"s3://{bucket}/{prefix}"
