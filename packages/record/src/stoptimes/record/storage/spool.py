"""Local DuckDB spool for MBTA VehiclePositions.

Why this exists: ``s3_writer.write_vehicle_positions`` runs one
``COPY ... PARTITION_BY (year, month, day, route_id)`` per call. Calling it
once per poll means one S3 PUT per active route per poll -- with ~160 active
MBTA routes that's ~160 round-trips from Hetzner to the MinIO server in Costa
Rica every 5 seconds. That fan-out is what pushed poll latency to 330+
seconds against a 5-second beat schedule, backed up the Celery queue without
bound, and eventually OOM-killed the worker (see the plan / commit history
for the full incident writeup).

The fix: a poll writes to this local spool instead (~200ms), and a separate
hourly task (``rt_pipeline.tasks.flush_vp_spool_s3``) drains it to S3 with
ONE ``COPY ... PARTITION_BY (year, month, day)`` per (day) partition --
``route_id`` is deliberately excluded from the flush partitioning; that is
the fan-out this module exists to avoid.

The durability, locking and flush machinery is shared with the TripUpdates
spool and lives in ``spool_base``; this module supplies only the VP schema.
See that module for the UTC and single-writer invariants.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Optional

from .config import S3Config
from .s3_writer import connect
from .schema import DATA_COLUMNS
from .spool_base import BaseSpool, SpoolSpec, flush_spool_to_staging

# DuckDB column types for the spool table, keyed by column name. The DDL is
# derived from schema.DATA_COLUMNS + route_id (SPOOL_COLUMNS below) rather
# than a hardcoded column list, so a schema change to DATA_COLUMNS only needs
# a type added here (unrecognized columns default to VARCHAR).
_COLUMN_TYPES: dict[str, str] = {
    "feed_name": "VARCHAR",
    "vehicle_id": "VARCHAR",
    "ts": "TIMESTAMP",
    "lat": "DOUBLE",
    "lon": "DOUBLE",
    "bearing": "DOUBLE",
    "speed": "DOUBLE",
    "trip_id": "VARCHAR",
    "current_stop_sequence": "INTEGER",
    "current_status": "VARCHAR",
    "stop_id": "VARCHAR",
    "ingested_at": "TIMESTAMP",
    "route_id": "VARCHAR",
}

# Spool table columns = the data columns (schema.py) + the route partition
# key, in a fixed order used for both DDL and INSERT statements.
SPOOL_COLUMNS: tuple[str, ...] = (*DATA_COLUMNS, "route_id")

# Natural key from S3_LAYOUT.md: "Natural key for dedupe on replay/backfill:
# (feed_name, vehicle_id, ts)".
NATURAL_KEY: tuple[str, ...] = ("feed_name", "vehicle_id", "ts")

DEFAULT_SPOOL_PATH = "/data/spool/vp_spool.duckdb"

TABLE_NAME = "vp_spool"

VP_SPOOL_SPEC = SpoolSpec(
    table=TABLE_NAME,
    columns=SPOOL_COLUMNS,
    natural_key=NATURAL_KEY,
    default_path=DEFAULT_SPOOL_PATH,
    column_types=_COLUMN_TYPES,
    ts_column="ts",
)


class Spool(BaseSpool):
    """Durable local staging area for VehiclePositions between polls and flush."""

    spec = VP_SPOOL_SPEC


def flush_to_staging(
    spool: Spool,
    base_uri: str,
    cutoff: dt.datetime,
    *,
    config: Optional[S3Config] = None,
) -> dict[str, Any]:
    """Drain VP rows older than ``cutoff`` to the staging dataset at ``base_uri``.

    Thin wrapper over ``spool_base.flush_spool_to_staging``; ``connect`` is
    passed by name (not imported inside the generic helper) so this module
    stays the single place a VP flush's S3 connection is resolved.
    """
    return flush_spool_to_staging(
        spool, base_uri, cutoff, connect_fn=connect, config=config
    )
