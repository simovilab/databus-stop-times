"""Local DuckDB spool for MBTA TripUpdates, with run-length compaction.

Same shape as the VehiclePositions spool (poll -> local DuckDB -> hourly S3
flush, never S3 per poll); the shared machinery is in ``spool_base`` and the
column layout in ``tu_schema``. What is specific to TripUpdates is the
compaction step below.

Why compact: the TU feed republishes every active prediction on every poll,
whether or not it changed. Measured against the live MBTA feed, 70.2% of
predictions change between consecutive 24s observations -- which also means
~30% of rows are pure repetition, and that fraction rises steeply on a
shorter beat. Storing a repeat teaches a replay nothing: "what did the agency
predict at time T" is answered by the most recent row with ``ts <= T``, so
dropping a row identical to its immediate predecessor is lossless for that
question.

Compaction is done at flush, not at poll:

* it is one SQL pass per hour instead of a lookup per row per poll, so the
  poll stays a ~200ms local insert;
* it compares against the *immediate predecessor* rather than deduping on
  value, which is what makes an A -> B -> A sequence survive intact (a
  value-keyed unique constraint would silently drop the second A and make the
  replay wrong);
* the first observation in each flush window is always kept, so the
  "most recent row at or before T" reconstruction stays correct across
  hourly file boundaries.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Optional

import duckdb
import pandas as pd

from .config import S3Config
from .s3_writer import connect
from .spool_base import BaseSpool, SpoolSpec, flush_spool_to_staging
from .tu_schema import COLUMN_TYPES, DATA_COLUMNS, NATURAL_KEY, SERIES_KEY, VALUE_COLUMNS

DEFAULT_TU_SPOOL_PATH = "/data/spool/tu_spool.duckdb"

TABLE_NAME = "tu_spool"

TU_SPOOL_SPEC = SpoolSpec(
    table=TABLE_NAME,
    columns=DATA_COLUMNS,
    natural_key=NATURAL_KEY,
    default_path=DEFAULT_TU_SPOOL_PATH,
    column_types=COLUMN_TYPES,
    ts_column="ts",
)


def compact_repeated_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows identical to the immediately preceding row of the same series.

    A "series" is one (feed, trip instance, stop) -- ``tu_schema.SERIES_KEY``
    -- ordered by ``ts``; "identical" means agreeing on every column in
    ``tu_schema.VALUE_COLUMNS``. Comparison is null-safe (``IS NOT DISTINCT
    FROM``), so a prediction that has no arrival time in two consecutive polls
    counts as unchanged rather than as two differing NULLs.

    Pure: returns a new frame and never mutates ``df``. Column order and
    dtypes are preserved so the result can be written straight to Parquet.
    """
    if df.empty:
        return df.copy()

    lags = ",\n            ".join(
        f"lag({c}) OVER series AS prev_{c}" for c in VALUE_COLUMNS
    )
    unchanged = "\n              AND ".join(
        f"{c} IS NOT DISTINCT FROM prev_{c}" for c in VALUE_COLUMNS
    )
    series_sql = ", ".join(SERIES_KEY)
    cols_sql = ", ".join(df.columns)

    con = duckdb.connect()
    try:
        con.execute("SET TimeZone='UTC'")
        con.register("tu_rows", df)
        return con.execute(
            f"""
            WITH ordered AS (
                SELECT *,
                    row_number() OVER series AS _rn,
                    {lags}
                FROM tu_rows
                WINDOW series AS (PARTITION BY {series_sql} ORDER BY ts)
            )
            SELECT {cols_sql} FROM ordered
            WHERE _rn = 1
               OR NOT ({unchanged})
            ORDER BY ts
            """
        ).df()
    finally:
        con.close()


class TripUpdateSpool(BaseSpool):
    """Durable local staging area for TripUpdates between polls and flush."""

    spec = TU_SPOOL_SPEC

    def prepare_flush_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        return compact_repeated_predictions(df)


def flush_tu_to_staging(
    spool: TripUpdateSpool,
    base_uri: str,
    cutoff: dt.datetime,
    *,
    config: Optional[S3Config] = None,
) -> dict[str, Any]:
    """Drain TU rows older than ``cutoff`` to the staging dataset at ``base_uri``.

    Returns the shared flush result: ``flushed`` is rows removed from the
    spool, ``written`` is rows that survived compaction and reached Parquet.
    """
    return flush_spool_to_staging(
        spool, base_uri, cutoff, connect_fn=connect, config=config
    )
