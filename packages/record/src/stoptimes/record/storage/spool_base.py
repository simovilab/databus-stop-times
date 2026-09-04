"""Agency-agnostic DuckDB spool machinery shared by the VP and TU pollers.

The VehiclePositions spool (``rt_pipeline.storage.spool``) exists because a
per-poll ``COPY ... PARTITION_BY (..., route_id)`` fanned out into ~160 S3
PUTs every 5 seconds and pushed poll latency past 300s -- see that module's
docstring for the incident writeup. TripUpdates archival (``tu_spool``) has
exactly the same shape, so the durability/locking/flush logic lives here once
and each feed supplies only its schema via a :class:`SpoolSpec`.

Everything in this module is Django-free and unit-tested against a local
temp directory (DuckDB reads/writes a local Hive-partitioned directory
identically to ``s3://``).

UTC invariant: a tz-naive timestamp is ALWAYS interpreted as UTC, never local
time. Spool timestamp columns are DuckDB ``TIMESTAMP`` (tz-naive) holding UTC
wall-clock values; cutoffs are normalized by :func:`as_utc_naive` before
comparison.

Connection handling: DuckDB permits one read-write process per database file.
Every method opens a fresh connection and closes it before returning, so no
lock is ever held across a Celery task boundary -- a killed worker cannot
leave a stale writer lock. The corollary is that the poll queue MUST run at
``--concurrency=1``.
"""

from __future__ import annotations

import datetime as dt
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Mapping, Optional

import duckdb
import pandas as pd

DEFAULT_COLUMN_TYPE = "VARCHAR"
COMPRESSION = "zstd"  # better ratio than snappy for cold S3 storage

# --- spool file compaction thresholds ---------------------------------------
# DuckDB grows its database file to the high-water mark of live data and never
# shrinks it: a DELETE frees blocks for reuse but returns nothing to the
# filesystem, and the free list is not always reused across checkpoints. The VP
# spool reached 2.38 GB while holding 146 rows after ~3 weeks of poll/flush
# cycles. Rewriting the file into a fresh database is the only way to reclaim
# the space, so the flush does that when the file is both large in absolute
# terms and mostly dead space.
COMPACT_MIN_BYTES = 256 * 1024 * 1024
# Bytes of file per live row, above which the file is judged mostly dead space.
# Every spool row in this codebase serializes to a few hundred bytes; a healthy
# spool measures ~110 B/row after a rewrite. 4 KiB/row is therefore a generous
# ceiling that a real backlog (many rows, large file) stays far below, while the
# observed failure -- 2.38 GB holding 146 rows, ~16 MB/row -- clears it by four
# orders of magnitude. DuckDB's own PRAGMA database_size is deliberately NOT
# used: it reported 16 of 29 blocks "used" for a file that rewrote from 4.4 MB
# to 1.6 MB, because most of the reclaimed space is row-group fragmentation
# inside used blocks rather than free-list slack.
COMPACT_MAX_BYTES_PER_ROW = 4096


@dataclass(frozen=True)
class SpoolSpec:
    """Everything that differs between one feed's spool and another's.

    Frozen on purpose: a spec is shared by every ``BaseSpool`` instance of a
    class and must never be mutated at runtime.
    """

    table: str
    columns: tuple[str, ...]
    natural_key: tuple[str, ...]
    default_path: str
    column_types: Mapping[str, str] = field(default_factory=dict)
    ts_column: str = "ts"

    def column_type(self, name: str) -> str:
        return self.column_types.get(name, DEFAULT_COLUMN_TYPE)

    def missing_columns(self, df: pd.DataFrame) -> list[str]:
        return [c for c in self.columns if c not in df.columns]


def _sql_literal(value: str) -> str:
    """Single-quoted SQL string literal (ATTACH takes no bind parameters)."""
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def as_utc_naive(ts: dt.datetime) -> dt.datetime:
    """Normalize a cutoff to tz-naive UTC.

    Tz-naive values are already treated as UTC and pass through unchanged;
    tz-aware values are converted to UTC and stripped of tzinfo so they
    compare correctly against the tz-naive ``TIMESTAMP`` spool columns.
    """
    if ts.tzinfo is not None:
        return ts.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return ts


def add_partition_columns(df: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    """Return a copy of ``df`` with year/month/day derived from ``ts_col``.

    Partition keys are ALWAYS computed in UTC so the same observation lands in
    the same ``year/month/day`` cell no matter where the poller runs (Boston,
    Germany, ...). Immutable: never mutates the input frame.
    """
    out = df.copy()
    # utc=True converts tz-aware values to UTC and localizes tz-naive ones AS
    # UTC (never the local zone) -- the cross-host determinism guarantee.
    ts = pd.to_datetime(out[ts_col], utc=True)
    out["year"] = ts.dt.year.astype("int64")
    out["month"] = ts.dt.month.astype("int64")
    out["day"] = ts.dt.day.astype("int64")
    return out


class BaseSpool:
    """Durable local staging area between polls and the hourly S3 flush.

    Backed by a DuckDB file on disk (never ``:memory:``) so a container
    restart cannot lose observations collected since the last flush --
    durability is the entire reason polls no longer write straight to S3.

    Subclasses set :attr:`spec` and nothing else is required.
    """

    spec: ClassVar[SpoolSpec]

    def __init__(self, path: Optional[str] = None, *, must_exist: bool = False) -> None:
        self.path = path or self.spec.default_path
        if must_exist and not Path(self.path).exists():
            # Guards against opening a spool in the wrong process. The VP
            # flush ran once on a worker with no spool volume mounted: DuckDB
            # happily created an empty database there and the task reported a
            # cheerful {'flushed': 0} while the real spool grew without bound.
            # An absent file at flush time is always a misconfiguration -- the
            # poll task creates it long before the first flush -- so fail loud.
            raise FileNotFoundError(
                f"spool database missing at {self.path!r}. The flush must run on "
                "the worker that mounts the spool volume (queue 'fetch'); "
                "opening it elsewhere silently creates an empty database."
            )
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # --- connection / DDL --------------------------------------------------

    def _connect(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect(self.path)
        # Force UTC as the session timezone. Without this, casting a tz-aware
        # (UTC) pandas column into a tz-naive TIMESTAMP column silently
        # applies the HOST's local offset: on a UTC-6 host, 08:00 UTC landed
        # in the table as 02:00. That would violate the UTC invariant above
        # on any non-UTC host (dev laptop, misconfigured container, ...).
        con.execute("SET TimeZone='UTC'")
        return con

    def _init_schema(self) -> None:
        spec = self.spec
        cols_sql = ", ".join(f"{c} {spec.column_type(c)}" for c in spec.columns)
        key_sql = ", ".join(spec.natural_key)
        con = self._connect()
        try:
            con.execute(
                f"CREATE TABLE IF NOT EXISTS {spec.table} "
                f"({cols_sql}, UNIQUE ({key_sql}))"
            )
        finally:
            con.close()

    # --- write / read ------------------------------------------------------

    def append(self, df: pd.DataFrame) -> int:
        """Insert new rows, skipping any that collide on the natural key.

        Returns the number of rows NEWLY inserted -- not the number offered --
        so callers can distinguish ``rows_fetched`` from ``rows_new``. Never
        mutates ``df``.
        """
        if df is None or df.empty:
            return 0
        spec = self.spec
        miss = spec.missing_columns(df)
        if miss:
            raise ValueError(f"DataFrame missing required columns: {miss}")

        # Fixed column order matching the spec; a new frame, not a mutation
        # of the caller's df.
        pdf = df.loc[:, list(spec.columns)]
        cols_sql = ", ".join(spec.columns)
        key_sql = ", ".join(spec.natural_key)
        con = self._connect()
        try:
            before = con.execute(f"SELECT count(*) FROM {spec.table}").fetchone()[0]
            con.register("spool_batch", pdf)
            try:
                con.execute(
                    f"INSERT INTO {spec.table} ({cols_sql}) "
                    f"SELECT {cols_sql} FROM spool_batch "
                    f"ON CONFLICT ({key_sql}) DO NOTHING"
                )
            finally:
                con.unregister("spool_batch")
            after = con.execute(f"SELECT count(*) FROM {spec.table}").fetchone()[0]
            return int(after - before)
        finally:
            con.close()

    def select_before(self, cutoff: dt.datetime) -> pd.DataFrame:
        """Rows with ``ts < cutoff`` (half-open, UTC). Read-only.

        Deliberately separate from :meth:`delete_before` so a flush can write
        to S3, verify the write, and only then delete -- a failed PUT never
        loses rows.
        """
        spec = self.spec
        con = self._connect()
        try:
            return con.execute(
                f"SELECT * FROM {spec.table} WHERE {spec.ts_column} < ? "
                f"ORDER BY {spec.ts_column}",
                [as_utc_naive(cutoff)],
            ).df()
        finally:
            con.close()

    def delete_before(self, cutoff: dt.datetime) -> int:
        """Delete rows with ``ts < cutoff`` (half-open, UTC). Returns rows deleted."""
        spec = self.spec
        con = self._connect()
        try:
            before = con.execute(f"SELECT count(*) FROM {spec.table}").fetchone()[0]
            con.execute(
                f"DELETE FROM {spec.table} WHERE {spec.ts_column} < ?",
                [as_utc_naive(cutoff)],
            )
            after = con.execute(f"SELECT count(*) FROM {spec.table}").fetchone()[0]
            return int(before - after)
        finally:
            con.close()

    def stats(self) -> dict:
        """Row count, oldest/newest ``ts``, and on-disk file size.

        Safe to call on an empty spool (``oldest_ts``/``newest_ts`` are
        ``None``, ``rows`` is ``0``).
        """
        spec = self.spec
        con = self._connect()
        try:
            rows, oldest_ts, newest_ts = con.execute(
                f"SELECT count(*), min({spec.ts_column}), max({spec.ts_column}) "
                f"FROM {spec.table}"
            ).fetchone()
        finally:
            con.close()
        try:
            size_bytes = os.path.getsize(self.path)
        except OSError:
            size_bytes = 0
        return {
            "rows": int(rows or 0),
            "oldest_ts": oldest_ts,
            "newest_ts": newest_ts,
            "bytes": size_bytes,
        }

    # --- file compaction ---------------------------------------------------

    def should_compact_file(
        self,
        *,
        min_bytes: int = COMPACT_MIN_BYTES,
        max_bytes_per_row: int = COMPACT_MAX_BYTES_PER_ROW,
    ) -> bool:
        """True when the file is worth rewriting.

        Both gates must hold. The size gate keeps the rewrite off the hourly
        path entirely for a healthy spool; the bytes-per-row gate stops us from
        rewriting a spool that is large because it legitimately holds a lot (a
        long S3 outage backlog), where the rewrite would be expensive and
        reclaim little. An empty spool in a large file always qualifies -- that
        is the clearest possible case of dead space.
        """
        try:
            size_bytes = os.path.getsize(self.path)
        except OSError:
            return False
        if size_bytes < min_bytes:
            return False
        con = self._connect()
        try:
            rows = con.execute(f"SELECT count(*) FROM {self.spec.table}").fetchone()[0]
        finally:
            con.close()
        rows = int(rows or 0)
        return rows == 0 or (size_bytes / rows) > max_bytes_per_row

    def compact_file(self) -> dict[str, Any]:
        """Rewrite the spool into a fresh database file, reclaiming dead space.

        Copies the live rows into a new DuckDB file next to this one, verifies
        the row count matches, and only then swaps it in with an atomic
        ``os.replace``. If anything raises, the temp file is removed and the
        original is left exactly as it was -- the spool's contents are never at
        risk, only its size.

        Requires the same exclusive access as every other method: one
        read-write process per file, i.e. the ``fetch`` worker at
        ``--concurrency=1``.
        """
        spec = self.spec
        src = Path(self.path)
        bytes_before = os.path.getsize(src)
        tmp = src.with_name(f"{src.name}.compact-{uuid.uuid4().hex}")
        tmp_wal = Path(f"{tmp}.wal")
        cols_sql = ", ".join(f"{c} {spec.column_type(c)}" for c in spec.columns)
        key_sql = ", ".join(spec.natural_key)
        names_sql = ", ".join(spec.columns)
        target = _sql_literal(str(tmp))

        try:
            con = self._connect()
            try:
                expected = con.execute(
                    f"SELECT count(*) FROM {spec.table}"
                ).fetchone()[0]
                con.execute(f"ATTACH {target} AS compact_target")
                try:
                    con.execute(
                        f"CREATE TABLE compact_target.{spec.table} "
                        f"({cols_sql}, UNIQUE ({key_sql}))"
                    )
                    con.execute(
                        f"INSERT INTO compact_target.{spec.table} ({names_sql}) "
                        f"SELECT {names_sql} FROM {spec.table}"
                    )
                    got = con.execute(
                        f"SELECT count(*) FROM compact_target.{spec.table}"
                    ).fetchone()[0]
                finally:
                    con.execute("DETACH compact_target")
                if int(got) != int(expected):
                    raise RuntimeError(
                        f"spool compaction verify mismatch: {expected} rows in "
                        f"{self.path}, {got} in the rewrite"
                    )
            finally:
                con.close()

            # A leftover WAL beside either file would be replayed against the
            # wrong database after the swap. Both connections were closed
            # cleanly, so this should never fire; if it does, abandon the
            # rewrite rather than risk the spool.
            if Path(f"{src}.wal").exists() or tmp_wal.exists():
                raise RuntimeError(
                    "WAL present after a clean close; skipping spool compaction"
                )
            os.replace(tmp, src)
        except Exception:
            for leftover in (tmp, tmp_wal):
                try:
                    leftover.unlink()
                except OSError:
                    pass
            raise

        return {
            "compacted": True,
            "rows": int(expected),
            "bytes_before": int(bytes_before),
            "bytes_after": int(os.path.getsize(src)),
        }

    def maybe_compact_file(
        self,
        *,
        min_bytes: int = COMPACT_MIN_BYTES,
        max_bytes_per_row: int = COMPACT_MAX_BYTES_PER_ROW,
    ) -> dict[str, Any]:
        """:meth:`compact_file` behind :meth:`should_compact_file`, fail-safe.

        Reclaiming disk is housekeeping, never a reason to fail a flush that
        already wrote and verified its rows, so a failure here is reported in
        the return value instead of raised.
        """
        try:
            if not self.should_compact_file(
                min_bytes=min_bytes, max_bytes_per_row=max_bytes_per_row
            ):
                return {"compacted": False}
            return self.compact_file()
        except Exception as exc:  # noqa: BLE001 - housekeeping must not fail a flush
            return {"compacted": False, "error": f"{type(exc).__name__}: {exc}"}

    # --- flush hook --------------------------------------------------------

    def prepare_flush_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform drained rows before they are written to staging.

        Default: identity. ``tu_spool`` overrides this to run-length encode
        repeated predictions, which is where its storage savings come from.
        Must be pure -- never mutate ``df``.
        """
        return df


def flush_spool_to_staging(
    spool: BaseSpool,
    base_uri: str,
    cutoff: dt.datetime,
    *,
    connect_fn: Callable[..., Any],
    config: Any = None,
) -> dict[str, Any]:
    """Drain rows older than ``cutoff`` into a Parquet dataset at ``base_uri``,
    partitioned ONLY by ``(year, month, day)``.

    Deliberately no ``route_id`` in the partitioning -- that per-route fan-out
    on every write is exactly what turned each poll into ~160 S3 PUTs and
    caused the OOM incident the spool exists to fix. One flush call writes at
    most a handful of objects (one per day the drained rows span).

    Write-then-verify-then-delete: after ``COPY``, the row count is read back
    from the written files' Parquet footers (no data scan) and checked against
    what was meant to be written; rows are deleted from the spool ONLY after
    that check passes. If the write or verify raises, the exception propagates
    and the spool is left untouched -- callers should treat that as "try again
    next flush", not data loss. A retried flush's overlapping rows are
    expected; downstream compaction dedups them.

    ``connect_fn`` is injected rather than imported here so each feed's module
    owns (and tests can monkeypatch) its own S3 connection factory.

    Returns the rows deleted from the spool, the days touched, the rows
    actually written (smaller than ``flushed`` whenever ``prepare_flush_frame``
    compacts), and a ``compaction`` report from
    :meth:`BaseSpool.maybe_compact_file` -- the spool file is rewritten after a
    successful drain when it has grown mostly-empty.
    """
    df = spool.select_before(cutoff)
    if df.empty:
        # Nothing to drain, but a spool that has been idle since its last big
        # flush is exactly the one worth shrinking.
        return {
            "flushed": 0,
            "days": [],
            "written": 0,
            "compaction": spool.maybe_compact_file(),
        }

    prepared = spool.prepare_flush_frame(df)
    if prepared.empty:  # pragma: no cover - a compactor must keep >=1 row
        raise RuntimeError("prepare_flush_frame dropped every row; refusing to delete")
    pdf = add_partition_columns(prepared, spool.spec.ts_column)

    file_id = uuid.uuid4().hex
    con = connect_fn(base_uri, config)
    try:
        con.register("spool_flush", pdf)
        try:
            con.execute(
                f"""
                COPY spool_flush TO '{base_uri}' (
                    FORMAT PARQUET,
                    PARTITION_BY (year, month, day),
                    COMPRESSION '{COMPRESSION}',
                    FILENAME_PATTERN '{file_id}_{{i}}',
                    OVERWRITE_OR_IGNORE TRUE
                );
                """
            )
        finally:
            con.unregister("spool_flush")

        # Verify: sum the row counts (footers only) of exactly the files this
        # call wrote, identified by the uuid filename prefix.
        written_glob = f"{base_uri}/**/{file_id}_*.parquet"
        row = con.execute(
            f"SELECT coalesce(sum(num_rows), 0) FROM parquet_file_metadata('{written_glob}')"
        ).fetchone()
        verified_rows = int(row[0]) if row else 0
        if verified_rows != len(pdf):
            raise RuntimeError(
                f"flush verify mismatch: wrote {len(pdf)} rows, "
                f"found {verified_rows} in staging"
            )
    finally:
        con.close()

    # Only reached once the write is verified -> safe to delete.
    deleted = spool.delete_before(cutoff)
    # The delete freed blocks inside the file but returned nothing to the
    # filesystem; reclaim it here, after the rows are safely in staging and
    # gone from the spool, so a rewrite can never race a pending flush.
    compaction = spool.maybe_compact_file()
    days_df = pdf[["year", "month", "day"]].drop_duplicates()
    days = [
        f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
        for y, m, d in days_df.itertuples(index=False)
    ]
    return {
        "flushed": deleted,
        "days": days,
        "written": int(len(pdf)),
        "compaction": compaction,
    }
