"""Archived-Parquet replay feed source. Phase A1.

Reads the gtfs-eta collector's on-disk layout: Hive-partitioned Parquet at
``year=/month=/day=/route_id=`` with:

- Vehicle-Positions (VP) columns: ``feed_name, vehicle_id, ts, lat, lon,
  bearing, speed, trip_id, current_stop_sequence, current_status, stop_id,
  ingested_at``.
- Trip-Updates (TU) rows: ONE ROW PER StopTimeUpdate — ``ts, trip_id,
  route_id, ..., stop_sequence, stop_id, arrival_delay, arrival_time, ...``.

``ts`` is epoch seconds UTC; ``arrival_time``/``departure_time`` may be epoch
seconds OR a timestamp column — both are handled.

The public surface:

- ``read_vp_groups`` / ``read_tu_groups`` — windowed, route-filtered reads.
- ``build_archive_snapshots`` — VP ticks (ascending, distinct ``ts``) each
  paired with the TU baseline observed at or before that tick.
- ``ArchiveReplayFeedSource`` — a ``ReplayFeedSource`` that yields those
  snapshots, so the engine's clock (``FeedSnapshot.header_timestamp``) walks
  the archived timeline — no wall-clock dependency.
- ``latest_local_snapshot_on_or_before`` / ``resolve_static_gtfs`` — pick the
  static GTFS zip whose schedule was in effect at replay time.
"""

from __future__ import annotations

import datetime as dt
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from stoptimes.val.engine.runtime import FeedSnapshot, ReplayFeedSource
from stoptimes.core.realtime.gtfsrt_json import StopTimeUpdate, TripUpdateRecord, VehiclePositionRecord

EpochLike = float | int | str | dt.datetime | None

# ---------------------------------------------------------------------------
# Time coercion helpers
# ---------------------------------------------------------------------------


def _to_epoch(value: EpochLike) -> float | None:
    """Coerce *value* (epoch number, ISO 8601 string, or datetime) to epoch seconds UTC."""
    if value is None:
        return None
    if isinstance(value, bool):  # guard: bool is an int subclass
        raise TypeError(f"Cannot convert bool {value!r} to epoch seconds")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dt.datetime):
        d = value if value.tzinfo is not None else value.replace(tzinfo=dt.UTC)
        return d.timestamp()
    if isinstance(value, str):
        s = value.strip()
        try:
            return float(s)
        except ValueError:
            pass
        iso = s.replace("Z", "+00:00")
        parsed = dt.datetime.fromisoformat(iso)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.UTC)
        return parsed.timestamp()
    raise TypeError(f"Cannot convert {value!r} to epoch seconds")


def _col_to_epoch_seconds(series: pd.Series) -> pd.Series:
    """Normalize a column that may hold epoch numbers OR timestamps to epoch-seconds floats.

    pandas datetime64 columns may carry any time resolution ("s", "ms", "us",
    "ns") depending on the writer/pandas version, so the column is normalized
    to nanosecond resolution before converting to an integer count — dividing
    a raw ``.astype("int64")`` by 1e9 would silently be wrong for non-ns
    resolutions.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        s = series
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_convert("UTC")
        else:
            s = s.dt.tz_localize("UTC")
        s = s.astype("datetime64[ns, UTC]")
        return s.astype("int64").astype("float64") / 1e9
    return pd.to_numeric(series, errors="coerce")


# ---------------------------------------------------------------------------
# Parquet reading (local dir/file or s3://)
# ---------------------------------------------------------------------------


def _read_table(path: str | Path) -> pd.DataFrame:
    """Read a Hive-partitioned Parquet dataset, a single Parquet file, or an s3:// URI."""
    path_str = str(path)
    if path_str.startswith("s3://"):
        return _read_table_s3(path_str)

    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Archive path does not exist: {p}")

    import pyarrow.dataset as ds

    if p.is_file():
        dataset = ds.dataset(str(p), format="parquet")
    else:
        dataset = ds.dataset(str(p), format="parquet", partitioning="hive")
    return dataset.to_table().to_pandas()


def _read_table_s3(uri: str) -> pd.DataFrame:
    """Read a Hive-partitioned Parquet dataset from S3 via lazily-imported fsspec/s3fs."""
    try:
        import fsspec
        import s3fs  # noqa: F401  (registers the "s3" fsspec protocol)
    except ImportError as exc:
        raise RuntimeError(
            "Reading from s3:// paths requires the optional 'fsspec' and 's3fs' "
            "packages. Install them (e.g. `uv pip install fsspec s3fs`) or use a "
            "local path instead."
        ) from exc

    import pyarrow.dataset as ds
    import pyarrow.fs as pafs

    fs = fsspec.filesystem("s3")
    pa_fs = pafs.PyFileSystem(pafs.FSSpecHandler(fs))
    key = uri[len("s3://") :]
    dataset = ds.dataset(key, filesystem=pa_fs, format="parquet", partitioning="hive")
    return dataset.to_table().to_pandas()


def _apply_window(
    df: pd.DataFrame, column: str, start: EpochLike, end: EpochLike
) -> pd.DataFrame:
    """Filter *df* to rows where *column* is within [start, end] (either bound optional)."""
    start_epoch = _to_epoch(start)
    end_epoch = _to_epoch(end)
    if start_epoch is not None:
        df = df[df[column] >= start_epoch]
    if end_epoch is not None:
        df = df[df[column] <= end_epoch]
    return df


def _apply_route_filter(df: pd.DataFrame, route_ids: Any) -> pd.DataFrame:
    if not route_ids or "route_id" not in df.columns:
        return df
    wanted = {route_ids} if isinstance(route_ids, str) else set(route_ids)
    return df[df["route_id"].astype(str).isin(wanted)]


# ---------------------------------------------------------------------------
# VP reads
# ---------------------------------------------------------------------------


def read_vp_groups(
    path: str | Path,
    *,
    start: EpochLike = None,
    end: EpochLike = None,
    route_ids: list[str] | str | None = None,
) -> pd.DataFrame:
    """Read archived VehiclePosition rows from *path*.

    Returns a DataFrame sorted by ``ts`` ascending, optionally windowed by
    ``[start, end]`` epoch seconds and filtered to *route_ids*.
    """
    df = _read_table(path)
    if df.empty:
        return df
    df = df.copy()
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df = _apply_window(df, "ts", start, end)
    df = _apply_route_filter(df, route_ids)
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


def _row_to_vp_record(row: dict[str, Any]) -> VehiclePositionRecord:
    """Build a VehiclePositionRecord from one archived VP row (dict form)."""
    from stoptimes.core.domain.models import VehicleState

    def _opt_float(v: Any) -> float | None:
        return float(v) if v is not None and pd.notna(v) else None

    def _opt_int(v: Any) -> int | None:
        return int(v) if v is not None and pd.notna(v) else None

    def _opt_str(v: Any) -> str | None:
        return str(v) if v is not None and pd.notna(v) else None

    ts = float(row["ts"])
    state = VehicleState(
        epoch=ts,
        lat=float(row["lat"]),
        lon=float(row["lon"]),
        progress_m=None,
        speed=_opt_float(row.get("speed")),
        current_status=_opt_str(row.get("current_status")),
        current_stop_sequence=_opt_int(row.get("current_stop_sequence")),
    )
    return VehiclePositionRecord(
        trip_id=str(row["trip_id"]),
        route_id=_opt_str(row.get("route_id")) or "",
        direction_id=None,
        start_date=None,
        vehicle_id=_opt_str(row.get("vehicle_id")) or "",
        state=state,
    )


# ---------------------------------------------------------------------------
# TU reads (merged per trip x observation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TuGroup:
    """All StopTimeUpdates for one trip observed at one archived ``ts``."""

    trip_id: str
    route_id: str | None
    ts: float
    stop_time_updates: list[StopTimeUpdate]


def read_tu_groups(
    path: str | Path,
    *,
    start: EpochLike = None,
    end: EpochLike = None,
    route_ids: list[str] | str | None = None,
) -> list[TuGroup]:
    """Read archived TripUpdate rows from *path* and merge them per (trip_id, ts).

    Each archived row is one StopTimeUpdate; rows sharing a (trip_id, ts) are
    merged into one :class:`TuGroup`, ordered by ``stop_sequence``.  Groups
    are returned sorted by ``(ts, trip_id)``.
    """
    df = _read_table(path)
    if df.empty:
        return []
    df = df.copy()
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df = _apply_window(df, "ts", start, end)
    df = _apply_route_filter(df, route_ids)
    if df.empty:
        return []

    df["_arrival_epoch"] = (
        _col_to_epoch_seconds(df["arrival_time"]) if "arrival_time" in df.columns else pd.NA
    )
    df["_departure_epoch"] = (
        _col_to_epoch_seconds(df["departure_time"]) if "departure_time" in df.columns else pd.NA
    )
    if "stop_sequence" in df.columns:
        df["stop_sequence"] = pd.to_numeric(df["stop_sequence"], errors="coerce")

    groups: list[TuGroup] = []
    for (trip_id, ts), g in df.groupby(["trip_id", "ts"], sort=True):
        g_sorted = g.sort_values("stop_sequence", kind="stable")
        stus: list[StopTimeUpdate] = []
        route_id: str | None = None
        for rec in g_sorted.to_dict("records"):
            if route_id is None and rec.get("route_id") is not None and pd.notna(rec["route_id"]):
                route_id = str(rec["route_id"])
            stus.append(
                StopTimeUpdate(
                    stop_sequence=(
                        int(rec["stop_sequence"])
                        if pd.notna(rec.get("stop_sequence"))
                        else None
                    ),
                    stop_id=(str(rec["stop_id"]) if pd.notna(rec.get("stop_id")) else None),
                    arrival_time=(
                        float(rec["_arrival_epoch"])
                        if pd.notna(rec.get("_arrival_epoch"))
                        else None
                    ),
                    departure_time=(
                        float(rec["_departure_epoch"])
                        if pd.notna(rec.get("_departure_epoch"))
                        else None
                    ),
                )
            )
        groups.append(
            TuGroup(
                trip_id=str(trip_id),
                route_id=route_id,
                ts=float(ts),
                stop_time_updates=stus,
            )
        )
    groups.sort(key=lambda gr: (gr.ts, gr.trip_id))
    return groups


# ---------------------------------------------------------------------------
# Snapshot construction
# ---------------------------------------------------------------------------


def build_archive_snapshots(
    vp_path: str | Path,
    tu_path: str | Path | None = None,
    *,
    start: EpochLike = None,
    end: EpochLike = None,
    route_ids: list[str] | str | None = None,
) -> list[FeedSnapshot]:
    """Build the ordered FeedSnapshot sequence for archived replay.

    Ticks = the distinct archived VP ``ts`` values, ascending.  Each tick's
    snapshot carries that tick's VP records and, if *tu_path* is given, the
    TU group with the greatest ``ts <= tick`` per active trip (i.e. every
    model sees the same archived TU baseline at a given tick).  VP-only
    replay (``tu_path=None``) is supported.
    """
    vp_df = read_vp_groups(vp_path, start=start, end=end, route_ids=route_ids)
    if vp_df.empty:
        return []

    tu_by_trip: dict[str, list[TuGroup]] = {}
    if tu_path is not None:
        for g in read_tu_groups(tu_path, start=start, end=end, route_ids=route_ids):
            tu_by_trip.setdefault(g.trip_id, []).append(g)
        for lst in tu_by_trip.values():
            lst.sort(key=lambda gr: gr.ts)

    ticks = sorted(float(t) for t in vp_df["ts"].dropna().unique().tolist())

    snapshots: list[FeedSnapshot] = []
    for tick in ticks:
        tick_rows = vp_df[vp_df["ts"] == tick]
        vp_records = [_row_to_vp_record(row) for row in tick_rows.to_dict("records")]

        tu_records: list[TripUpdateRecord] = []
        for trip_id in {r.trip_id for r in vp_records}:
            candidates = tu_by_trip.get(trip_id)
            if not candidates:
                continue
            best: TuGroup | None = None
            for g in candidates:
                if g.ts <= tick:
                    best = g
                else:
                    break
            if best is None:
                continue
            tu_records.append(
                TripUpdateRecord(
                    trip_id=best.trip_id,
                    route_id=best.route_id,
                    direction_id=None,
                    start_date=None,
                    stop_time_updates=best.stop_time_updates,
                )
            )

        snapshots.append(
            FeedSnapshot(
                header_timestamp=tick,
                vp_records=vp_records,
                tu_header_timestamp=tick,
                tu_records=tu_records,
            )
        )

    return snapshots


class ArchiveReplayFeedSource(ReplayFeedSource):
    """Replays an archived Parquet VP/TU feed, wall-clock independent.

    Builds the full :class:`~stoptimes.val.engine.runtime.FeedSnapshot` sequence
    eagerly (via :func:`build_archive_snapshots`) and delegates iteration to
    :class:`~stoptimes.val.engine.runtime.ReplayFeedSource` — the engine derives its
    clock from ``snapshot.header_timestamp``, so it advances to each archived
    timestamp with no dependency on ``time.time()``.
    """

    def __init__(
        self,
        vp_path: str | Path,
        tu_path: str | Path | None = None,
        *,
        start: EpochLike = None,
        end: EpochLike = None,
        route_ids: list[str] | str | None = None,
    ) -> None:
        snapshots = build_archive_snapshots(
            vp_path, tu_path, start=start, end=end, route_ids=route_ids
        )
        super().__init__(snapshots)

    @property
    def snapshots(self) -> list[FeedSnapshot]:
        """The full archived snapshot sequence (read-only copy)."""
        return list(self._snapshots)


# ---------------------------------------------------------------------------
# Static GTFS matching
# ---------------------------------------------------------------------------


def latest_local_snapshot_on_or_before(
    directory: str | Path, target_date: dt.date | str
) -> Path | None:
    """Return the newest ``YYYY-MM-DD.zip`` in *directory* with date <= *target_date*.

    Returns ``None`` if the directory does not exist or has no matching zip.
    """
    directory = Path(directory)
    if isinstance(target_date, str):
        target_date = dt.date.fromisoformat(target_date)
    if not directory.exists():
        return None

    candidates: list[tuple[dt.date, Path]] = []
    for p in directory.glob("*.zip"):
        try:
            d = dt.date.fromisoformat(p.stem)
        except ValueError:
            continue
        if d <= target_date:
            candidates.append((d, p))
    if not candidates:
        return None
    return max(candidates, key=lambda t: t[0])[1]


def resolve_static_gtfs(
    *,
    local_zip: str | Path | None = None,
    local_dir: str | Path | None = None,
    target_date: dt.date | str | None = None,
    s3_agency: str | None = None,
    s3_kwargs: dict[str, Any] | None = None,
) -> Path:
    """Resolve the static GTFS zip in effect for a replay run.

    Priority: an explicit *local_zip* wins outright; otherwise *local_dir* is
    searched for the newest dated snapshot <= *target_date* (the default,
    tested path); otherwise, if *s3_agency* is given and gtfs-eta's
    ``stoptimes.record.storage.static_gtfs`` module is importable, the matching S3
    snapshot is downloaded to a temp file and that path returned.

    Raises:
        FileNotFoundError: no matching snapshot found.
        ValueError: none of local_zip/local_dir/s3_agency given, or
            target_date missing when required.
        RuntimeError: s3_agency given but the gtfs-eta helper isn't importable.
    """
    if local_zip is not None:
        p = Path(local_zip)
        if not p.exists():
            raise FileNotFoundError(f"Static GTFS zip not found: {p}")
        return p

    if local_dir is not None:
        if target_date is None:
            raise ValueError("target_date is required when resolving from local_dir")
        found = latest_local_snapshot_on_or_before(local_dir, target_date)
        if found is None:
            raise FileNotFoundError(
                f"No static GTFS snapshot <= {target_date} found in {local_dir}"
            )
        return found

    if s3_agency is not None:
        if target_date is None:
            raise ValueError("target_date is required when resolving from S3")
        target_date_obj = (
            dt.date.fromisoformat(target_date) if isinstance(target_date, str) else target_date
        )
        try:
            from stoptimes.record.storage.static_gtfs import (
                get_snapshot,
                latest_snapshot_on_or_before,
            )
        except ImportError as exc:
            raise RuntimeError(
                "S3 static-GTFS resolution requires the optional gtfs-eta "
                "'rt_pipeline' package to be importable."
            ) from exc

        kwargs = s3_kwargs or {}
        snap_date = latest_snapshot_on_or_before(s3_agency, target_date_obj, **kwargs)
        if snap_date is None:
            raise FileNotFoundError(
                f"No S3 static GTFS snapshot <= {target_date_obj} for agency {s3_agency!r}"
            )
        content = get_snapshot(s3_agency, snap_date, **kwargs)
        tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        try:
            tmp.write(content)
        finally:
            tmp.close()
        return Path(tmp.name)

    raise ValueError("One of local_zip, local_dir, or s3_agency must be provided")
