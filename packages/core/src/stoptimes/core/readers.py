"""Unified readers: GTFS-RT protobuf/JSON, archive parquet, AVL CSV with column mapping.

Every public `read_*` function returns a `pandas.DataFrame` shaped like
`contracts.data.OBSERVATION_VP_SCHEMA` / `OBSERVATION_TU_SCHEMA` (validated
with `contracts.data.validate_frame`), plus two convenience columns every
reader here adds beyond those schemas: `source_id` and `kind` (validate_frame
ignores columns it doesn't know about, so this is additive, not a schema
break). Timestamps are always coerced to `datetime64[ns, UTC]`
(`contracts.data.coerce_utc_ns`).

Readers, one function each:

- `read_gtfsrt_json` / `read_gtfsrt_protobuf` — a live URL or a local file,
  for vehicle positions or trip updates. `google.transit.gtfs_realtime_pb2`
  is imported lazily; a missing install raises `ImportError` naming the
  `stoptimes-core[pb]` extra.
- `read_archive_parquet` — the Hive-partitioned parquet `stoptimes.record`
  writes (`year=/month=/day=[/route_id=]`), from a local directory or
  `s3://` prefix (`fsspec`, lazy, optional — see docs/notes/p2-t1.md, it is
  not yet a declared extra). `pyarrow` is imported lazily, naming
  `stoptimes-core[parquet]` on failure. Date/route filters are pushed down
  to the file *list* (parsed from the partition path), never read to filter.
- `read_avl_csv` — any AVL export describable as a `ColumnMapping`: which
  CSV column holds the timestamp/lat/lon/vehicle/trip/route, the timestamp's
  strptime format (or `None` to let pandas infer it) and the source
  timezone naive timestamps are interpreted in. This is how a non-GTFS-RT
  agency (bUCR's navsat CSVs) gets in — as configuration, not code.
- `read_static` — wraps `stoptimes.core.gtfs.static.GtfsStaticLoader` (never
  rewritten here) and reports which GTFS static tables are present via
  `contracts.data.StaticFeed`, without asking the loader to parse a table it
  doesn't support (`calendar`/`calendar_dates` are detected from the zip's
  member list only).
- `read_observations` — the common entry point: resolves a reader `format`
  (mirrors `contracts.source.RealtimeFeed.format`, plus `archive_parquet`/
  `avl_csv`) and a location to the matching function above.
- `read_observations_for_source` — same, but resolves the URL/format/headers
  from a `contracts.source.Source`'s matching `RealtimeFeed` instead of
  taking them explicitly (`docs/CONTRACTS.md` §1's "a `Source` + a
  location" entry point).

Known limitations (see docs/notes/p2-t1.md for the full writeup):
- `bearing` is never populated from GTFS-RT JSON/protobuf — extracted here
  directly rather than through `core.realtime.gtfsrt_json.parse_vp_feed`,
  because that function's `VehicleState` doesn't carry it (nor do
  `TripUpdateRecord`/`StopTimeUpdate` carry delay/uncertainty/schedule
  relationship/revenue for trip updates) — this module parses the feed
  dict directly instead, reusing `gtfsrt_json`'s private accessor helpers.
- AVL CSV rows with no `trip` column map to `trip_id=""` — trip inference
  from an AVL trace (bUCR) is T2's job, not this reader's.
"""

from __future__ import annotations

import json
import re
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict
from stoptimes.core.contracts.adapters import observation_from_tu_row, observation_from_vp_row
from stoptimes.core.contracts.data import (
    OBSERVATION_TU_SCHEMA,
    OBSERVATION_VP_SCHEMA,
    UTC_DTYPE,
    ColumnSpec,
    ObservationKind,
    StaticFeed,
    TableSchema,
    coerce_utc_ns,
)
from stoptimes.core.contracts.prediction import StopScheduleRelationship, TripScheduleRelationship
from stoptimes.core.gtfs.static import GtfsStaticLoader
from stoptimes.core.realtime import gtfsrt_json

__all__ = [
    "ColumnMapping",
    "ReaderFormat",
    "read_gtfsrt_json",
    "read_gtfsrt_protobuf",
    "read_archive_parquet",
    "read_avl_csv",
    "read_static",
    "read_observations",
    "read_observations_for_source",
]


# ---------------------------------------------------------------------------
# Reader format enum + common entry point
# ---------------------------------------------------------------------------


class ReaderFormat(str, Enum):
    """Which reader `read_observations` should dispatch to."""

    GTFSRT_JSON = "gtfsrt_json"
    GTFSRT_PB = "gtfsrt_pb"
    ARCHIVE_PARQUET = "archive_parquet"
    AVL_CSV = "avl_csv"


def read_observations(
    location: str,
    *,
    format: str | ReaderFormat,
    kind: ObservationKind,
    source_id: str,
    feed_name: str | None = None,
    headers: Mapping[str, str] | None = None,
    mapping: ColumnMapping | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    routes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Resolve `format` + `location` to the matching reader and call it.

    `format` mirrors `contracts.source.RealtimeFeed.format` (`json`/`pb`,
    accepted here as `gtfsrt_json`/`gtfsrt_pb`) plus the two formats a
    `RealtimeFeed` has no room for: `archive_parquet` and `avl_csv`.
    """
    fmt = ReaderFormat(format)
    if fmt is ReaderFormat.GTFSRT_JSON:
        return read_gtfsrt_json(
            location, kind, source_id=source_id, feed_name=feed_name, headers=headers
        )
    if fmt is ReaderFormat.GTFSRT_PB:
        return read_gtfsrt_protobuf(
            location, kind, source_id=source_id, feed_name=feed_name, headers=headers
        )
    if fmt is ReaderFormat.ARCHIVE_PARQUET:
        return read_archive_parquet(
            location,
            kind,
            source_id=source_id,
            start_date=start_date,
            end_date=end_date,
            routes=routes,
        )
    if fmt is ReaderFormat.AVL_CSV:
        if mapping is None:
            raise ValueError(
                "avl_csv format requires a ColumnMapping (see ColumnMapping.from_file)"
            )
        return read_avl_csv(location, mapping, source_id=source_id, feed_name=feed_name)
    # pragma: no cover - ReaderFormat(format) above already raises for an unknown value
    raise ValueError(f"unknown reader format: {format!r}")


def read_observations_for_source(
    source: Any,
    kind: Any,
    *,
    location: str | None = None,
    mapping: ColumnMapping | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    routes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Resolve a `contracts.source.Source` + `FeedKind` to the matching feed
    and read it — `location` overrides the feed's own `url` (e.g. to read a
    committed fixture instead of the live feed) without changing anything
    else about how it's read.
    """
    from stoptimes.core.contracts.source import FeedFormat, FeedKind

    feed = next((f for f in source.realtime_feeds if f.kind == kind), None)
    if feed is None:
        kind_desc = kind.value if hasattr(kind, "value") else kind
        raise ValueError(f"source {source.id!r} has no {kind_desc} feed")
    obs_kind = (
        ObservationKind.VEHICLE_POSITION
        if feed.kind == FeedKind.VEHICLE_POSITIONS
        else ObservationKind.TRIP_UPDATE
    )
    reader_format = (
        ReaderFormat.GTFSRT_JSON if feed.format == FeedFormat.JSON else ReaderFormat.GTFSRT_PB
    )
    return read_observations(
        location or feed.url,
        format=reader_format,
        kind=obs_kind,
        source_id=source.id,
        headers=feed.headers,
        mapping=mapping,
        start_date=start_date,
        end_date=end_date,
        routes=routes,
    )


# ---------------------------------------------------------------------------
# Fetch helpers (local path or http(s) URL)
# ---------------------------------------------------------------------------


def _is_url(location: str) -> bool:
    return location.startswith("http://") or location.startswith("https://")


def _fetch_bytes(location: str, headers: Mapping[str, str] | None = None) -> bytes:
    if _is_url(location):
        response = httpx.get(location, headers=dict(headers or {}), timeout=30.0)
        response.raise_for_status()
        return response.content
    return Path(location).read_bytes()


def _fetch_json(location: str, headers: Mapping[str, str] | None = None) -> dict[str, Any]:
    raw = _fetch_bytes(location, headers)
    return json.loads(raw.decode("utf-8"))


# ---------------------------------------------------------------------------
# Empty-frame construction (so every reader agrees on dtypes when there are
# no rows — validate_frame must pass on an empty result too).
# ---------------------------------------------------------------------------


def _column_dtype(col: ColumnSpec) -> str:
    if col.dtype == UTC_DTYPE:
        return UTC_DTYPE
    if col.dtype == "string":
        return "object"
    if col.dtype == "int64":
        return "Int64"  # nullable — most of these columns are optional
    if col.dtype == "float64":
        return "float64"
    if col.dtype == "bool":
        return "bool"
    return "object"  # pragma: no cover - no other dtype strings exist today


def _empty_frame(schema: TableSchema) -> pd.DataFrame:
    return pd.DataFrame({c.name: pd.Series(dtype=_column_dtype(c)) for c in schema.columns})


def _finalize(df: pd.DataFrame, *, source_id: str, kind: ObservationKind) -> pd.DataFrame:
    """Attach the two enrichment columns every reader adds."""
    df = df.copy()
    df["source_id"] = source_id
    df["kind"] = kind.value
    return df


# ---------------------------------------------------------------------------
# GTFS-RT JSON / protobuf — vehicle positions
# ---------------------------------------------------------------------------


def _vp_rows_from_feed(feed: dict[str, Any], header_ts: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entity in feed.get("entity") or []:
        veh = entity.get("vehicle")
        if not isinstance(veh, dict):
            continue
        trip = veh.get("trip") or {}
        trip_id = gtfsrt_json._opt_str(trip, "trip_id")
        if not trip_id:
            continue
        pos = veh.get("position") or {}
        lat = gtfsrt_json._opt_float(pos, "latitude")
        lon = gtfsrt_json._opt_float(pos, "longitude")
        if lat is None or lon is None:
            continue
        epoch = gtfsrt_json._opt_float(veh, "timestamp")
        if epoch is None:
            epoch = header_ts
        vehicle_id = gtfsrt_json._opt_str(veh, "vehicle", "id") or gtfsrt_json._opt_str(
            entity, "id"
        )
        rows.append(
            {
                "vehicle_id": vehicle_id,
                "ts": epoch,
                "lat": lat,
                "lon": lon,
                "bearing": gtfsrt_json._opt_float(pos, "bearing"),
                "speed": gtfsrt_json._opt_float(pos, "speed"),
                "trip_id": trip_id,
                "route_id": gtfsrt_json._opt_str(trip, "route_id"),
                "current_stop_sequence": gtfsrt_json._opt_int(veh, "current_stop_sequence"),
                "current_status": gtfsrt_json._opt_str(veh, "current_status"),
                "stop_id": gtfsrt_json._opt_str(veh, "stop_id"),
            }
        )
    return rows


def _finalize_vp_frame(rows: list[dict[str, Any]], *, feed_name: str | None) -> pd.DataFrame:
    if not rows:
        return _empty_frame(OBSERVATION_VP_SCHEMA)
    raw = pd.DataFrame(rows)
    out = pd.DataFrame(index=raw.index)
    out["feed_name"] = feed_name
    out["vehicle_id"] = raw["vehicle_id"]
    out["ts"] = coerce_utc_ns(pd.to_datetime(raw["ts"].astype("float64"), unit="s", utc=True))
    out["lat"] = raw["lat"].astype("float64")
    out["lon"] = raw["lon"].astype("float64")
    out["bearing"] = raw["bearing"].astype("float64")
    out["speed"] = raw["speed"].astype("float64")
    out["trip_id"] = raw["trip_id"]
    out["route_id"] = raw["route_id"]
    out["current_stop_sequence"] = raw["current_stop_sequence"].astype("Int64")
    out["current_status"] = raw["current_status"]
    out["stop_id"] = raw["stop_id"]
    now = pd.Series([pd.Timestamp.now(tz="UTC")] * len(out), index=out.index)
    out["ingested_at"] = coerce_utc_ns(now)
    return out.reindex(columns=OBSERVATION_VP_SCHEMA.column_names)


# ---------------------------------------------------------------------------
# GTFS-RT JSON / protobuf — trip updates
# ---------------------------------------------------------------------------


def _norm_enum(value: str | None, enum_cls: type) -> str | None:
    if value is None:
        return None
    try:
        return enum_cls(value).value
    except ValueError:
        return None


def _tu_rows_from_feed(feed: dict[str, Any], header_ts: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entity in feed.get("entity") or []:
        tu = entity.get("trip_update")
        if not isinstance(tu, dict):
            continue
        trip = tu.get("trip") or {}
        trip_id = gtfsrt_json._opt_str(trip, "trip_id")
        if not trip_id:
            continue
        vehicle = tu.get("vehicle") or {}
        ts_val = gtfsrt_json._opt_float(tu, "timestamp")
        if ts_val is None:
            ts_val = header_ts
        trip_schedule_relationship = gtfsrt_json._opt_str(trip, "schedule_relationship")
        route_id = gtfsrt_json._opt_str(trip, "route_id")
        direction_id = gtfsrt_json._opt_int(trip, "direction_id")
        start_date_val = gtfsrt_json._opt_str(trip, "start_date")
        vehicle_id = gtfsrt_json._opt_str(vehicle, "id")
        revenue = tu.get("revenue") if isinstance(tu.get("revenue"), bool) else None

        for stu in tu.get("stop_time_update") or []:
            if not isinstance(stu, dict):
                continue
            stop_id = gtfsrt_json._opt_str(stu, "stop_id")
            stop_sequence = gtfsrt_json._opt_int(stu, "stop_sequence")
            if stop_id is None or stop_sequence is None:
                continue
            arrival = stu.get("arrival") or {}
            departure = stu.get("departure") or {}
            rows.append(
                {
                    "route_id": route_id,
                    "trip_id": trip_id,
                    "start_date": start_date_val,
                    "direction_id": direction_id,
                    "vehicle_id": vehicle_id,
                    "trip_schedule_relationship": trip_schedule_relationship,
                    "revenue": revenue,
                    "stop_id": stop_id,
                    "stop_sequence": stop_sequence,
                    "arrival_time_epoch": gtfsrt_json._opt_float(arrival, "time"),
                    "arrival_delay": gtfsrt_json._opt_int(arrival, "delay"),
                    "arrival_uncertainty": gtfsrt_json._opt_int(arrival, "uncertainty"),
                    "departure_time_epoch": gtfsrt_json._opt_float(departure, "time"),
                    "departure_delay": gtfsrt_json._opt_int(departure, "delay"),
                    "departure_uncertainty": gtfsrt_json._opt_int(departure, "uncertainty"),
                    "schedule_relationship": gtfsrt_json._opt_str(stu, "schedule_relationship"),
                    "ts_epoch": ts_val,
                }
            )
    return rows


def _attach_revenue_column(out: pd.DataFrame, revenue: pd.Series) -> pd.DataFrame:
    """Add `revenue` only when it can be a real (non-nullable) `bool` column.

    `revenue` is optional per `OBSERVATION_TU_SCHEMA`; omitting the column
    entirely (rather than a nullable dtype `validate_frame` would flag as
    "not exactly bool") is a valid pass, and never fabricates a value for a
    row the feed didn't report one for.
    """
    if revenue.isna().all():
        return out
    if revenue.isna().any():
        out["revenue"] = revenue  # left as object; documented limitation, see module docstring
    else:
        out["revenue"] = revenue.astype("bool")
    return out


def _finalize_tu_frame(rows: list[dict[str, Any]], *, feed_name: str | None) -> pd.DataFrame:
    if not rows:
        return _empty_frame(OBSERVATION_TU_SCHEMA)
    raw = pd.DataFrame(rows)
    out = pd.DataFrame(index=raw.index)
    out["feed_name"] = feed_name
    out["route_id"] = raw["route_id"]
    out["trip_id"] = raw["trip_id"]
    out["start_date"] = raw["start_date"]
    out["direction_id"] = raw["direction_id"].astype("Int64")
    out["vehicle_id"] = raw["vehicle_id"]
    out["trip_schedule_relationship"] = raw["trip_schedule_relationship"]
    out = _attach_revenue_column(out, raw["revenue"])
    out["stop_id"] = raw["stop_id"]
    out["stop_sequence"] = raw["stop_sequence"].astype("Int64")
    arrival_epoch = raw["arrival_time_epoch"].astype("float64")
    out["arrival_time"] = coerce_utc_ns(pd.to_datetime(arrival_epoch, unit="s", utc=True))
    out["arrival_delay"] = raw["arrival_delay"].astype("Int64")
    out["arrival_uncertainty"] = raw["arrival_uncertainty"].astype("Int64")
    departure_epoch = raw["departure_time_epoch"].astype("float64")
    out["departure_time"] = coerce_utc_ns(pd.to_datetime(departure_epoch, unit="s", utc=True))
    out["departure_delay"] = raw["departure_delay"].astype("Int64")
    out["departure_uncertainty"] = raw["departure_uncertainty"].astype("Int64")
    out["ts"] = coerce_utc_ns(pd.to_datetime(raw["ts_epoch"].astype("float64"), unit="s", utc=True))
    now = pd.Series([pd.Timestamp.now(tz="UTC")] * len(out), index=out.index)
    out["ingested_at"] = coerce_utc_ns(now)
    out["schedule_relationship"] = raw["schedule_relationship"]
    out["schedule_relationship_norm"] = out["schedule_relationship"].map(
        lambda v: _norm_enum(v, StopScheduleRelationship)
    )
    out["trip_schedule_relationship_norm"] = out["trip_schedule_relationship"].map(
        lambda v: _norm_enum(v, TripScheduleRelationship)
    )
    # Column order: the schema's own columns (revenue included only if _attach_revenue_column
    # added it — OBSERVATION_TU_SCHEMA lists it as optional), plus the enrichment columns this
    # reader adds beyond the schema: the raw `schedule_relationship` (not itself a schema
    # column, despite Observation carrying it — see docs/CONTRACTS.md open question 3) and its
    # normalized form, and `trip_schedule_relationship`'s normalized form.
    ordered = [
        c for c in OBSERVATION_TU_SCHEMA.column_names if c != "revenue" or "revenue" in out.columns
    ]
    ordered += [
        "schedule_relationship",
        "schedule_relationship_norm",
        "trip_schedule_relationship_norm",
    ]
    return out.reindex(columns=ordered)


def _observations_from_feed_dict(
    feed: dict[str, Any], kind: ObservationKind, *, source_id: str, feed_name: str | None
) -> pd.DataFrame:
    header_ts = gtfsrt_json._opt_float(feed.get("header") or {}, "timestamp") or 0.0
    if kind is ObservationKind.VEHICLE_POSITION:
        df = _finalize_vp_frame(_vp_rows_from_feed(feed, header_ts), feed_name=feed_name)
    else:
        df = _finalize_tu_frame(_tu_rows_from_feed(feed, header_ts), feed_name=feed_name)
    return _finalize(df, source_id=source_id, kind=kind)


def read_gtfsrt_json(
    location: str,
    kind: ObservationKind,
    *,
    source_id: str,
    feed_name: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Read a GTFS-RT JSON feed (URL or local file) into an Observation frame."""
    feed = _fetch_json(location, headers)
    return _observations_from_feed_dict(feed, kind, source_id=source_id, feed_name=feed_name)


def _import_gtfs_realtime_pb2():
    try:
        from google.transit import gtfs_realtime_pb2
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "reading GTFS-RT protobuf feeds needs the 'pb' extra: "
            "uv sync --extra pb  (or pip install 'stoptimes-core[pb]')"
        ) from exc
    return gtfs_realtime_pb2


def read_gtfsrt_protobuf(
    location: str,
    kind: ObservationKind,
    *,
    source_id: str,
    feed_name: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Read a GTFS-RT protobuf feed (URL or local file) into an Observation frame.

    Parses with `google.transit.gtfs_realtime_pb2.FeedMessage`, converts to
    the same nested-dict shape `core.realtime.gtfsrt_json` reads (via
    `google.protobuf.json_format.MessageToDict(preserving_proto_field_name=True)`),
    then shares this module's dict-walking with `read_gtfsrt_json`.
    """
    pb2 = _import_gtfs_realtime_pb2()
    from google.protobuf.json_format import MessageToDict

    raw = _fetch_bytes(location, headers)
    message = pb2.FeedMessage()
    try:
        message.ParseFromString(raw)
    except Exception as exc:
        raise ValueError(f"could not parse GTFS-RT protobuf feed from {location!r}: {exc}") from exc
    feed = MessageToDict(message, preserving_proto_field_name=True)
    return _observations_from_feed_dict(feed, kind, source_id=source_id, feed_name=feed_name)


# ---------------------------------------------------------------------------
# Archive parquet — local or s3://
# ---------------------------------------------------------------------------

_PARTITION_RE = re.compile(r"year=(\d+)/month=(\d+)/day=(\d+)(?:/route_id=([^/]+))?")


def _import_pyarrow():
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "reading archive parquet needs the 'parquet' extra: "
            "uv sync --extra parquet  (or pip install 'stoptimes-core[parquet]')"
        ) from exc


def _import_fsspec():
    try:
        import fsspec
    except ImportError as exc:  # pragma: no cover - fsspec is optional, not yet a declared extra
        raise ImportError(
            "reading an s3:// archive needs fsspec (and a protocol package such as "
            "s3fs) installed — not yet a declared stoptimes-core extra, see "
            "docs/notes/p2-t1.md: pip install fsspec s3fs"
        ) from exc
    return fsspec


def _parse_date_arg(value: str | None) -> date | None:
    if value is None:
        return None
    digits = value.replace("-", "")
    return date(int(digits[0:4]), int(digits[4:6]), int(digits[6:8]))


def _parse_partition(path: str) -> dict[str, Any] | None:
    match = _PARTITION_RE.search(path.replace("\\", "/"))
    if not match:
        return None
    year, month, day, route_id = match.groups()
    return {"date": date(int(year), int(month), int(day)), "route_id": route_id}


def _date_in_range(d: date, start: date | None, end: date | None) -> bool:
    if start is not None and d < start:
        return False
    if end is not None and d > end:
        return False
    return True


def _list_archive_files(
    location: str,
    *,
    start_date: str | None,
    end_date: str | None,
    routes: Sequence[str] | None,
) -> list[str]:
    start = _parse_date_arg(start_date)
    end = _parse_date_arg(end_date)
    routes_set = set(routes) if routes else None

    if location.startswith("s3://"):
        fs = _import_fsspec().filesystem("s3")
        candidates = [f"s3://{p}" for p in fs.find(location) if p.endswith(".parquet")]
    else:
        candidates = [str(p) for p in Path(location).rglob("*.parquet")]

    matched: list[str] = []
    for candidate in candidates:
        parts = _parse_partition(candidate)
        if parts is None:
            continue
        if not _date_in_range(parts["date"], start, end):
            continue
        route_id = parts["route_id"]
        if routes_set is not None and route_id is not None and route_id not in routes_set:
            continue
        matched.append(candidate)
    return sorted(matched)


def _read_parquet_file(path: str) -> pd.DataFrame:
    if path.startswith("s3://"):
        fs = _import_fsspec().filesystem("s3")
        with fs.open(path, "rb") as fh:
            return pd.read_parquet(fh)
    return pd.read_parquet(path)


def read_archive_parquet(
    location: str,
    kind: ObservationKind,
    *,
    source_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
    routes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read a `stoptimes.record` Hive-partitioned parquet archive.

    `location` is a local directory or an `s3://` prefix holding
    `year=/month=/day=[/route_id=]/*.parquet` files (`OBSERVATION_VP_PARTITION`
    / `OBSERVATION_TU_PARTITION`). `start_date`/`end_date` (`YYYY-MM-DD` or
    `YYYYMMDD`) and `routes` are pushed down to the file *list* — parsed from
    each candidate's partition path — before any file is opened; `routes` is
    also applied to the loaded rows (trip-update archives have no
    `route_id` partition to prune by).

    Rows are converted through `contracts.adapters.observation_from_vp_row`/
    `observation_from_tu_row` (the same semantics `stoptimes.record`'s writer
    and this reader agree on), so archive column names never need to be
    imported from `stoptimes.record` directly (docs/PLAN.md rule 1).
    """
    is_vp = kind is ObservationKind.VEHICLE_POSITION
    schema = OBSERVATION_VP_SCHEMA if is_vp else OBSERVATION_TU_SCHEMA
    files = _list_archive_files(location, start_date=start_date, end_date=end_date, routes=routes)
    if not files:
        return _finalize(_empty_frame(schema), source_id=source_id, kind=kind)

    _import_pyarrow()
    frames = [_read_parquet_file(f) for f in files]
    raw = pd.concat(frames, ignore_index=True)
    if routes and "route_id" in raw.columns:
        raw = raw[raw["route_id"].isin(set(routes))].reset_index(drop=True)

    adapter = observation_from_vp_row if is_vp else observation_from_tu_row
    observations = [adapter(row, source_id) for row in raw.to_dict("records")]
    if not observations:
        return _finalize(_empty_frame(schema), source_id=source_id, kind=kind)

    out = pd.DataFrame([o.model_dump() for o in observations]).reindex(columns=schema.column_names)
    for col in schema.columns:
        if col.dtype == UTC_DTYPE:
            out[col.name] = coerce_utc_ns(out[col.name])
        elif col.dtype == "int64":
            out[col.name] = out[col.name].astype("Int64")
    return _finalize(out, source_id=source_id, kind=kind)


# ---------------------------------------------------------------------------
# AVL CSV with a user-supplied column mapping
# ---------------------------------------------------------------------------


class ColumnMapping(BaseModel):
    """Which CSV column supplies each field an AVL export can offer.

    `trip`/`route` are optional — many AVL sources (bUCR's navsat export)
    have neither; a `read_avl_csv` row from such a mapping gets
    `trip_id=""` (real trip inference is T2's job) and `route_id=None`.
    """

    model_config = ConfigDict(extra="forbid")

    timestamp: str
    lat: str
    lon: str
    vehicle: str | None = None
    trip: str | None = None
    route: str | None = None
    timestamp_format: str | None = None
    """A `strftime`/`strptime` format string; `None` lets pandas infer it."""
    source_timezone: str = "UTC"
    """IANA timezone a *timezone-naive* timestamp column is interpreted in
    before converting to UTC. Ignored if the column already carries an
    offset."""

    @classmethod
    def from_file(cls, path: str | Path) -> ColumnMapping:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data or {})


def read_avl_csv(
    location: str,
    mapping: ColumnMapping,
    *,
    source_id: str,
    feed_name: str | None = None,
) -> pd.DataFrame:
    """Read an AVL CSV export into a vehicle_position Observation frame.

    Rows with an unparseable/missing latitude or longitude are dropped (an
    `Observation` requires both); rows with an unparseable timestamp are
    kept with `ts = NaT` rather than dropped, so a caller can still see them
    in `data inspect`'s "usable timestamp share".
    """
    df = pd.read_csv(location)
    required = {"timestamp": mapping.timestamp, "lat": mapping.lat, "lon": mapping.lon}
    missing = [csv_col for field, csv_col in required.items() if csv_col not in df.columns]
    if missing:
        raise ValueError(f"AVL CSV {location!r} is missing mapped column(s): {missing}")

    if df.empty:
        empty = _empty_frame(OBSERVATION_VP_SCHEMA)
        return _finalize(empty, source_id=source_id, kind=ObservationKind.VEHICLE_POSITION)

    out = pd.DataFrame(index=df.index)
    out["feed_name"] = feed_name
    out["vehicle_id"] = df[mapping.vehicle] if mapping.vehicle else None

    if mapping.timestamp_format:
        parsed = pd.to_datetime(
            df[mapping.timestamp], format=mapping.timestamp_format, errors="coerce"
        )
    else:
        parsed = pd.to_datetime(df[mapping.timestamp], errors="coerce")
    if getattr(parsed.dt, "tz", None) is None:
        tz = ZoneInfo(mapping.source_timezone)
        parsed = parsed.dt.tz_localize(tz, nonexistent="NaT", ambiguous="NaT").dt.tz_convert("UTC")
    else:
        parsed = parsed.dt.tz_convert("UTC")
    out["ts"] = parsed

    out["lat"] = pd.to_numeric(df[mapping.lat], errors="coerce")
    out["lon"] = pd.to_numeric(df[mapping.lon], errors="coerce")
    out["bearing"] = pd.Series(float("nan"), index=out.index)
    out["speed"] = pd.Series(float("nan"), index=out.index)
    out["trip_id"] = df[mapping.trip] if mapping.trip else ""
    out["route_id"] = df[mapping.route] if mapping.route else None
    out["current_stop_sequence"] = pd.array([pd.NA] * len(out), dtype="Int64")
    out["current_status"] = None
    out["stop_id"] = None

    out = out.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    out["ingested_at"] = coerce_utc_ns(pd.Series([pd.NaT] * len(out), index=out.index))
    out["ts"] = coerce_utc_ns(out["ts"])
    out = out.reindex(columns=OBSERVATION_VP_SCHEMA.column_names)
    return _finalize(out, source_id=source_id, kind=ObservationKind.VEHICLE_POSITION)


# ---------------------------------------------------------------------------
# GTFS static — wraps GtfsStaticLoader, never rewrites it
# ---------------------------------------------------------------------------

_STATIC_TABLE_FILES: tuple[tuple[str, str], ...] = (
    ("stops", "stops.txt"),
    ("trips", "trips.txt"),
    ("stop_times", "stop_times.txt"),
    ("shapes", "shapes.txt"),
    ("routes", "routes.txt"),
    ("calendar", "calendar.txt"),
    ("calendar_dates", "calendar_dates.txt"),
)


def _materialize_local_zip(location: str, headers: Mapping[str, str] | None) -> tuple[Path, bool]:
    """Return (local_path, is_temporary)."""
    if _is_url(location):
        data = _fetch_bytes(location, headers)
        tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        try:
            tmp.write(data)
        finally:
            tmp.close()
        return Path(tmp.name), True
    return Path(location), False


def read_static(
    location: str,
    source_id: str,
    *,
    headers: Mapping[str, str] | None = None,
) -> tuple[GtfsStaticLoader | None, StaticFeed]:
    """Load a GTFS static feed (local `.zip` path or URL) and report which
    tables it has.

    Inspects the zip's member list directly (never asks `GtfsStaticLoader`
    to parse `calendar`/`calendar_dates`, which it doesn't support) to build
    an accurate `StaticFeed`. If a *required* table (`stops`/`trips`/
    `stop_times`) is missing, the loader is never constructed (it would
    raise `KeyError` opening a member that isn't in the zip) — the returned
    loader is `None` and `feed.missing_required()` names what's absent.
    """
    path, is_temp = _materialize_local_zip(location, headers)
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
        present = [table for table, filename in _STATIC_TABLE_FILES if filename in names]
        feed = StaticFeed(source_id=source_id, tables=present)
        if feed.missing_required():
            return None, feed
        loader = GtfsStaticLoader(path)
        return loader, feed
    finally:
        if is_temp:
            path.unlink(missing_ok=True)
