"""Pure functions converting existing structures into contracts.

No behavior change to the modules being adapted — every function here reads
an existing shape and returns a contract instance. Kept in one place so a
Phase 2 track can find "how do I get a `Source`/`Observation`/... from what I
already have" without re-deriving the mapping.

Direction discipline: this module lives in `stoptimes.core`, so it may import
`stoptimes.core.domain` (same package) but must NOT import from
`stoptimes.record`, `stoptimes.val`, `stoptimes.serve`, `stoptimes.models` or
`stoptimes.train` (core is upstream of all of them — docs/PLAN.md rule 1).
Where an adapter's input is "a row from the record archive", it takes a
plain `Mapping`/`pandas.Series` shaped like that archive's documented
columns (see docs/CONTRACTS.md), not an import of the record package.

`legacy _meta.json dict -> ArtifactMetadata` is NOT here — that adapter
belongs to C3 (`stoptimes models migrate`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from stoptimes.core.contracts.data import Arrival, ArrivalMethod, Observation, ObservationKind
from stoptimes.core.contracts.prediction import StopTimePrediction
from stoptimes.core.contracts.source import (
    Archive,
    ArchiveKind,
    FeedFormat,
    FeedKind,
    GroundTruthDetector,
    RealtimeFeed,
    Source,
)
from stoptimes.core.domain.models import ActualArrival, PredictionSample

# ---------------------------------------------------------------------------
# source: val's flat source YAML -> Source
# ---------------------------------------------------------------------------


def source_from_val_yaml(data: Mapping[str, Any]) -> Source:
    """Adapt a val source config dict (as loaded from
    `packages/val/.../config/sources/{name}.yaml`, one `SourceConfig` per
    `stoptimes.core.domain.models`) into a `Source`.

    The val shape is flat and always publishes exactly one vehicle-positions
    and one trip-updates feed over JSON, so the mapping is total — every
    field in the val YAML has a home here.
    """
    poll_interval_s = float(data["poll_interval_s"])
    headers = dict(data.get("headers") or {})
    feeds = [
        RealtimeFeed(
            kind=FeedKind.VEHICLE_POSITIONS,
            url=data["vehicle_positions_url"],
            format=FeedFormat.JSON,
            poll_interval_s=poll_interval_s,
            headers=headers,
        ),
        RealtimeFeed(
            kind=FeedKind.TRIP_UPDATES,
            url=data["trip_updates_url"],
            format=FeedFormat.JSON,
            poll_interval_s=poll_interval_s,
            headers=headers,
        ),
    ]
    return Source(
        id=str(data["name"]),
        name=str(data["name"]),
        gtfs_static_url=data["gtfs_static_url"],
        realtime_feeds=feeds,
        detector=GroundTruthDetector(
            arrival_radius_m=float(data.get("arrival_radius_m", 40.0)),
            dwell_s=float(data.get("dwell_s", 0.0)),
            shape_threshold_m=float(data.get("shape_threshold_m", 30.0)),
        ),
    )


def source_from_archive_uri(source: Source, uri: str) -> Source:
    """Return a copy of *source* with its archive location set.

    `uri` starting with `s3://` sets `kind="s3"`; anything else is treated as
    a local path. Immutable: does not mutate `source`."""
    kind = ArchiveKind.S3 if uri.startswith("s3://") else ArchiveKind.LOCAL
    return source.model_copy(update={"archive": Archive(kind=kind, uri=uri)})


# ---------------------------------------------------------------------------
# data: record archive rows -> Observation / Arrival
# ---------------------------------------------------------------------------

# Column names below intentionally mirror stoptimes.record.storage.schema.DATA_COLUMNS
# and .tu_schema.DATA_COLUMNS (see docs/CONTRACTS.md) without importing that module.


def observation_from_vp_row(row: Mapping[str, Any], source_id: str) -> Observation:
    """Adapt one row of the record VehiclePositions archive to an `Observation`."""
    return Observation(
        kind=ObservationKind.VEHICLE_POSITION,
        source_id=source_id,
        feed_name=row.get("feed_name"),
        trip_id=row["trip_id"],
        route_id=row.get("route_id"),
        vehicle_id=row.get("vehicle_id"),
        ts=_opt_ts(row["ts"]),
        ingested_at=_opt_ts(row.get("ingested_at")),
        lat=float(row["lat"]),
        lon=float(row["lon"]),
        bearing=_opt_float(row.get("bearing")),
        speed=_opt_float(row.get("speed")),
        current_status=row.get("current_status"),
        current_stop_sequence=_opt_int(row.get("current_stop_sequence")),
        stop_id=row.get("stop_id"),
    )


def observation_from_tu_row(row: Mapping[str, Any], source_id: str) -> Observation:
    """Adapt one row of the record TripUpdates archive to an `Observation`."""
    return Observation(
        kind=ObservationKind.TRIP_UPDATE,
        source_id=source_id,
        feed_name=row.get("feed_name"),
        trip_id=row["trip_id"],
        route_id=row.get("route_id"),
        direction_id=_opt_int(row.get("direction_id")),
        start_date=row.get("start_date"),
        vehicle_id=row.get("vehicle_id"),
        ts=_opt_ts(row["ts"]),
        ingested_at=_opt_ts(row.get("ingested_at")),
        stop_id=row["stop_id"],
        stop_sequence=int(row["stop_sequence"]),
        arrival_time=_opt_ts(row.get("arrival_time")),
        arrival_delay=_opt_int(row.get("arrival_delay")),
        arrival_uncertainty=_opt_int(row.get("arrival_uncertainty")),
        departure_time=_opt_ts(row.get("departure_time")),
        departure_delay=_opt_int(row.get("departure_delay")),
        departure_uncertainty=_opt_int(row.get("departure_uncertainty")),
        schedule_relationship=row.get("stu_schedule_relationship"),
        trip_schedule_relationship=row.get("trip_schedule_relationship"),
        revenue=row.get("revenue"),
    )


def arrival_from_actual_arrival(
    actual: ActualArrival,
    *,
    source_id: str,
    start_date: str,
    route_id: str | None = None,
    vehicle_id: str | None = None,
) -> Arrival:
    """Adapt a `stoptimes.core.domain.models.ActualArrival` (epoch-seconds
    float) into an `Arrival` (UTC timestamp), filling in the fields
    `ActualArrival` doesn't carry (it identifies a trip by `trip_key`, not by
    source/start_date/route/vehicle)."""
    try:
        method = ArrivalMethod(actual.method)
    except ValueError:
        method = ArrivalMethod.MAP_MATCHED
    return Arrival(
        source_id=source_id,
        trip_id=actual.trip_key,
        start_date=start_date,
        route_id=route_id,
        vehicle_id=vehicle_id,
        stop_id=actual.stop_id,
        stop_sequence=actual.stop_sequence,
        arrival_time=pd.Timestamp(actual.actual_arrival_epoch, unit="s", tz="UTC"),
        method=method,
    )


def _opt_float(value: Any) -> float | None:
    return None if value is None or (isinstance(value, float) and pd.isna(value)) else float(value)


def _opt_int(value: Any) -> int | None:
    return None if value is None or (isinstance(value, float) and pd.isna(value)) else int(value)


def _opt_ts(value: Any) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value if value.tzinfo is not None else value.tz_localize("UTC")
    return pd.Timestamp(value, tz="UTC")


# ---------------------------------------------------------------------------
# prediction: PredictionSample -> StopTimePrediction
# ---------------------------------------------------------------------------


def stop_time_prediction_from_sample(
    sample: PredictionSample,
    *,
    trip_id: str,
    start_date: str,
    route_id: str | None = None,
    vehicle_id: str | None = None,
) -> StopTimePrediction:
    """Adapt a `stoptimes.core.domain.models.PredictionSample` (epoch-seconds
    floats, keyed by `trip_key`) into a `StopTimePrediction` (UTC datetimes,
    keyed by GTFS identifiers). `PredictionSample.trip_key` packs
    `source:trip_id:vehicle_id:start_date`, so the identifiers it doesn't
    carry directly are passed in explicitly."""
    return StopTimePrediction(
        trip_id=trip_id,
        route_id=route_id,
        start_date=start_date,
        stop_id=sample.stop_id,
        stop_sequence=sample.stop_sequence,
        vehicle_id=vehicle_id,
        predicted_arrival_time=pd.Timestamp(sample.predicted_arrival_epoch, unit="s", tz="UTC").to_pydatetime(),
        horizon_s=None,
        model_id=sample.predictor_id,
        generated_at=pd.Timestamp(sample.made_at_epoch, unit="s", tz="UTC").to_pydatetime(),
    )
