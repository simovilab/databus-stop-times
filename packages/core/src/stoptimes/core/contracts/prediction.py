"""Contract: prediction output.

What every serving pipeline (T4) produces and every predictor result (T6
val, existing `stoptimes.core.domain.models.PredictionSample`) can be
expressed as: `StopTimePrediction` (one stop of one trip) grouped into a
`TripUpdatePrediction` (one GTFS-RT TripUpdate entity's worth). Both know how
to render themselves as a plain dict shaped exactly like the GTFS-RT
TripUpdate / StopTimeUpdate JSON message — the same snake_case field names
`stoptimes.core.realtime.gtfsrt_json.parse_tu_feed` already reads, just
written instead of parsed. No protobuf dependency; a caller that wants
protobuf wraps this dict.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _require_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware (UTC)")
    if value.utcoffset() != timezone.utc.utcoffset(None):
        raise ValueError(f"{field_name} must be UTC")
    return value


class StopScheduleRelationship(str, Enum):
    """GTFS-RT `StopTimeUpdate.schedule_relationship`."""

    SCHEDULED = "SCHEDULED"
    SKIPPED = "SKIPPED"
    NO_DATA = "NO_DATA"
    UNSCHEDULED = "UNSCHEDULED"


class TripScheduleRelationship(str, Enum):
    """GTFS-RT `TripDescriptor.schedule_relationship`."""

    SCHEDULED = "SCHEDULED"
    ADDED = "ADDED"
    UNSCHEDULED = "UNSCHEDULED"
    CANCELED = "CANCELED"
    DUPLICATED = "DUPLICATED"
    DELETED = "DELETED"


class StopTimePrediction(BaseModel):
    """One predicted arrival/departure at one stop of one trip instance."""

    model_config = ConfigDict(extra="forbid")

    trip_id: str
    route_id: str | None = None
    direction_id: int | None = None
    start_date: str
    """YYYYMMDD."""
    stop_id: str
    stop_sequence: int
    vehicle_id: str | None = None

    predicted_arrival_time: datetime | None = None
    predicted_departure_time: datetime | None = None
    delay_s: float | None = None
    """Predicted arrival minus scheduled arrival, seconds (may be negative)."""
    uncertainty_s: float | None = None
    """GTFS-RT `uncertainty`: expected error of the prediction, in seconds. Not
    a confidence interval width — the spec leaves the exact statistic to the
    producer; document the producer's choice in the model card."""
    horizon_s: float | None = None
    """Seconds between when this prediction was made and the predicted event —
    how far ahead the prediction reaches, for horizon-bucketed scoring."""

    schedule_relationship: StopScheduleRelationship = StopScheduleRelationship.SCHEDULED

    model_id: str
    artifact_id: str | None = None
    generated_at: datetime
    """When this prediction was produced (UTC) — the pipeline's `ctx.now`, not `datetime.now()`."""

    @field_validator("predicted_arrival_time", "predicted_departure_time", "generated_at")
    @classmethod
    def _utc(cls, value: datetime | None, info: Any) -> datetime | None:
        if value is None:
            return None
        return _require_utc(value, info.field_name)

    @model_validator(mode="after")
    def _at_least_one_time(self) -> "StopTimePrediction":
        if (
            self.schedule_relationship != StopScheduleRelationship.SKIPPED
            and self.predicted_arrival_time is None
            and self.predicted_departure_time is None
        ):
            raise ValueError("prediction needs predicted_arrival_time and/or predicted_departure_time")
        return self

    def to_stop_time_update(self) -> dict[str, Any]:
        """Render as a GTFS-RT `StopTimeUpdate` JSON-style dict."""
        stu: dict[str, Any] = {"stop_sequence": self.stop_sequence, "stop_id": self.stop_id}
        if self.predicted_arrival_time is not None:
            arrival: dict[str, Any] = {"time": int(self.predicted_arrival_time.timestamp())}
            if self.delay_s is not None:
                arrival["delay"] = int(round(self.delay_s))
            if self.uncertainty_s is not None:
                arrival["uncertainty"] = int(round(self.uncertainty_s))
            stu["arrival"] = arrival
        if self.predicted_departure_time is not None:
            departure: dict[str, Any] = {"time": int(self.predicted_departure_time.timestamp())}
            if self.delay_s is not None:
                departure["delay"] = int(round(self.delay_s))
            if self.uncertainty_s is not None:
                departure["uncertainty"] = int(round(self.uncertainty_s))
            stu["departure"] = departure
        if self.schedule_relationship != StopScheduleRelationship.SCHEDULED:
            stu["schedule_relationship"] = self.schedule_relationship.value
        return stu


class TripUpdatePrediction(BaseModel):
    """One trip's worth of `StopTimePrediction`s — one GTFS-RT TripUpdate entity."""

    model_config = ConfigDict(extra="forbid")

    trip_id: str
    route_id: str | None = None
    direction_id: int | None = None
    start_date: str
    start_time: str | None = None
    vehicle_id: str | None = None
    schedule_relationship: TripScheduleRelationship = TripScheduleRelationship.SCHEDULED
    stop_time_updates: list[StopTimePrediction] = Field(min_length=1)
    generated_at: datetime

    @field_validator("generated_at")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return _require_utc(value, "generated_at")

    @model_validator(mode="after")
    def _stop_updates_belong_to_this_trip(self) -> "TripUpdatePrediction":
        for stu in self.stop_time_updates:
            if stu.trip_id != self.trip_id or stu.start_date != self.start_date:
                raise ValueError(
                    "stop_time_updates must share this trip's trip_id and start_date "
                    f"(got trip_id={stu.trip_id!r}, start_date={stu.start_date!r})"
                )
        return self

    def to_trip_update_dict(self) -> dict[str, Any]:
        """Render as a full GTFS-RT `TripUpdate` entity, JSON-style dict."""
        trip: dict[str, Any] = {"trip_id": self.trip_id, "start_date": self.start_date}
        if self.route_id is not None:
            trip["route_id"] = self.route_id
        if self.direction_id is not None:
            trip["direction_id"] = self.direction_id
        if self.start_time is not None:
            trip["start_time"] = self.start_time
        if self.schedule_relationship != TripScheduleRelationship.SCHEDULED:
            trip["schedule_relationship"] = self.schedule_relationship.value

        out: dict[str, Any] = {
            "trip": trip,
            "stop_time_update": [stu.to_stop_time_update() for stu in self.stop_time_updates],
            "timestamp": int(self.generated_at.timestamp()),
        }
        if self.vehicle_id is not None:
            out["vehicle"] = {"id": self.vehicle_id}
        return out
