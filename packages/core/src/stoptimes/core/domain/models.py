"""Domain model (Pydantic v2). Phase P0.

Defines the immutable data carried through the pipeline:
- SourceConfig    : agency feed URLs + poll interval + detector params (loaded from YAML)
- TripInstance    : the unit of validation, identified from VehiclePositions
- Stop            : (stop_id, stop_sequence, lat, lon, progress_m)
- VehicleState    : a single position-history sample for a TripInstance
- PredictionSample: one predictor x stop x tick arrival prediction
- ActualArrival   : a detected ground-truth arrival at a stop
- ValidationResult: derived once actual is known (error_s, horizon_s, ...)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SourceConfig(BaseModel):
    """Agency feed URLs + detector parameters, loaded from a YAML source config."""

    name: str
    gtfs_static_url: str
    vehicle_positions_url: str
    trip_updates_url: str
    poll_interval_s: int
    headers: dict[str, str] = {}
    # Optional auth — kept loose so different auth schemes can be added later
    auth: dict[str, str] | None = None
    # Ground-truth detector parameters
    arrival_radius_m: float = 40.0
    dwell_s: float = 0.0
    shape_threshold_m: float = 30.0


class TripInstance(BaseModel):
    """The unit of validation — one vehicle trip tracked end-to-end."""

    model_config = ConfigDict(frozen=True)

    source: str
    trip_id: str
    vehicle_id: str
    route_id: str
    direction_id: int
    shape_id: str
    start_date: str  # YYYYMMDD
    started_at: float  # epoch seconds

    @property
    def trip_key(self) -> str:
        """Stable composite identifier for this trip instance."""
        return f"{self.source}:{self.trip_id}:{self.vehicle_id}:{self.start_date}"


class Stop(BaseModel):
    """A scheduled stop along a trip's route."""

    model_config = ConfigDict(frozen=True)

    stop_id: str
    stop_sequence: int
    lat: float
    lon: float
    progress_m: float | None = None  # assigned after spatial projection (P2)


class VehicleState(BaseModel):
    """One position-history sample for a TripInstance."""

    model_config = ConfigDict(frozen=True)

    epoch: float
    lat: float
    lon: float
    progress_m: float | None = None
    speed: float | None = None
    current_status: str | None = None
    current_stop_sequence: int | None = None


class PredictionSample(BaseModel):
    """One predictor × stop × tick arrival prediction."""

    model_config = ConfigDict(frozen=True)

    trip_key: str
    predictor_id: str
    stop_id: str
    stop_sequence: int
    predicted_arrival_epoch: float
    made_at_epoch: float
    vehicle_progress_m_at_made: float | None = None
    dist_remaining_m: float | None = None
    stops_remaining: int | None = None


class ActualArrival(BaseModel):
    """A detected ground-truth arrival at a stop."""

    model_config = ConfigDict(frozen=True)

    trip_key: str
    stop_id: str
    stop_sequence: int
    actual_arrival_epoch: float
    method: str


class ValidationResult(BaseModel):
    """Derived once actual arrival is known: joins prediction to ground truth."""

    model_config = ConfigDict(frozen=True)

    trip_key: str
    predictor_id: str
    stop_id: str
    stop_sequence: int
    route_id: str | None = None
    error_s: float  # predicted_arrival_epoch - actual_arrival_epoch
    horizon_s: float  # actual_arrival_epoch - made_at_epoch  (lead time when prediction was made)
    dist_remaining_m: float | None = None
    stops_remaining: int | None = None
    predicted_arrival_epoch: float
    actual_arrival_epoch: float
    made_at_epoch: float

    @classmethod
    def from_sample_and_arrival(
        cls,
        sample: PredictionSample,
        arrival: ActualArrival,
        route_id: str | None = None,
    ) -> ValidationResult:
        """Construct a ValidationResult by joining a PredictionSample and ActualArrival."""
        return cls(
            trip_key=sample.trip_key,
            predictor_id=sample.predictor_id,
            stop_id=sample.stop_id,
            stop_sequence=sample.stop_sequence,
            route_id=route_id,
            error_s=sample.predicted_arrival_epoch - arrival.actual_arrival_epoch,
            horizon_s=arrival.actual_arrival_epoch - sample.made_at_epoch,
            dist_remaining_m=sample.dist_remaining_m,
            stops_remaining=sample.stops_remaining,
            predicted_arrival_epoch=sample.predicted_arrival_epoch,
            actual_arrival_epoch=arrival.actual_arrival_epoch,
            made_at_epoch=sample.made_at_epoch,
        )
