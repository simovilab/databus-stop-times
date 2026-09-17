"""Pluggable interfaces (typing.Protocol). Phase P0.

- Predictor.predict(trip, upcoming_stops, vehicle_state, trip_updates) -> list[StopPrediction]
- GroundTruthDetector.detect(trip, position_history, stops) -> list[ActualArrival]
- FeedAdapter.poll() -> parsed entities
- Storage.save_samples / save_arrivals / save_results / load_results

Note: StopPrediction is defined in stoptimes.val.predictors.base (Phase P3). It is referenced
here only under TYPE_CHECKING so that the interfaces module has zero runtime dependency
on the predictors package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from stoptimes.core.domain.models import (
        ActualArrival,
        PredictionSample,
        Stop,
        TripInstance,
        ValidationResult,
        VehicleState,
    )
    from stoptimes.val.predictors.base import StopPrediction


class Predictor(Protocol):
    """Pluggable arrival predictor — emits per-stop ETAs for a given vehicle state."""

    predictor_id: str

    #: Declarative opt-in for receiving position history.  When ``True`` the
    #: engine calls ``predict(..., position_history=enriched_history)``;
    #: otherwise it omits the keyword entirely.  Defaults to ``False`` so
    #: predictors that don't declare it keep the plain 4-arg call signature.
    accepts_position_history: bool = False

    def predict(
        self,
        trip: TripInstance,
        upcoming_stops: list[Stop],
        vehicle_state: VehicleState,
        trip_updates: Any,
    ) -> list[StopPrediction]: ...


class GroundTruthDetector(Protocol):
    """Pluggable ground-truth arrival detector — fires ActualArrivals from position history."""

    method: str

    def detect(
        self,
        trip: TripInstance,
        position_history: list[VehicleState],
        stops: list[Stop],
    ) -> list[ActualArrival]: ...


class FeedAdapter(Protocol):
    """Pluggable feed adapter — polls one JSON GTFS-RT endpoint asynchronously."""

    async def poll(self) -> Any:
        """Poll the feed, parse, and return entities. Exact return type varies by adapter."""
        ...


class Storage(Protocol):
    """Persistent storage contract for all pipeline outputs."""

    def save_samples(self, samples: list[PredictionSample]) -> None: ...

    def save_arrivals(self, arrivals: list[ActualArrival]) -> None: ...

    def save_results(self, results: list[ValidationResult]) -> None: ...

    def load_results(
        self,
        predictor_id: str | None = None,
    ) -> list[ValidationResult]: ...
