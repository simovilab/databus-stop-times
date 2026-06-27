"""Predictor protocol + StopPrediction dataclass. Phase P3.

StopPrediction is the output unit of every Predictor. The engine (P5) collects
these, then calls to_sample() to produce a PredictionSample for persistence.
"""

from __future__ import annotations

from dataclasses import dataclass

from etaval.domain.models import PredictionSample


@dataclass(frozen=True)
class StopPrediction:
    """One predicted arrival at a stop, emitted by a :class:`Predictor`.

    Fields
    ------
    stop_id:
        Stop identifier matching ``Stop.stop_id``.
    stop_sequence:
        Stop sequence number matching ``Stop.stop_sequence``.
    predicted_arrival_epoch:
        Predicted Unix epoch (seconds) when the vehicle will arrive at the stop.
    made_at_epoch:
        Unix epoch (seconds) when this prediction was produced ("now" for the
        constant-speed baseline; ``header.timestamp`` for the GTFS-RT predictor).
    dist_remaining_m:
        Distance in metres from the vehicle's current position to the stop,
        if known. ``None`` when the predictor cannot compute it (e.g. raw
        GTFS-RT arrival times do not carry this).
    stops_remaining:
        Number of upcoming stops at or after this one (inclusive), if known.
    """

    stop_id: str
    stop_sequence: int
    predicted_arrival_epoch: float
    made_at_epoch: float
    dist_remaining_m: float | None = None
    stops_remaining: int | None = None

    def to_sample(
        self,
        trip_key: str,
        predictor_id: str,
        vehicle_progress_m_at_made: float | None = None,
    ) -> PredictionSample:
        """Build a :class:`~etaval.domain.models.PredictionSample` for this prediction.

        The engine (P5) calls this once per prediction to produce the row that is
        persisted to storage and later joined with ``ActualArrival`` by the scorer.

        Args:
            trip_key:
                Composite trip identifier (``TripInstance.trip_key``).
            predictor_id:
                Identifier of the predictor that produced this sample.
            vehicle_progress_m_at_made:
                Vehicle's ``progress_m`` at the time this prediction was made,
                or ``None`` if unavailable.

        Returns:
            A frozen :class:`PredictionSample` with all fields populated.
        """
        return PredictionSample(
            trip_key=trip_key,
            predictor_id=predictor_id,
            stop_id=self.stop_id,
            stop_sequence=self.stop_sequence,
            predicted_arrival_epoch=self.predicted_arrival_epoch,
            made_at_epoch=self.made_at_epoch,
            vehicle_progress_m_at_made=vehicle_progress_m_at_made,
            dist_remaining_m=self.dist_remaining_m,
            stops_remaining=self.stops_remaining,
        )
