"""Scorer. Phase P5.

Join PredictionSample <-> ActualArrival on (trip_key, stop_id) -> ValidationResult.

A primary_method filter (default "shape_distance") ensures that when multiple
detectors produce arrivals for the same stop, only one method drives scoring.

Two entry points:
- Scorer.score(): batch join for report generation.
- Scorer.score_stop(): incremental join called by the engine when a stop first resolves.
"""

from __future__ import annotations

from stoptimes.core.domain.models import ActualArrival, PredictionSample, ValidationResult


class Scorer:
    """Join PredictionSamples to ActualArrivals to produce ValidationResults.

    The ``primary_method`` parameter selects which detector method is used as
    ground truth (default ``"shape_distance"``).  Arrivals from other methods are
    ignored during scoring so that multiple detectors never create duplicate results.
    """

    def __init__(self, primary_method: str = "shape_distance") -> None:
        self.primary_method = primary_method

    # ------------------------------------------------------------------
    # Batch scoring
    # ------------------------------------------------------------------

    def score(
        self,
        samples: list[PredictionSample],
        arrivals: list[ActualArrival],
        route_id_map: dict[str, str] | None = None,
    ) -> list[ValidationResult]:
        """Join all samples to primary-method arrivals on ``(trip_key, stop_id)``.

        Args:
            samples:      All collected PredictionSamples.
            arrivals:     All detected ActualArrivals (any method).
            route_id_map: Optional ``trip_key → route_id`` for enrichment.

        Returns:
            One ValidationResult per (sample, primary-method arrival) match.
            Samples that have no matching primary-method arrival are silently
            skipped.
        """
        # Index primary-method arrivals by (trip_key, stop_id).
        arrival_index: dict[tuple[str, str], ActualArrival] = {}
        for arrival in arrivals:
            if arrival.method == self.primary_method:
                key = (arrival.trip_key, arrival.stop_id)
                arrival_index[key] = arrival  # last one wins (dedup handled upstream)

        rmap = route_id_map or {}
        results: list[ValidationResult] = []
        for sample in samples:
            key = (sample.trip_key, sample.stop_id)
            arrival = arrival_index.get(key)
            if arrival is None:
                continue
            route_id = rmap.get(sample.trip_key)
            results.append(
                ValidationResult.from_sample_and_arrival(sample, arrival, route_id)
            )
        return results

    # ------------------------------------------------------------------
    # Incremental scoring (called by engine when a stop newly resolves)
    # ------------------------------------------------------------------

    def score_stop(
        self,
        trip_key: str,
        stop_id: str,
        arrival: ActualArrival,
        samples: list[PredictionSample],
        route_id: str | None = None,
    ) -> list[ValidationResult]:
        """Score all stored samples for ``(trip_key, stop_id)`` against a new arrival.

        The engine calls this the first time a primary-method arrival fires for
        a given stop so that predictions made earlier in the trip are scored
        immediately and persisted.

        Args:
            trip_key:  Composite trip identifier (``TripInstance.trip_key``).
            stop_id:   Stop whose arrival just resolved.
            arrival:   The newly-detected :class:`ActualArrival`.
            samples:   All samples collected so far (will be filtered in O(n)).
            route_id:  Optional route_id for enrichment.

        Returns:
            One ValidationResult per matching sample.
        """
        results: list[ValidationResult] = []
        for sample in samples:
            if sample.trip_key == trip_key and sample.stop_id == stop_id:
                results.append(
                    ValidationResult.from_sample_and_arrival(sample, arrival, route_id)
                )
        return results
