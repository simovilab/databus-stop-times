"""Segment ETA training-dataset builder for inferred BUCR trips (roadmap Step 5, 3.1).

Turns the *inferred* BUCR vehicle-position frame (the output of
``bucr_trip_inference.infer_bucr_trips`` -- one row per assigned, non-anomalous
point, carrying ``trip_id``/``route_id``/``stop_id``/``current_stop_sequence``)
into a **segment** training dataset: one row per
(trip instance x stop-to-stop segment), where the target is the *observed*
seconds a vehicle took to traverse from its arrival at stop k to its arrival
at stop k+1.

Why segment-based (vs. the stop-level ``dataset_builder.build_vp_training_dataset``):
a segment's traversal time is a stable, additive quantity -- stop-level ETA at
any horizon is just the sum of the predicted segment times ahead. See
``docs/BUCR_DATASET_PROMPT.md`` step 5 / roadmap 3.1.

This module is now only the BUCR **resolver**: it maps each inferred
``trip_id`` back to its route-direction candidate (ordered stops + polyline) and
hands the resulting :class:`~stoptimes.core.segments.TripInstance`
objects to :func:`~stoptimes.core.segments.build_segments_from_instances`,
which owns the shared arrival-detection / per-segment feature+target emission /
schema (identical across agencies -- MBTA uses the same core via
:mod:`stoptimes.train.datasets.mbta_segments`). The constants and ``OUTPUT_COLUMNS``
are re-exported here for backward compatibility with existing importers.

Arrival detection (in the core) reuses the same **50 m first-within-threshold,
else closest-approach<=200 m** rule as ``dataset_builder.find_actual_arrival_time``
so segment targets are position-derived (never schedule-derived -- BUCR's
offset timetable does not poison labels).

Pure/deterministic: no I/O, no network, no mutation of inputs. Runs under
``uv run --group bucr``.
"""

from __future__ import annotations

from typing import Dict, Iterator, Optional, Sequence, Tuple

import pandas as pd

from stoptimes.train.sources.bucr_gtfs import RouteDirectionCandidate
from stoptimes.core.segments import (
    ARRIVAL_RADIUS_M,
    CLOSEST_APPROACH_FALLBACK_M,
    MAX_SEGMENT_SECONDS,
    OUTPUT_COLUMNS,
    TripInstance,
    build_segments_from_instances,
    resolve_agency_temporal,
)

__all__ = [
    "OUTPUT_COLUMNS",
    "ARRIVAL_RADIUS_M",
    "CLOSEST_APPROACH_FALLBACK_M",
    "MAX_SEGMENT_SECONDS",
    "build_bucr_segment_dataset",
]


def _parse_trip_id(trip_id: str) -> Optional[Tuple[str, str, int, str]]:
    """Recover ``(vehicle_id, route_id, direction_id, shape_id)`` from a trip id.

    Inverse of ``bucr_trip_inference._make_trip_id``'s recipe
    ``bucr:{vehicle}:{route}:{direction}:{shape}:{start}`` -- none of vehicle
    (plate), route, or BUCR shape ids contain ``:``, so a plain 6-field split
    is unambiguous. Returns ``None`` for any id not matching the recipe (so an
    unexpected id skips the trip rather than crashing the build).
    """
    parts = str(trip_id).split(":")
    if len(parts) != 6 or parts[0] != "bucr":
        return None
    _, vehicle_id, route_id, direction_str, shape_id, _start = parts
    try:
        direction_id = int(direction_str)
    except ValueError:
        return None
    return vehicle_id, route_id, direction_id, shape_id


def _candidate_index(
    candidates: Sequence[RouteDirectionCandidate],
) -> Dict[Tuple[str, int, str], RouteDirectionCandidate]:
    """Index candidates by ``(route_id, direction_id, shape_id)`` for O(1) lookup."""
    return {(c.route_id, c.direction_id, c.shape_id): c for c in candidates}


def _bucr_trip_instances(
    df: pd.DataFrame,
    cand_by_key: Dict[Tuple[str, int, str], RouteDirectionCandidate],
) -> Iterator[TripInstance]:
    """Yield one :class:`TripInstance` per inferred BUCR trip, in trip-id order.

    Each ``trip_id`` is parsed back to its ``(vehicle, route, direction, shape)``
    and matched to a route-direction candidate for its ordered stops + polyline;
    unparseable ids and unmatched candidates are skipped.
    """
    for trip_id, trip_vps in df.groupby("trip_id", sort=True):
        parsed = _parse_trip_id(str(trip_id))
        if parsed is None:
            continue
        vehicle_id, route_id, direction_id, shape_id = parsed
        candidate = cand_by_key.get((route_id, direction_id, shape_id))
        if candidate is None:
            continue
        yield TripInstance(
            trip_id=trip_id,
            route_id=route_id,
            vehicle_id=vehicle_id,
            stops=list(candidate.stops),
            polyline=candidate.polyline,
            trip_vps=trip_vps,
        )


def build_bucr_segment_dataset(
    inferred_df: pd.DataFrame,
    candidates: Sequence[RouteDirectionCandidate],
    *,
    arrival_radius_m: float = ARRIVAL_RADIUS_M,
    closest_fallback_m: float = CLOSEST_APPROACH_FALLBACK_M,
    max_segment_seconds: float = MAX_SEGMENT_SECONDS,
    agency: str = "bucr",
    tz_for_temporal: Optional[str] = None,
) -> pd.DataFrame:
    """Build the segment training dataset from an inferred BUCR VP frame.

    Args:
        inferred_df: Output of ``bucr_trip_inference.infer_bucr_trips`` -- must
            carry ``trip_id``, ``vehicle_id``, ``ts``, ``lat``, ``lon``,
            ``speed``, ``current_status`` (and optionally ``bearing``). Each
            distinct ``trip_id`` is one trip instance.
        candidates: The same ``RouteDirectionCandidate`` list inference ran
            against; used to recover each trip's ordered stops (with
            ``progress_m`` along-shape) and its polyline (for cross-track).
        arrival_radius_m: "arrived" radius for the position-derived target.
        closest_fallback_m: closest-approach fallback when no fix lands inside
            ``arrival_radius_m``.
        max_segment_seconds: drop segments whose observed traversal exceeds this
            (a hidden layover / mis-paired arrival, not a real leg).
        agency: temporal-feature agency key (default ``"bucr"`` ->
            America/Costa_Rica / region CR).
        tz_for_temporal: explicit timezone override; defaults to the agency's.

    Returns:
        A DataFrame with :data:`OUTPUT_COLUMNS`. One row per
        (trip instance x traversed segment k->k+1) where arrivals at BOTH
        endpoints were detected, are chronologically ordered, and the traversal
        is in ``(0, max_segment_seconds]``. Empty (with the right columns) when
        no segment qualifies. Column semantics per segment k->k+1 (entry =
        arrival at stop k, destination = stop k+1):
          * ``stop_id``/``stop_sequence``/``stop_lat``/``stop_lon`` -> destination k+1
          * ``from_stop_id`` -> entry stop k
          * ``vp_ts``/``vp_lat``/``vp_lon``/``vp_bearing``/``vp_speed`` -> entry VP
          * ``distance_to_stop`` / ``shape_distance_to_stop`` -> segment along-shape length
          * ``distance_to_next_stop`` -> next segment's along-shape length (NaN if last)
          * ``shape_progress`` -> progress_m(k) / shape total (fraction done at entry)
          * ``progress_ratio`` -> k / (n_stops-1) (stop-index fraction)
          * ``cross_track_error`` -> entry VP's cross-track to the assigned shape
          * ``stops_ahead`` -> stops remaining after the destination (stops-remaining)
          * ``is_at_stop`` -> entry VP reported ``STOPPED_AT``
          * ``time_to_arrival_seconds`` -> observed arrival(k+1) - arrival(k) seconds
          * ``actual_arrival`` -> arrival(k+1) ts

    Raises:
        TypeError: ``inferred_df`` is not a DataFrame or ``candidates`` wrong type.
        ValueError: required columns missing, or ``agency`` unknown.
    """
    if not isinstance(inferred_df, pd.DataFrame):
        raise TypeError(f"inferred_df must be a DataFrame, got {type(inferred_df)!r}")
    if not isinstance(candidates, (list, tuple)):
        raise TypeError(f"candidates must be a list/tuple, got {type(candidates)!r}")

    required = {"trip_id", "vehicle_id", "ts", "lat", "lon"}
    missing = [c for c in required if c not in inferred_df.columns]
    if missing:
        raise ValueError(f"inferred_df missing required columns: {missing}")

    temporal_tz, temporal_region = resolve_agency_temporal(agency, tz_for_temporal)

    if inferred_df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    cand_by_key = _candidate_index(candidates)

    df = inferred_df.copy(deep=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")

    return build_segments_from_instances(
        _bucr_trip_instances(df, cand_by_key),
        arrival_radius_m=arrival_radius_m,
        closest_fallback_m=closest_fallback_m,
        max_segment_seconds=max_segment_seconds,
        temporal_tz=temporal_tz,
        temporal_region=temporal_region,
    )
