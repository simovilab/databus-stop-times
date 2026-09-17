"""Segment ETA training-dataset builder for MBTA vehicle positions (roadmap 3.1,
generalized to MBTA).

MBTA's raw ``VehiclePositions`` already carry REAL GTFS ids (``trip_id``,
``route_id``, ``current_stop_sequence``, ``stop_id``) -- unlike bUCR's navsat
AVL feed, no trip inference / map-matching is needed to recover which route,
direction, and stop pattern a position belongs to. A ``trip_id`` maps
directly through the static GTFS ``trips`` -> ``shape_id`` and through
``stop_times`` -> the ordered ``(stop_id, stop_sequence)`` list.

This module is only the MBTA **resolver**: it maps each ``(trip_id,
service_date)`` instance to its GTFS shape/stops and hands the resulting
:class:`~feature_engineering.segment_core.TripInstance` objects to
:func:`~feature_engineering.segment_core.build_segments_from_instances`, which
owns the shared arrival-detection / per-segment feature+target emission / schema.
That shared core is identical to the one the bUCR resolver
(:mod:`feature_engineering.segment_dataset_builder`) uses, so both agencies emit
the **same** :data:`~feature_engineering.segment_core.OUTPUT_COLUMNS` schema and
feed the same trainers / leakage probe / metrics unchanged.

Pure/deterministic: no I/O, no network, no mutation of inputs. GTFS parsing from
``feature_engineering.bucr_gtfs`` (a generic loader despite its name -- it reads
any standard GTFS zip/dir, MBTA's included); map-matching from
``etaval.spatial.polyline``. Runs under ``uv run --group bucr`` (needs
``etaval``).
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional

import pandas as pd

from etaval.spatial.polyline import PolylinePoint, assign_stops_monotonic

from feature_engineering.bucr_gtfs import BucrGtfs, BucrGtfsError, build_shape_polyline, stops_for_trip
from feature_engineering.segment_core import (
    ARRIVAL_RADIUS_M,
    CLOSEST_APPROACH_FALLBACK_M,
    MAX_SEGMENT_SECONDS,
    OUTPUT_COLUMNS,
    TripInstance,
    build_segments_from_instances,
    resolve_agency_temporal,
)

__all__ = ["build_mbta_segment_dataset"]


def _normalize_gtfs_ids(gtfs: BucrGtfs) -> BucrGtfs:
    """Return a copy of ``gtfs`` with every join id column coerced to ``str``.

    Real-world GTFS csvs routinely load id columns as ``object`` dtype with a
    MIX of str/int python values (some rows numeric-looking, pandas' per-chunk
    type sniffing doesn't unify them) -- MBTA's static feed does this for
    ``stop_times.trip_id``/``stop_id``. A raw ``==`` comparison against a string
    then silently matches nothing. Normalize on a *copy* so lookups here (and
    the ``bucr_gtfs`` helpers called on this copy) are dtype-safe; the caller's
    ``gtfs`` object itself is never mutated.
    """
    return BucrGtfs(
        routes=gtfs.routes,
        trips=gtfs.trips.assign(
            trip_id=gtfs.trips["trip_id"].astype(str),
            route_id=gtfs.trips["route_id"].astype(str),
            shape_id=gtfs.trips["shape_id"].astype(str),
        ),
        stops=gtfs.stops.assign(stop_id=gtfs.stops["stop_id"].astype(str)),
        stop_times=gtfs.stop_times.assign(
            trip_id=gtfs.stop_times["trip_id"].astype(str),
            stop_id=gtfs.stop_times["stop_id"].astype(str),
        ),
        shapes=gtfs.shapes.assign(shape_id=gtfs.shapes["shape_id"].astype(str)),
    )


def _mbta_trip_instances(df: pd.DataFrame, gtfs: BucrGtfs) -> Iterator[TripInstance]:
    """Yield one :class:`TripInstance` per (trip_id, service_date), in key order.

    Each real GTFS ``trip_id`` is resolved to its shape polyline and ordered,
    along-shape-assigned stops; instances with an unknown ``trip_id``, an
    unbuildable polyline, or missing stop_times are skipped.
    """
    trips = gtfs.trips.drop_duplicates(subset="trip_id").set_index("trip_id")
    polyline_cache: Dict[str, Optional[List[PolylinePoint]]] = {}

    for (trip_id, _service_date), trip_vps in df.groupby(
        ["trip_id", "_service_date"], sort=True
    ):
        trip_id_str = str(trip_id)
        if trip_id_str not in trips.index:
            continue
        trip_row = trips.loc[trip_id_str]
        shape_id = str(trip_row["shape_id"])
        route_id = str(trip_row["route_id"])

        if shape_id not in polyline_cache:
            try:
                polyline_cache[shape_id] = build_shape_polyline(gtfs, shape_id)
            except BucrGtfsError:
                polyline_cache[shape_id] = None
        polyline = polyline_cache[shape_id]
        if not polyline:
            continue

        try:
            raw_stops = stops_for_trip(gtfs, trip_id_str)
        except BucrGtfsError:
            continue
        stops = assign_stops_monotonic(raw_stops, polyline)
        stops = [s for s in stops if s.progress_m is not None]
        if len(stops) < 2:
            continue

        # vehicle_id from the earliest fix BY TS (the core sorts trip_vps by ts
        # too, but resolves nothing here) -- for the rare instance served by two
        # vehicles this must match "first vehicle of the sorted trip", not the
        # first row in raw frame order.
        vehicle_id = trip_vps.sort_values("ts", kind="stable")["vehicle_id"].iloc[0]

        yield TripInstance(
            trip_id=trip_id_str,
            route_id=route_id,
            vehicle_id=vehicle_id,
            stops=stops,
            polyline=polyline,
            trip_vps=trip_vps,
        )


def build_mbta_segment_dataset(
    vps_df: pd.DataFrame,
    gtfs: BucrGtfs,
    *,
    arrival_radius_m: float = ARRIVAL_RADIUS_M,
    closest_fallback_m: float = CLOSEST_APPROACH_FALLBACK_M,
    max_segment_seconds: float = MAX_SEGMENT_SECONDS,
    agency: str = "mbta",
    tz_for_temporal: Optional[str] = None,
) -> pd.DataFrame:
    """Build the segment training dataset from raw MBTA VehiclePositions.

    Args:
        vps_df: Raw MBTA VP frame (e.g. from
            ``rt_pipeline.storage.s3_writer.read_vehicle_positions``) -- must
            carry ``trip_id``, ``vehicle_id``, ``ts``, ``lat``, ``lon`` (and
            optionally ``bearing``, ``speed``, ``current_status``). A trip
            *instance* is ``(trip_id, service_date)``: MBTA reuses one
            ``trip_id`` across every service date, so grouping by ``trip_id``
            alone would collapse several days into one instance (arrival
            detection resolves only the earliest date). The service date uses a
            3am local-time boundary (standard transit service day, DST-safe), so
            a run crossing local midnight stays one instance. Rows with a
            null/missing ``trip_id`` are dropped. NO trip inference is
            performed -- ``trip_id`` maps straight through ``gtfs``.
        gtfs: A :class:`~feature_engineering.bucr_gtfs.BucrGtfs` loaded via
            ``feature_engineering.bucr_gtfs.load_gtfs`` from the static GTFS
            snapshot in effect for this VP window.
        arrival_radius_m: "arrived" radius for the position-derived target.
        closest_fallback_m: closest-approach fallback when no fix lands
            inside ``arrival_radius_m``.
        max_segment_seconds: drop segments whose observed traversal exceeds
            this (a hidden layover / mis-paired arrival, not a real leg).
        agency: temporal-feature agency key (default ``"mbta"`` ->
            America/New_York / region US_MA).
        tz_for_temporal: explicit timezone override; defaults to the
            agency's.

    Returns:
        A DataFrame with :data:`segment_core.OUTPUT_COLUMNS`. One row per
        (trip instance x traversed segment k->k+1) where arrivals at both
        endpoints were detected, chronologically ordered, and the traversal is
        in ``(0, max_segment_seconds]``. Empty (with the right columns) when no
        segment qualifies. Column semantics mirror ``build_bucr_segment_dataset``
        exactly (both delegate to the same core); the only difference is
        provenance: ``route_id`` and ``shape_id`` come directly from the GTFS
        ``trips`` table keyed by the real ``trip_id``, not from parsing a
        synthetic id.

    Raises:
        TypeError: ``vps_df`` is not a DataFrame or ``gtfs`` wrong type.
        ValueError: required columns missing, or ``agency`` unknown.
    """
    if not isinstance(vps_df, pd.DataFrame):
        raise TypeError(f"vps_df must be a DataFrame, got {type(vps_df)!r}")
    if not isinstance(gtfs, BucrGtfs):
        raise TypeError(f"gtfs must be a BucrGtfs, got {type(gtfs)!r}")

    required = {"trip_id", "vehicle_id", "ts", "lat", "lon"}
    missing = [c for c in required if c not in vps_df.columns]
    if missing:
        raise ValueError(f"vps_df missing required columns: {missing}")

    temporal_tz, temporal_region = resolve_agency_temporal(agency, tz_for_temporal)

    if vps_df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = vps_df.copy(deep=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["trip_id"])
    df["trip_id"] = df["trip_id"].astype(str)
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Service date with a 3am local-time boundary (DST-safe via tz_convert) --
    # see the ``vps_df`` arg note for why a trip *instance* is
    # (trip_id, service_date), not trip_id alone.
    _local = df["ts"].dt.tz_convert(temporal_tz)
    df["_service_date"] = (_local - pd.Timedelta(hours=3)).dt.date

    gtfs = _normalize_gtfs_ids(gtfs)

    return build_segments_from_instances(
        _mbta_trip_instances(df, gtfs),
        arrival_radius_m=arrival_radius_m,
        closest_fallback_m=closest_fallback_m,
        max_segment_seconds=max_segment_seconds,
        temporal_tz=temporal_tz,
        temporal_region=temporal_region,
    )
