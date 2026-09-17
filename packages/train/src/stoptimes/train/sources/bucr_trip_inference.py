"""Trip / route / stop inference for BUCR navsat traces (roadmap Step 3).

BUCR (navsat AVL) hands over none of GTFS-RT's structural columns
(``route_id``/``trip_id``/``stop_id``/``current_stop_sequence``). This module
infers all four from a *cleaned, adapted* vehicle-position trace (the output
of ``navsat_cleaning.clean_navsat_trace`` -> ``navsat_adapter
.navsat_to_vehicle_positions``) plus the static-GTFS candidates produced by
``bucr_gtfs.route_direction_candidates``. Pipeline order is:

    clean_navsat_trace(raw) -> navsat_to_vehicle_positions(cleaned)
        -> infer_bucr_trips(vp_df, candidates)

Per (vehicle_id x service-day) this module:

  1. Projects the trace onto every candidate (route+direction+shape-variant)
     polyline via ``etaval.spatial.polyline.project_point_to_polyline`` and
     scores each candidate by a high-percentile (``DIVERGENCE_SCORE_PERCENTILE``)
     cross-track error + a monotonic-progress fraction (how much of the
     trace advances along-shape rather than backward). Assignment is
     HIERARCHICAL: a DIRECTION (route_id, direction_id) is chosen first by
     the best eligible direction's score; if the best two DIRECTIONS are
     within ``AMBIGUITY_MARGIN_M``, the whole group is PARKED (not
     force-assigned) -- this is genuine cross-direction ambiguity.
     Same-direction shape VARIANTS (e.g. BUCR's two origin stops crossed
     with its optional "milla" detour -- variation is two-dimensional, not
     just a single detour) are then resolved within the chosen direction by
     DISTINGUISHING-STOP visitation (did the trace pass within
     ``DISTINGUISHING_STOP_RADIUS_M`` of each candidate's own stops that
     aren't shared by every variant in the direction, and avoid its rivals'?
     see ``bucr_trip_scoring.resolve_variant_by_stop_coverage``), never by
     cross-track score -- same-direction variants routinely score within the
     ambiguity margin of each other since they share almost their whole
     path, and that must not cause parking.
  2. Segments the assigned trace into discrete trip instances wherever
     along-shape progress resets (end-of-shape -> layover -> restart) or a
     long time/space gap occurs.
  3. Within each instance, assigns ``current_stop_sequence``/``stop_id`` from
     the candidate's ``assign_stops_monotonic``-derived stops, loop-back-safe
     (a later point can never regress to an earlier stop within one
     instance). Synthesizes a stable, deterministic ``trip_id``.
  4. Rejects ("anomaly") points whose cross-track error to the assigned shape
     exceeds ``ANOMALY_CROSS_TRACK_M`` -- "the bus went somewhere else."
  5. Emits the canonical VP output frame
     (``rt_source.OUTPUT_COLUMNS`` + ``route_id``) plus a frozen
     ``InferenceStats`` with raw counts. The Step-4 quality report (match
     rate vs. timetable, spot checks) is explicitly OUT OF SCOPE here.

Candidate scoring, trip-instance segmentation, and stop assignment are
implemented in the sibling module ``bucr_trip_scoring.py`` (kept separate
purely to stay under this repo's ~500-line-per-module target); their named
thresholds are re-exported here for convenience and documented in full below.

Pure/deterministic: no I/O, no network, no mutation of inputs. All spatial
math (haversine, projection, stop assignment) is reused from
``etaval.spatial.polyline`` -- this module only orchestrates scoring,
segmentation, and row assembly. Mirrors the error-handling style of
``navsat_adapter.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from feature_engineering.bucr_gtfs import RouteDirectionCandidate
from feature_engineering.bucr_trip_scoring import (
    AMBIGUITY_MARGIN_M,
    DISTINGUISHING_STOP_RADIUS_M,
    DIVERGENCE_SCORE_PERCENTILE,
    IDLE_DISPLACEMENT_M,
    MAX_GAP_METERS,
    MAX_GAP_SECONDS,
    MIN_INSTANCE_DISTANCE_FRACTION,
    MIN_INSTANCE_DURATION_SECONDS,
    MIN_INSTANCE_POINTS,
    MIN_MONOTONIC_PROGRESS_FRACTION,
    MIN_PROGRESS_BEFORE_RESET_FRACTION,
    MONOTONIC_JITTER_TOLERANCE_M,
    PRESEGMENT_DWELL_RADIUS_M,
    PRESEGMENT_DWELL_SECONDS,
    PRESEGMENT_MAX_GAP_SECONDS,
    RESET_NEAR_END_FRACTION,
    RESET_NEAR_START_FRACTION,
    SCORING_OUTLIER_EXCLUSION_M,
    assign_stop_sequences,
    instance_index_ranges,
    instance_passes_min_thresholds,
    presegment_boundaries,
    score_candidates,
    segment_boundaries,
    select_best_candidate,
)
__all__ = [
    "REQUIRED_VP_COLUMNS",
    "OUTPUT_COLUMNS",
    "AMBIGUITY_MARGIN_M",
    "MIN_MONOTONIC_PROGRESS_FRACTION",
    "ANOMALY_CROSS_TRACK_M",
    "DISTINGUISHING_STOP_RADIUS_M",
    "DIVERGENCE_SCORE_PERCENTILE",
    "SCORING_OUTLIER_EXCLUSION_M",
    "MAX_GAP_SECONDS",
    "MAX_GAP_METERS",
    "RESET_NEAR_END_FRACTION",
    "RESET_NEAR_START_FRACTION",
    "MONOTONIC_JITTER_TOLERANCE_M",
    "IDLE_DISPLACEMENT_M",
    "PRESEGMENT_DWELL_SECONDS",
    "PRESEGMENT_DWELL_RADIUS_M",
    "PRESEGMENT_MAX_GAP_SECONDS",
    "MIN_PROGRESS_BEFORE_RESET_FRACTION",
    "MIN_INSTANCE_POINTS",
    "MIN_INSTANCE_DURATION_SECONDS",
    "MIN_INSTANCE_DISTANCE_FRACTION",
    "InferenceStats",
    "infer_bucr_trips",
]

# Columns this module requires on the input VP frame. Matches
# ``navsat_adapter.OUTPUT_COLUMNS`` -- the expected upstream producer.
REQUIRED_VP_COLUMNS: List[str] = [
    "vehicle_id",
    "ts",
    "lat",
    "lon",
    "speed",
    "current_status",
]

# Output columns: mirrors ``rt_source.OUTPUT_COLUMNS`` (LEGACY_COLUMNS +
# EXTRA_COLUMNS) exactly, plus ``route_id`` (BUCR has one route today but the
# schema stays general). Spelled out here rather than imported from
# ``feature_engineering.rt_source`` because that module transitively imports
# ``rt_pipeline.storage``, which is not an installed dependency under the
# ``bucr`` uv group (a pre-existing repo quirk affecting ``rt_source.py`` and
# ``dataset_builder.py`` too -- out of scope to fix here; see this module's
# test-run report for the two known-broken test files it causes).
OUTPUT_COLUMNS: List[str] = [
    "trip_id",
    "vehicle_id",
    "ts",
    "lat",
    "lon",
    "bearing",
    "speed",
    "current_stop_sequence",
    "current_status",
    "stop_id",
    "route_id",
]

# A point whose cross-track distance to its assigned shape exceeds this many
# metres is rejected as an anomaly ("the bus went somewhere else" -- a
# parking lot, an ad-hoc detour, a wrong-way street). Campus roads are
# narrow; a point that's on-route typically shows <20 m cross-track error
# even with GPS noise, so 150 m is generous enough to tolerate multipath
# error near buildings while still catching genuine off-route excursions.
# Lives here (not bucr_trip_scoring.py) because it's applied to the already
# -selected candidate, not used during candidate scoring itself.
# NOTE: Raised from 75m to 150m after Step 4 quality report showed 28% of
# points were being rejected as anomalies with 75m threshold (p75 cross-track
# error was 145m). The higher threshold better accommodates GPS noise and
# minor shape geometry mismatches while still catching genuine off-route
# excursions.
ANOMALY_CROSS_TRACK_M: float = 150.0


@dataclass(frozen=True)
class InferenceStats:
    """Deterministic, immutable summary of one ``infer_bucr_trips`` call.

    ``trips_per_direction_variant`` is a tuple (not a dict) of
    ``(route_id, direction_id, shape_id, trip_count)`` rows, sorted by that
    key, so the whole dataclass stays hashable/comparable and genuinely
    immutable (a ``dict`` field would defeat ``frozen=True``'s intent).

    ``points_dropped_as_noise``/``instances_dropped_as_noise`` count points
    and segmented instances that WERE successfully matched to a
    route/direction/shape (i.e. not parked, not anomaly-rejected) but whose
    resulting instance failed the minimum-instance filter
    (``bucr_trip_scoring.instance_passes_min_thresholds`` -- too few points,
    too short a duration, or too little along-shape distance covered to be a
    real trip). Kept separate from ``points_parked_ambiguous``/
    ``points_anomaly_rejected`` because the failure mode is different: those
    two mean "couldn't tell which route/shape," this one means "know exactly
    which shape, but this cluster of points is an idle dwell or a blip, not
    a trip."
    """

    points_in: int
    points_assigned: int
    points_parked_ambiguous: int
    points_anomaly_rejected: int
    points_dropped_as_noise: int
    trips_inferred: int
    instances_dropped_as_noise: int
    trips_per_direction_variant: Tuple[Tuple[str, int, str, int], ...]

    @property
    def points_accounted_for(self) -> int:
        return (
            self.points_assigned
            + self.points_parked_ambiguous
            + self.points_anomaly_rejected
            + self.points_dropped_as_noise
        )


def _missing_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in REQUIRED_VP_COLUMNS if c not in df.columns]


def _empty_stats() -> InferenceStats:
    return InferenceStats(
        points_in=0,
        points_assigned=0,
        points_parked_ambiguous=0,
        points_anomaly_rejected=0,
        points_dropped_as_noise=0,
        trips_inferred=0,
        instances_dropped_as_noise=0,
        trips_per_direction_variant=(),
    )


def _make_trip_id(
    vehicle_id: str, route_id: str, direction_id: int, shape_id: str, instance_start_ts: pd.Timestamp
) -> str:
    """Stable, deterministic, human-readable trip id.

    Recipe: ``bucr:{vehicle_id}:{route_id}:{direction_id}:{shape_id}:{start_ts}``
    where ``start_ts`` is the instance's first point's ``ts`` formatted as
    strict UTC ``YYYYMMDDTHHMMSSZ``. Collision-safe because a single vehicle
    cannot be in two places (two candidate shapes) at once, and cannot start
    two distinct trip instances in the same second (segmentation already
    guarantees instance start timestamps for one vehicle are strictly
    increasing) -- so the 5-tuple is unique per real-world trip instance, and
    re-running inference on the same input always reproduces the same id.
    """
    start = instance_start_ts.tz_convert("UTC") if instance_start_ts.tzinfo else instance_start_ts.tz_localize("UTC")
    start_str = start.strftime("%Y%m%dT%H%M%SZ")
    return f"bucr:{vehicle_id}:{route_id}:{direction_id}:{shape_id}:{start_str}"


def _process_coarse_segment(
    seg: pd.DataFrame, candidates: Sequence[RouteDirectionCandidate]
) -> Tuple[List[pd.DataFrame], int, int, int, int, Dict[Tuple[str, int, str], int]]:
    """Assign a direction+variant to ONE direction-agnostic coarse segment, then finish it.

    A coarse segment (produced by ``presegment_boundaries``) is mostly a single
    trip in a single direction, so scoring it against every candidate and
    picking the best one works as ``select_best_candidate`` intends -- unlike
    scoring a whole mixed-direction vehicle-day, which parks on ambiguity (see
    ``bucr_trip_scoring``'s pre-segmentation module note).

    Returns the same 6-tuple shape as :func:`_process_group`:
    (assigned_row_frames, n_parked, n_anomaly, n_noise_points,
    n_noise_instances, trips_per_variant_delta).
    """
    lats = seg["lat"].to_numpy(dtype=float)
    lons = seg["lon"].to_numpy(dtype=float)

    scores = score_candidates(lats, lons, candidates)
    best = select_best_candidate(scores, lats, lons)
    if best is None:
        return [], len(seg), 0, 0, 0, {}

    cross_track = best.cross_track_m
    progress = best.progress_m
    shape_total_m = best.candidate.polyline[-1].cum_m

    anomaly_mask = cross_track > ANOMALY_CROSS_TRACK_M
    n_anomaly = int(anomaly_mask.sum())
    keep_mask = ~anomaly_mask

    kept = seg.loc[keep_mask].reset_index(drop=True)
    kept_progress = progress[keep_mask]
    if kept.empty:
        return [], 0, n_anomaly, 0, 0, {}

    # A coarse segment can still hold >1 loop of the SAME direction (a dwell too
    # short to break, or two back-to-back same-direction trips), so the
    # progress-reset within-segment segmentation still runs -- now on a single,
    # already-assigned shape.
    boundary = segment_boundaries(
        kept["ts"],
        kept["lat"].to_numpy(),
        kept["lon"].to_numpy(),
        kept_progress,
        shape_total_m,
        current_status=kept["current_status"].to_numpy(),
    )
    ranges = instance_index_ranges(boundary)

    variant_key = (best.candidate.route_id, best.candidate.direction_id, best.candidate.shape_id)
    trips_delta: Dict[Tuple[str, int, str], int] = {}
    out_frames: List[pd.DataFrame] = []
    n_noise_points = 0
    n_noise_instances = 0

    for start, end in ranges:
        inst_progress = kept_progress[start:end]
        inst_ts = kept["ts"].iloc[start:end]

        if not instance_passes_min_thresholds(inst_ts, inst_progress, shape_total_m):
            n_noise_points += end - start
            n_noise_instances += 1
            continue

        inst = kept.iloc[start:end].reset_index(drop=True)
        seqs, ids = assign_stop_sequences(inst_progress, best.candidate.stops)

        trip_id = _make_trip_id(
            vehicle_id=str(inst["vehicle_id"].iloc[0]),
            route_id=best.candidate.route_id,
            direction_id=best.candidate.direction_id,
            shape_id=best.candidate.shape_id,
            instance_start_ts=inst["ts"].iloc[0],
        )

        out = pd.DataFrame(index=inst.index)
        out["trip_id"] = trip_id
        out["vehicle_id"] = inst["vehicle_id"]
        out["ts"] = inst["ts"]
        out["lat"] = inst["lat"]
        out["lon"] = inst["lon"]
        out["bearing"] = inst["bearing"] if "bearing" in inst.columns else np.nan
        out["speed"] = inst["speed"]
        out["current_stop_sequence"] = seqs
        out["current_status"] = inst["current_status"]
        out["stop_id"] = ids
        out["route_id"] = best.candidate.route_id
        out_frames.append(out)

        trips_delta[variant_key] = trips_delta.get(variant_key, 0) + 1

    return out_frames, 0, n_anomaly, n_noise_points, n_noise_instances, trips_delta


def _process_group(
    group: pd.DataFrame, candidates: Sequence[RouteDirectionCandidate]
) -> Tuple[List[pd.DataFrame], int, int, int, int, Dict[Tuple[str, int, str], int]]:
    """Process one (vehicle_id, service-day) group.

    Splits the day into direction-agnostic coarse segments FIRST
    (:func:`bucr_trip_scoring.presegment_boundaries` -- one per trip, on
    terminal-turnaround dwells and long gaps), then assigns a direction+variant
    to each segment independently via :func:`_process_coarse_segment`. This
    inversion is the Step-4 architectural fix: a vehicle-day trace contains both
    directions interleaved, so assigning one direction to the whole day parked
    it as ambiguous (see the pre-segmentation module note in
    ``bucr_trip_scoring``).

    Returns (assigned_row_frames, n_parked, n_anomaly, n_dropped_as_noise_points,
    n_instances_dropped_as_noise, trips_per_variant_delta).
    """
    group = group.sort_values("ts").reset_index(drop=True)

    coarse = presegment_boundaries(
        group["ts"],
        group["lat"].to_numpy(dtype=float),
        group["lon"].to_numpy(dtype=float),
    )
    coarse_ranges = instance_index_ranges(coarse)

    out_frames: List[pd.DataFrame] = []
    n_parked = 0
    n_anomaly = 0
    n_noise_points = 0
    n_noise_instances = 0
    trips_delta: Dict[Tuple[str, int, str], int] = {}

    for start, end in coarse_ranges:
        seg = group.iloc[start:end].reset_index(drop=True)
        frames, parked, anomaly, noise_points, noise_instances, seg_trips = _process_coarse_segment(
            seg, candidates
        )
        out_frames.extend(frames)
        n_parked += parked
        n_anomaly += anomaly
        n_noise_points += noise_points
        n_noise_instances += noise_instances
        for key, count in seg_trips.items():
            trips_delta[key] = trips_delta.get(key, 0) + count

    return out_frames, n_parked, n_anomaly, n_noise_points, n_noise_instances, trips_delta


def infer_bucr_trips(
    vp_df: pd.DataFrame, candidates: Sequence[RouteDirectionCandidate]
) -> Tuple[pd.DataFrame, InferenceStats]:
    """Infer route/direction/shape-variant, trip instances, and stops for a BUCR VP trace.

    Args:
        vp_df: Canonical VP frame (the output of
            ``navsat_adapter.navsat_to_vehicle_positions`` on an already
            ``navsat_cleaning.clean_navsat_trace``-cleaned raw frame), with
            (at least) ``REQUIRED_VP_COLUMNS``. May span multiple vehicles
            and days; grouped internally by ``(vehicle_id, service day)``
            where service day is the UTC calendar date of ``ts`` (a
            documented simplification -- see module response / Step 4
            limitations, since BUCR's true service day is
            America/Costa_Rica local time).
        candidates: All ``RouteDirectionCandidate``s to score the trace
            against (typically every candidate for every
            (route_id, direction_id) pair from
            ``bucr_gtfs.route_direction_candidates``, across the feed's one
            or more routes).

    Returns:
        ``(out_df, stats)``. ``out_df`` has columns ``OUTPUT_COLUMNS``
        (``rt_source.OUTPUT_COLUMNS`` + ``route_id``), one row per
        *assigned, non-anomalous* input point, in a stable but not
        necessarily input-preserving order (grouped by vehicle/day, then by
        instance, then by ``ts`` ascending -- deterministic given the same
        input). ``stats`` is a frozen ``InferenceStats``.

    Raises:
        TypeError: if ``vp_df`` is not a DataFrame or ``candidates`` is empty/wrong type.
        ValueError: if any of ``REQUIRED_VP_COLUMNS`` is missing.
    """
    if not isinstance(vp_df, pd.DataFrame):
        raise TypeError(f"infer_bucr_trips expects a DataFrame, got {type(vp_df)!r}")
    if not isinstance(candidates, (list, tuple)):
        raise TypeError(
            f"candidates must be a list/tuple of RouteDirectionCandidate, got {type(candidates)!r}"
        )
    if not candidates:
        raise ValueError("candidates must not be empty")

    missing = _missing_columns(vp_df)
    if missing:
        raise ValueError(f"VP frame missing required columns: {missing}")

    if vp_df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), _empty_stats()

    df = vp_df.copy(deep=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")

    # Defensive: rows with unusable ts/lat/lon should not occur post-cleaning
    # (clean_navsat_trace already drops these upstream) -- if present anyway,
    # they cannot be projected onto any polyline, so fold them into the
    # anomaly-rejected count rather than crashing.
    valid_mask = df["ts"].notna() & df["lat"].notna() & df["lon"].notna()
    n_invalid = int((~valid_mask).sum())
    df = df.loc[valid_mask].reset_index(drop=True)

    points_in = len(vp_df)

    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), InferenceStats(
            points_in=points_in,
            points_assigned=0,
            points_parked_ambiguous=0,
            points_anomaly_rejected=n_invalid,
            points_dropped_as_noise=0,
            trips_inferred=0,
            instances_dropped_as_noise=0,
            trips_per_direction_variant=(),
        )

    service_day = df["ts"].dt.floor("D")
    group_keys = list(zip(df["vehicle_id"], service_day))
    df = df.assign(_group_key=group_keys)

    out_frames: List[pd.DataFrame] = []
    n_parked = 0
    n_anomaly = n_invalid
    n_noise_points = 0
    n_noise_instances = 0
    trips_totals: Dict[Tuple[str, int, str], int] = {}

    for _, group in df.groupby("_group_key", sort=True):
        group = group.drop(columns="_group_key")
        frames, parked, anomaly, noise_points, noise_instances, trips_delta = _process_group(group, candidates)
        out_frames.extend(frames)
        n_parked += parked
        n_anomaly += anomaly
        n_noise_points += noise_points
        n_noise_instances += noise_instances
        for key, count in trips_delta.items():
            trips_totals[key] = trips_totals.get(key, 0) + count

    if out_frames:
        result = pd.concat(out_frames, ignore_index=True)
    else:
        result = pd.DataFrame(columns=OUTPUT_COLUMNS)

    result = result.reindex(columns=OUTPUT_COLUMNS)
    if not result.empty:
        result["trip_id"] = result["trip_id"].astype("string")
        result["vehicle_id"] = result["vehicle_id"].astype("string")
        result["route_id"] = result["route_id"].astype("string")
        result["stop_id"] = result["stop_id"].astype("string")
        result["current_stop_sequence"] = result["current_stop_sequence"].astype("Int64")

    points_assigned = len(result)
    trips_per_variant = tuple(
        sorted(
            (
                (route_id, direction_id, shape_id, count)
                for (route_id, direction_id, shape_id), count in trips_totals.items()
            )
        )
    )

    stats = InferenceStats(
        points_in=points_in,
        points_assigned=points_assigned,
        points_parked_ambiguous=n_parked,
        points_anomaly_rejected=n_anomaly,
        points_dropped_as_noise=n_noise_points,
        trips_inferred=sum(trips_totals.values()),
        instances_dropped_as_noise=n_noise_instances,
        trips_per_direction_variant=trips_per_variant,
    )

    return result, stats
