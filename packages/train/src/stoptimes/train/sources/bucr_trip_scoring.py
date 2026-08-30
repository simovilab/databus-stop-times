"""Candidate scoring, trip-instance segmentation, and stop assignment helpers.

Sibling module to ``bucr_trip_inference.py`` -- split out purely to keep each
file under the repo's ~500-line target; this module has no independent
public entry point and is not meant to be used outside
``bucr_trip_inference.infer_bucr_trips``. See that module's docstring for the
overall pipeline and the reasoning behind each named threshold.

Pure/deterministic: no I/O, no network, no mutation of inputs. All spatial
math (haversine, polyline projection) is reused from
``etaval.spatial.polyline``.

Candidate selection is HIERARCHICAL and uses two different signals for its
two different questions:

  * DIRECTION ((route_id, direction_id)) is chosen by a divergence-sensitive
    cross-track score (:func:`score_candidates` / ``divergence_score_m`` --
    a high percentile, not the median; see its docstring). Distinct
    directions/routes diverge over the *entire* trace, so an aggregate
    cross-track statistic discriminates them cleanly (empirically, hundreds
    of metres apart on the real BUCR feed -- see this module's report).
  * VARIANT (shape_id within an already-chosen direction) is chosen by
    :func:`resolve_variant_by_stop_coverage`, NOT by cross-track score.
    BUCR's real shapes vary along TWO independent dimensions within one
    direction (which origin the trip starts from, e.g. "educacion" vs
    "artes"; and whether it takes an optional mid-route detour, e.g.
    "sin_milla" vs "con_milla"), and same-direction variants can share
    ~90%+ of their path. That means the segment where they actually diverge
    is a small enough fraction of total trace points that even a high
    percentile of cross-track error is unreliable -- empirically, forcing a
    choice by percentile-of-cross-track alone (p90/p95/p97/p98/p99, or a
    robust top-k max) got the wrong variant in ~13-15% of trials against the
    real feed with typical GPS noise, because the noise floor on the SHARED
    corridor is comparable in magnitude to the true geometric signal on the
    DIVERGENT leg. Stop-visitation is far more robust: GTFS stops are
    discrete, well-separated physical points (BUCR's origin/detour stops
    sit hundreds of metres from any other candidate's distinguishing stop --
    see the module report), so "did the trace come within a small radius of
    stop X" is a clean binary signal per distinguishing stop, immune to the
    shared-corridor noise problem. See :func:`resolve_variant_by_stop_coverage`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from etaval.spatial.polyline import Projection, haversine_m, project_point_to_polyline

from feature_engineering.bucr_gtfs import RouteDirectionCandidate

# ---------------------------------------------------------------------------
# Named thresholds -- see bucr_trip_inference.py module docstring for the
# full reasoning behind each; kept here (not duplicated) since these are the
# ones this module's functions consume directly.
# ---------------------------------------------------------------------------

# Two DIRECTION groups are indistinguishable when their best member's
# divergence score differs by less than this many metres. Only compared
# across distinct (route_id, direction_id) groups -- see module docstring;
# same-direction variants are resolved separately (never by this margin).
AMBIGUITY_MARGIN_M: float = 5.0

# Minimum forward-progress fraction for a candidate to be eligible at all.
MIN_MONOTONIC_PROGRESS_FRACTION: float = 0.6

# Consecutive-fix time/space gaps beyond these start a new trip instance.
MAX_GAP_SECONDS: float = 900.0
MAX_GAP_METERS: float = 500.0

# Fraction of shape length counting as "near the end" / "near the start"
# when detecting a loop-completion progress reset.
RESET_NEAR_END_FRACTION: float = 0.10
RESET_NEAR_START_FRACTION: float = 0.10

# Backward progress steps no larger than this are treated as GPS jitter, not
# a real regression, when scoring/segmenting.
MONOTONIC_JITTER_TOLERANCE_M: float = 15.0

# Two consecutive fixes whose great-circle displacement is at or below this
# many metres represent (at most) GPS jitter around a stationary vehicle,
# not real movement -- BUCR devices report noise on the order of 5-15 m at
# rest (see DISTINGUISHING_STOP_RADIUS_M's docstring), so 25 m sits above
# that noise floor while staying well below any real single-ping travel
# distance during a ~8-30 s poll interval on a moving bus. Combined with the
# device's own ``estado`` field (canonicalized to ``current_status`` ==
# "STOPPED_AT") in :func:`segment_boundaries` -- the real per-point idle
# signal -- this displacement check is a defensive fallback for when the
# device's status lags the fix (see that function's docstring).
IDLE_DISPLACEMENT_M: float = 25.0

# A progress "reset" (point i-1 near shape-end, point i near shape-start) is
# only accepted as a genuine completed loop if the CURRENT instance already
# covered at least this fraction of the shape's total length beforehand. A
# vehicle idling near a shape's start/end (BUCR's loop shapes begin and end
# at the same physical stop) can have its projected progress wobble between
# "near start" and "near end" on pure GPS noise -- with no gating, that
# wobble is repeatedly misread as "completed the whole loop, starting a new
# one," fragmenting one idle dwell into dozens of near-empty instances (see
# the module report: on the real 5-day BUCR sample this was the dominant
# cause of the ~1,790 mostly single-point "trip" instances a naive read of
# InferenceStats showed). 0.5 (50%) is deliberately generous -- half the
# shape length is far more than jitter can fake, but well under 100% so a
# real trip that started slightly athwart the nominal start point (a common
# GPS cold-start effect) still counts.
MIN_PROGRESS_BEFORE_RESET_FRACTION: float = 0.5

# ``navsat_adapter.map_estado_to_status``'s "detenido" (stopped) mapping --
# duplicated here (not imported) for the same reason SCORING_OUTLIER_EXCLUSION_M
# is: this module has no dependency on the adapter (the dependency runs the
# other way, through bucr_trip_inference).
STOPPED_STATUS: str = "STOPPED_AT"

# The percentile (not the median) of per-point cross-track error used as a
# candidate's DIRECTION-level divergence score. A high percentile is
# sensitive to a candidate diverging ANYWHERE along the trace (wrong
# route/direction), whereas the median is dominated by whatever fraction of
# the trace is on-corridor and can stay near-zero even for a badly wrong
# candidate that only diverges over part of its length. 90 is high enough to
# catch a real off-route excursion affecting a meaningful slice of points,
# while not being so extreme (e.g. p99/max) that a single stray GPS spike
# swings the whole score -- outlier points are separately excluded below
# before the percentile is taken.
DIVERGENCE_SCORE_PERCENTILE: float = 90.0

# Cross-track outliers above this many metres are excluded before computing
# a candidate's divergence score (see DIVERGENCE_SCORE_PERCENTILE). Matches
# ``bucr_trip_inference.ANOMALY_CROSS_TRACK_M`` by design -- a point that far
# from a candidate's shape is going to be rejected as an anomaly once that
# candidate is selected anyway, so it shouldn't be allowed to also distort
# *which* candidate gets selected. Duplicated here (not imported) because
# this module has no dependency on ``bucr_trip_inference`` (the dependency
# runs the other way); keep the two values in sync if either changes.
# NOTE: Raised from 75m to 150m after Step 4 quality report showed 28% of
# points were being rejected as anomalies with 75m threshold (p75 cross-track
# error was 145m).
SCORING_OUTLIER_EXCLUSION_M: float = 150.0

# A trace point within this many metres of a stop counts as evidence the
# vehicle actually visited that stop. Used by
# :func:`resolve_variant_by_stop_coverage` to resolve which shape VARIANT
# within an already-chosen direction applies, by checking visitation of each
# candidate's DISTINGUISHING stops (stops that differ between variants of
# one direction -- e.g. BUCR's alternate origin stop or its optional "milla"
# detour stops). 30 m matches etaval's own SourceConfig.shape_threshold_m
# default and sits comfortably above typical navsat GPS noise (5-15 m) while
# staying tight enough that a point merely passing along a shared corridor
# near a distinguishing stop (but not actually visiting it) doesn't
# false-positive -- real BUCR distinguishing stops sit hundreds of metres
# apart from each other (verified against the real feed; see module report).
DISTINGUISHING_STOP_RADIUS_M: float = 30.0


# ---------------------------------------------------------------------------
# Projection + candidate scoring
# ---------------------------------------------------------------------------


def project_points(lats: np.ndarray, lons: np.ndarray, polyline) -> Tuple[np.ndarray, np.ndarray]:
    """Project every (lat, lon) onto ``polyline``. Returns (progress_m, cross_track_m) arrays."""
    n = len(lats)
    progress = np.empty(n, dtype=float)
    cross_track = np.empty(n, dtype=float)
    for i in range(n):
        proj: Projection = project_point_to_polyline(float(lats[i]), float(lons[i]), polyline)
        progress[i] = proj.progress_m
        cross_track[i] = proj.cross_track_m
    return progress, cross_track


def monotonic_progress_fraction(progress: np.ndarray, shape_total_m: float) -> float:
    """Fraction of consecutive point-pairs that advance forward (or are an explained loop reset)."""
    if len(progress) < 2:
        return 1.0
    diffs = np.diff(progress)
    forward = diffs >= -MONOTONIC_JITTER_TOLERANCE_M
    near_end = progress[:-1] >= (1.0 - RESET_NEAR_END_FRACTION) * shape_total_m
    near_start = progress[1:] <= RESET_NEAR_START_FRACTION * shape_total_m
    reset = near_end & near_start
    ok = forward | reset
    return float(ok.sum()) / float(len(diffs))


def _divergence_score(cross_track: np.ndarray) -> float:
    """High-percentile cross-track score, robust to a lone outlier point.

    Points beyond ``SCORING_OUTLIER_EXCLUSION_M`` are dropped first (see its
    docstring); if that empties the array (every point is a huge outlier --
    a badly wrong candidate), fall back to the unfiltered array so such a
    candidate still scores (very) badly rather than raising.
    """
    mask = cross_track <= SCORING_OUTLIER_EXCLUSION_M
    filtered = cross_track[mask]
    sample = filtered if filtered.size > 0 else cross_track
    return float(np.percentile(sample, DIVERGENCE_SCORE_PERCENTILE))


@dataclass(frozen=True)
class CandidateScore:
    candidate: RouteDirectionCandidate
    progress_m: np.ndarray
    cross_track_m: np.ndarray
    divergence_score_m: float
    monotonic_fraction: float
    eligible: bool


def score_candidates(
    lats: np.ndarray, lons: np.ndarray, candidates: Sequence[RouteDirectionCandidate]
) -> List[CandidateScore]:
    """Project the trace onto every candidate and score each (see module docstring)."""
    scores: List[CandidateScore] = []
    for cand in candidates:
        shape_total_m = cand.polyline[-1].cum_m
        progress, cross_track = project_points(lats, lons, cand.polyline)
        fraction = monotonic_progress_fraction(progress, shape_total_m)
        divergence = _divergence_score(cross_track)
        eligible = fraction >= MIN_MONOTONIC_PROGRESS_FRACTION
        scores.append(
            CandidateScore(
                candidate=cand,
                progress_m=progress,
                cross_track_m=cross_track,
                divergence_score_m=divergence,
                monotonic_fraction=fraction,
                eligible=eligible,
            )
        )
    return scores


def _group_by_direction(
    scores: Sequence[CandidateScore],
) -> Dict[Tuple[str, int], List[CandidateScore]]:
    """Group candidate scores by (route_id, direction_id).

    Same-direction shape variants (e.g. BUCR's origin/detour combinations for
    one direction) share almost their entire path, so scoring them as
    independent "candidates" for the ambiguity check produces a near-zero
    gap that is NOT genuine route/direction ambiguity -- it's just variant
    selection, resolved separately by
    :func:`resolve_variant_by_stop_coverage`. Grouping first means the
    ambiguity margin only ever compares across genuinely different (route,
    direction) pairs.
    """
    groups: Dict[Tuple[str, int], List[CandidateScore]] = {}
    for s in scores:
        key = (s.candidate.route_id, s.candidate.direction_id)
        groups.setdefault(key, []).append(s)
    return groups


def _distinguishing_stop_ids(group: Sequence[CandidateScore]) -> Set[str]:
    """Stop ids not shared by every candidate in the group.

    These are the stops that actually carry information about which variant
    a trace took (e.g. BUCR's two alternate origin stops, or its two
    optional "milla" detour stops); stops common to every variant in the
    group are on the shared corridor and carry no discriminating evidence.
    """
    if not group:
        return set()
    stop_sets = [{st.stop_id for st in s.candidate.stops} for s in group]
    union: Set[str] = set().union(*stop_sets)
    common = set.intersection(*stop_sets) if len(stop_sets) > 1 else set()
    return union - common


def _stop_coverage_score(
    group: Sequence[CandidateScore],
    lats: np.ndarray,
    lons: np.ndarray,
) -> Dict[str, int]:
    """Per-candidate integer penalty from distinguishing-stop visitation.

    For each candidate C, penalty(C) = (number of C's OWN distinguishing
    stops the trace never comes within ``DISTINGUISHING_STOP_RADIUS_M`` of --
    evidence C's leg was NOT taken) + (number of a RIVAL's distinguishing
    stops -- one C does not have -- that the trace DID visit -- evidence a
    rival's leg WAS taken instead). The correct candidate scores 0: the
    trace visits everything it should and nothing it shouldn't. Lower is
    better; ties are possible (e.g. a trace confined entirely to the shared
    corridor, visiting no distinguishing stop at all -- genuine ambiguity,
    resolved by the caller's tie-break).
    """
    distinguishing = _distinguishing_stop_ids(group)
    if not distinguishing:
        return {s.candidate.shape_id: 0 for s in group}

    # One min-distance-to-trace computation per distinguishing stop id,
    # shared across candidates that reference the same physical stop.
    stop_lookup: Dict[str, Tuple[float, float]] = {}
    for s in group:
        for st in s.candidate.stops:
            if st.stop_id in distinguishing and st.stop_id not in stop_lookup:
                stop_lookup[st.stop_id] = (st.lat, st.lon)

    visited: Dict[str, bool] = {}
    for stop_id, (slat, slon) in stop_lookup.items():
        dists = np.array(
            [haversine_m(float(lat), float(lon), slat, slon) for lat, lon in zip(lats, lons)]
        )
        visited[stop_id] = bool(len(dists)) and float(dists.min()) <= DISTINGUISHING_STOP_RADIUS_M

    scores: Dict[str, int] = {}
    for s in group:
        own_distinguishing = {st.stop_id for st in s.candidate.stops} & distinguishing
        rival_distinguishing = distinguishing - own_distinguishing
        missing = sum(1 for sid in own_distinguishing if not visited.get(sid, False))
        wrongly_visited = sum(1 for sid in rival_distinguishing if visited.get(sid, False))
        scores[s.candidate.shape_id] = missing + wrongly_visited
    return scores


def resolve_variant_by_stop_coverage(
    group: Sequence[CandidateScore], lats: np.ndarray, lons: np.ndarray
) -> CandidateScore:
    """Pick the shape VARIANT within one already-chosen direction, by stop coverage.

    Unlike direction assignment, variant selection is NOT decided by
    cross-track score (same-direction variants share too much path for that
    to reliably discriminate -- see module docstring). Instead, each
    candidate is scored by :func:`_stop_coverage_score` (visitation of its
    distinguishing stops vs. its rivals'); the lowest-penalty candidate
    wins. This also resolves the "genuine near-tie" case naturally: a
    no-detour trace scores 0 against its own no-detour variant (all its
    stops visited, no rival stops visited) but scores >=1 against the
    same-origin detour variant (the detour's stops were never visited, so
    they count as "missing") -- so the shorter/no-detour variant wins
    outright, without parking or any special-cased rule.

    Ties (including the trivial all-zero tie when ``group`` has no
    distinguishing stops, or a trace confined to the shared corridor that
    visits no distinguishing stop at all) are broken, in order: (1) fewest
    total stops -- the "simpler"/shorter variant, matching the preference
    the near-tie case above already encodes; (2) lowest
    ``divergence_score_m`` as a final deterministic fallback. Variant
    selection never parks (``None``) -- parking is reserved for genuine
    cross-direction/cross-route ambiguity in :func:`select_best_candidate`.

    If ``group`` has only one member, that member is returned unconditionally
    (nothing to resolve).
    """
    if len(group) == 1:
        return group[0]

    penalties = _stop_coverage_score(group, lats, lons)
    best_penalty = min(penalties.values())
    tied = [s for s in group if penalties[s.candidate.shape_id] == best_penalty]
    if len(tied) == 1:
        return tied[0]

    tied.sort(key=lambda s: (len(s.candidate.stops), s.divergence_score_m, s.candidate.shape_id))
    return tied[0]


def select_best_candidate(
    scores: Sequence[CandidateScore], lats: np.ndarray, lons: np.ndarray
) -> Optional[CandidateScore]:
    """Hierarchical assignment: pick a DIRECTION first, then a shape VARIANT within it.

    1. Group scores by (route_id, direction_id). Within each group, the
       "direction score" is the best (lowest) ``divergence_score_m`` among
       that group's ELIGIBLE members -- a group with no eligible member
       drops out entirely.
    2. Rank groups by their direction score. If the best two groups are
       within ``AMBIGUITY_MARGIN_M`` of each other, or no group has an
       eligible member, the whole trace is parked (``None``) -- this is
       genuine cross-direction/cross-route ambiguity, not decided by variant
       near-ties within one direction.
    3. Within the single winning group, resolve the shape variant by stop
       coverage (:func:`resolve_variant_by_stop_coverage`), NOT by score --
       same-direction variants routinely score within the ambiguity margin
       of each other and that is not grounds to park.
    """
    groups = _group_by_direction(scores)

    direction_bests: List[Tuple[Tuple[str, int], CandidateScore]] = []
    for key, group in groups.items():
        eligible = [s for s in group if s.eligible]
        if not eligible:
            continue
        best = min(eligible, key=lambda s: s.divergence_score_m)
        direction_bests.append((key, best))

    if not direction_bests:
        return None

    direction_bests.sort(key=lambda kb: kb[1].divergence_score_m)
    if len(direction_bests) >= 2:
        best_score = direction_bests[0][1].divergence_score_m
        second_score = direction_bests[1][1].divergence_score_m
        if (second_score - best_score) < AMBIGUITY_MARGIN_M:
            return None

    winning_key = direction_bests[0][0]
    return resolve_variant_by_stop_coverage(groups[winning_key], lats, lons)


# ---------------------------------------------------------------------------
# Direction-agnostic pre-segmentation (coarse trip splitting BEFORE candidate
# assignment)
# ---------------------------------------------------------------------------
#
# CRITICAL ARCHITECTURE NOTE (Step 4 finding, 2026-08-24): a BUCR vehicle runs
# continuous round trips all day (direction 0 out to a terminal, direction 1
# back, repeat). So one (vehicle, service-day) trace genuinely contains BOTH
# directions interleaved. If we scored the WHOLE day against every candidate
# and tried to pick ONE direction (the original design), the best direction-0
# and best direction-1 candidates always landed within ``AMBIGUITY_MARGIN_M``
# of each other -- because both directions really are present -- so the entire
# day was parked as "ambiguous" (measured: 58% of all points on the real 5-day
# sample). The fix is to split the trace into per-trip coarse segments FIRST,
# direction-agnostically, THEN assign a direction+variant to each segment
# independently (where each segment is mostly one direction, so the ambiguity
# check works as intended). Measured effect of this inversion on the real
# sample: assigned 42% -> 91%, parked 58% -> 9%.
#
# The direction-agnostic signal for "one trip ended, another begins" is the
# terminal turnaround: between a direction-0 trip and the direction-1 trip
# back, the vehicle sits at the terminal for a bit (passenger exchange, driver
# break). That shows up as a STATIONARY DWELL -- a run of consecutive fixes
# staying within a small radius for at least some time -- even when there is no
# data gap. A long data gap (device offline, midday break) is the other,
# simpler boundary signal.

# A run of consecutive fixes all staying within ``PRESEGMENT_DWELL_RADIUS_M``
# of the run's first fix, lasting at least this many seconds, is a terminal
# turnaround (or other real layover) and ends the current coarse segment. 90 s
# sits above a normal in-service passenger stop (a few seconds to ~half a
# minute) but at/below a real terminal turnaround, so it splits trips without
# chopping a single trip at its ordinary stops. Grounded on the real sample:
# 90 s produced ~21 coarse segments per vehicle-day (~one per real trip for a
# ~5-10 min loop over a service day), and drove the assigned rate to 91%.
PRESEGMENT_DWELL_SECONDS: float = 90.0

# Radius defining "stationary" for the dwell check above. Matches
# ``DISTINGUISHING_STOP_RADIUS_M`` (30 m) -- comfortably above the 5-15 m
# navsat noise floor at rest, tight enough that a moving bus leaves the radius
# within one or two polls.
PRESEGMENT_DWELL_RADIUS_M: float = 30.0

# A consecutive-fix time gap beyond this also ends a coarse segment (device
# offline, a long midday break). Matches ``MAX_GAP_SECONDS`` (900 s is far too
# long to be a single trip's internal poll gap); kept as its own named
# constant so the pre-segmentation rule reads independently of the
# within-segment progress-reset rule that reuses MAX_GAP_SECONDS.
PRESEGMENT_MAX_GAP_SECONDS: float = MAX_GAP_SECONDS


def presegment_boundaries(
    ts: pd.Series,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """Direction-agnostic coarse boundaries: True at i means a new segment starts at i.

    Splits a (vehicle, service-day) trace into per-trip coarse segments WITHOUT
    needing any candidate/shape -- see this section's module note for why this
    must happen before candidate assignment. A new coarse segment starts at
    index i > 0 when, relative to point i-1, EITHER:

      * the time gap exceeds ``PRESEGMENT_MAX_GAP_SECONDS`` (device offline / a
        long break), OR
      * the vehicle just left a STATIONARY DWELL that lasted at least
        ``PRESEGMENT_DWELL_SECONDS`` -- i.e. points ``[dwell_start, i-1]`` all
        stayed within ``PRESEGMENT_DWELL_RADIUS_M`` of the fix at
        ``dwell_start`` and spanned at least that long in time, and point i is
        the first fix to move outside that radius (movement resumed -> the next
        trip has begun).

    Index 0 is always a boundary. A dwell that is still open at the end of the
    trace does not create a trailing boundary (there is no "next trip" to open).

    Returns a boolean array the same length as ``ts``.
    """
    n = len(ts)
    boundary = np.zeros(n, dtype=bool)
    if n == 0:
        return boundary
    boundary[0] = True
    if n == 1:
        return boundary

    ts_vals = ts.to_numpy()
    # Anchor of the dwell currently being tracked (the last fix from which the
    # vehicle had not yet moved beyond the radius).
    dwell_start = 0

    for i in range(1, n):
        time_gap_s = (ts_vals[i] - ts_vals[i - 1]) / np.timedelta64(1, "s")
        if time_gap_s > PRESEGMENT_MAX_GAP_SECONDS:
            boundary[i] = True
            dwell_start = i
            continue

        moved_m = haversine_m(
            float(lats[dwell_start]), float(lons[dwell_start]), float(lats[i]), float(lons[i])
        )
        if moved_m > PRESEGMENT_DWELL_RADIUS_M:
            dwell_dur_s = (ts_vals[i - 1] - ts_vals[dwell_start]) / np.timedelta64(1, "s")
            # Only a dwell that (a) lasted long enough and (b) actually sat for
            # more than the single anchor fix (dwell_start < i-1) is a real
            # layover; a lone fix followed by immediate movement is just travel.
            if dwell_dur_s >= PRESEGMENT_DWELL_SECONDS and dwell_start < i - 1:
                boundary[i] = True
            dwell_start = i

    return boundary


# ---------------------------------------------------------------------------
# Segmentation (trip-instance boundaries)
# ---------------------------------------------------------------------------


def segment_boundaries(
    ts: pd.Series,
    lats: np.ndarray,
    lons: np.ndarray,
    progress: np.ndarray,
    shape_total_m: float,
    current_status: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Boolean array (len == len(ts)) -- True at index i means a NEW instance starts at i.

    Index 0 is always a boundary (the start of the first instance). A new
    instance starts at i > 0 when, relative to point i-1:
      - the time gap exceeds ``MAX_GAP_SECONDS``, or
      - the great-circle distance exceeds ``MAX_GAP_METERS``, or
      - progress resets (point i-1 near shape-end, point i near shape-start
        -- a completed loop) AND the instance already covered at least
        ``MIN_PROGRESS_BEFORE_RESET_FRACTION`` of the shape beforehand (see
        that constant's docstring -- an idling vehicle near a loop's
        coincident start/end point must not be read as "completing" a loop
        it never actually ran), or
      - progress regresses by more than ``MONOTONIC_JITTER_TOLERANCE_M``
        without being an explained loop reset AND without both points being
        IDLE (an unexplained backward jump between two genuinely moving
        fixes -- treated conservatively as a discontinuity; between two
        idle fixes it's just noise, not a discontinuity).

    A step from point i-1 to i is IDLE when either the device itself
    reports both fixes stopped (``current_status[i-1]`` and
    ``current_status[i]`` both equal ``STOPPED_STATUS``) or their
    great-circle displacement is at most ``IDLE_DISPLACEMENT_M`` (a
    defensive fallback for when the device's reported status lags the fix,
    or ``current_status`` is unavailable/null). Idle steps never seed a new
    instance on their own -- see module report for why this matters: a
    parked vehicle's GPS jitter previously fragmented one long idle dwell
    into dozens of spurious single-point "instances."

    Args:
        current_status: Optional per-point ``current_status`` values
            (``"STOPPED_AT"``/``"IN_TRANSIT_TO"``/other), same length as
            ``ts``. If omitted, idle detection relies solely on the
            displacement fallback.
    """
    n = len(ts)
    boundary = np.zeros(n, dtype=bool)
    if n == 0:
        return boundary
    boundary[0] = True
    if n == 1:
        return boundary

    ts_vals = ts.to_numpy()
    status = current_status if current_status is not None else np.full(n, "", dtype=object)

    # Tracks the furthest along-shape progress reached since the start of
    # the CURRENT (still-open) instance -- gates whether a "reset" is a real
    # completed loop (see MIN_PROGRESS_BEFORE_RESET_FRACTION).
    instance_max_progress = float(progress[0])

    for i in range(1, n):
        time_gap_s = (ts_vals[i] - ts_vals[i - 1]) / np.timedelta64(1, "s")
        space_gap_m = haversine_m(float(lats[i - 1]), float(lons[i - 1]), float(lats[i]), float(lons[i]))

        prev_p, cur_p = progress[i - 1], progress[i]

        both_stopped = bool(status[i - 1] == STOPPED_STATUS and status[i] == STOPPED_STATUS)
        is_idle_step = both_stopped or space_gap_m <= IDLE_DISPLACEMENT_M

        near_end = prev_p >= (1.0 - RESET_NEAR_END_FRACTION) * shape_total_m
        near_start = cur_p <= RESET_NEAR_START_FRACTION * shape_total_m
        made_substantial_progress = instance_max_progress >= MIN_PROGRESS_BEFORE_RESET_FRACTION * shape_total_m
        is_reset = bool(near_end and near_start and made_substantial_progress)

        unexplained_backward = (
            (cur_p < prev_p - MONOTONIC_JITTER_TOLERANCE_M) and not is_reset and not is_idle_step
        )

        is_boundary = (
            time_gap_s > MAX_GAP_SECONDS
            or space_gap_m > MAX_GAP_METERS
            or is_reset
            or unexplained_backward
        )
        boundary[i] = is_boundary

        instance_max_progress = float(cur_p) if is_boundary else max(instance_max_progress, float(cur_p))

    return boundary


def instance_index_ranges(boundary: np.ndarray) -> List[Tuple[int, int]]:
    """Convert a boundary mask into a list of half-open [start, end) index ranges."""
    starts = np.flatnonzero(boundary)
    ends = list(starts[1:]) + [len(boundary)]
    return list(zip(starts.tolist(), ends))


# ---------------------------------------------------------------------------
# Minimum-instance filter -- drop segmented instances too small to be a real
# trip (idle-dwell noise that survived segmentation, or a genuine short
# blip). Grounded against the real 5-day BUCR sample (Step 4 quality
# report, 2026-08-24): even AFTER the segment_boundaries idle-gating fix
# above, the surviving instance-duration/point/distance distribution was
# still clearly bimodal -- a large cluster of near-zero-everything noise
# instances (parked-vehicle dwells, GPS multipath blips) well separated from
# a cluster of instances covering most of a shape over several minutes,
# consistent with real ~5-10 min BUCR loops. See this module's report for
# the exact counts observed on each side of the cuts below.
# ---------------------------------------------------------------------------

# A real BUCR loop is a few hundred GPS fixes (device polls roughly every
# 8-30 s while moving); an instance with fewer points than this cannot have
# covered a meaningful slice of a ~1-2 km shape and is noise regardless of
# what its duration/distance happen to read.
MIN_INSTANCE_POINTS: int = 5

# A real BUCR loop takes several minutes; this many seconds sits well under
# the shortest real loop time (~5 min) while safely excluding zero/near-zero
# duration noise instances (a burst of idle fixes at effectively the same
# instant).
MIN_INSTANCE_DURATION_SECONDS: float = 120.0

# Fraction of the candidate shape's total length the instance's progress
# must span (max progress - min progress, within the instance) to count as
# having actually traveled the route, not just sat somewhere along it.
MIN_INSTANCE_DISTANCE_FRACTION: float = 0.3


def instance_distance_covered_fraction(progress: np.ndarray, shape_total_m: float) -> float:
    """Fraction of ``shape_total_m`` this instance's progress values span."""
    if len(progress) == 0 or shape_total_m <= 0:
        return 0.0
    return float(progress.max() - progress.min()) / shape_total_m


def instance_passes_min_thresholds(
    ts: pd.Series,
    progress: np.ndarray,
    shape_total_m: float,
) -> bool:
    """True if a segmented instance is large enough to plausibly be a real trip.

    Applies all three named thresholds above -- point count, wall-clock
    duration, and along-shape distance covered. All three must pass; a
    long-but-stationary dwell (duration ok, distance ~0) and a short burst
    of fast-moving noise (distance ok, duration ~0) are both noise, not
    trips, and this rejects either.

    Args:
        ts: The instance's ``ts`` values, in order (its length is the
            instance's point count).
        progress: The instance's along-shape progress values, same order.
        shape_total_m: The candidate shape's total length in metres.
    """
    if len(ts) < MIN_INSTANCE_POINTS:
        return False
    duration_s = (pd.Timestamp(ts.iloc[-1]) - pd.Timestamp(ts.iloc[0])).total_seconds()
    if duration_s < MIN_INSTANCE_DURATION_SECONDS:
        return False
    if instance_distance_covered_fraction(progress, shape_total_m) < MIN_INSTANCE_DISTANCE_FRACTION:
        return False
    return True


# ---------------------------------------------------------------------------
# Stop assignment
# ---------------------------------------------------------------------------


def assign_stop_sequences(progress: np.ndarray, stops: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    """current_stop_sequence/stop_id arrays for one trip instance, loop-back-safe.

    ``current_stop_sequence`` follows GTFS-RT convention: the stop the
    vehicle is currently approaching or standing at -- the first stop (by
    ``progress_m``) that is >= the point's progress. A point past the last
    stop is clamped to the last stop. A running index floor prevents a later
    point from ever regressing to an earlier stop within the instance
    (loop-back-safety), guarding against residual GPS jitter that
    ``segment_boundaries`` didn't already split into a new instance.
    """
    ordered = sorted(stops, key=lambda s: s.stop_sequence)
    stop_progress = np.array([s.progress_m for s in ordered], dtype=float)
    n_stops = len(ordered)

    seqs = np.empty(len(progress), dtype=np.int64)
    ids = np.empty(len(progress), dtype=object)

    last_idx = 0
    for i, p in enumerate(progress):
        idx = int(np.searchsorted(stop_progress, p, side="left"))
        if idx >= n_stops:
            idx = n_stops - 1
        idx = max(idx, last_idx)
        last_idx = idx
        seqs[i] = ordered[idx].stop_sequence
        ids[i] = ordered[idx].stop_id
    return seqs, ids
