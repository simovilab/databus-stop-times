"""Causal, leakage-free "live network conditions" features (roadmap 5.x).

The MBTA XGBoost model plateaus at ~24.6s test MAE / R2 0.65 across every
hyperparameter config and every *static* feature tried so far (see
`docs/RESEARCH_ROADMAP.md` 5.1, and the `scratchpad/feature_sched_duration.py`
probe which showed scheduled segment duration moves nothing). The roadmap's
standing hypothesis is that the residual is *within-segment live variance* --
this segment is running slow/fast right now, for reasons no static feature
(geometry, time-of-day, schedule) can see -- and that only a live "conditions"
signal, computed from the recent past of the vehicle-position feed itself, can
move it. This module builds that signal.

Every feature here answers "what did we already know, strictly before this
row's `vp_ts`?" -- nothing else. That is the entire contract; getting it wrong
silently inflates offline metrics with information that will not exist at
serving time. Concretely:

- A segment traversal (`from_stop_id` -> `stop_id` on one trip instance)
  becomes an observation **only once `actual_arrival` is known and is
  strictly less than the query row's `vp_ts`** -- not when the vehicle
  entered the segment, not when the row's own trip started. A row whose own
  segment is still in progress (`actual_arrival >= vp_ts`, true for every row
  except possibly the arrival ping itself, where `vp_ts == actual_arrival`
  and is still excluded by the strict inequality) never sees its own target.
- The traversal *start* time for segment `segment_index = i` (i > 0) of a
  trip instance is taken to be `actual_arrival` of that same trip instance's
  segment `i - 1` -- the moment the vehicle arrived at `from_stop_id`. This
  is an exact, already-observed timestamp (not an estimate from VP ping
  spacing), so `duration = actual_arrival[i] - actual_arrival[i - 1]` is a
  precise traversal time with no serving-time-unavailable data in it.
  Segment 0 of a trip instance has no known start and contributes no
  duration (see "cold start" below) but still contributes a headway event
  (the arrival itself is a real, causal fact even without a start time).
- The "long-run" baseline each recent observation is compared against is an
  **expanding, causal, online mean** -- the mean of every prior observation
  for that segment/route/vehicle key up to (not including) the observation
  itself -- never a mean computed once over the full corpus (which would
  leak future-corpus statistics backward into early rows) and never fit on
  train alone and frozen (which would silently go stale and, worse, still
  implicitly assumes train/test are drawn from the same regime rather than
  being one continuous causal-online stream). Because the whole corpus is
  processed as a single chronological pass, test-period rows legitimately
  see earlier test-period observations -- exactly what a live feature store
  would have available online -- while never seeing anything at or after
  their own `vp_ts`.

Implementation is a set of causal point-in-time joins (`pandas.merge_asof`,
`direction="backward"`, `allow_exact_matches=False`) against a table of
*already-completed* traversal events, grouped by segment / route / vehicle.
This is deliberately not a `groupby().transform()` / `groupby().mean()`
pattern: an ordinary groupby aggregate is computed once over the whole group
and is visible to every row in that group regardless of order in time, which
is exactly the leak this module exists to prevent. `merge_asof` instead walks
both sides in timestamp order and, for each query row, can only ever attach
the state of *prior* events.

Pure/deterministic; no I/O, no network, no mutation of inputs (every function
here returns a new DataFrame).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

__all__ = [
    "RollingConditionsConfig",
    "ROLLING_CONDITIONS_FEATURES",
    "add_rolling_conditions_features",
]

# Columns this module adds, in the order they are documented above. Exposed so
# callers (and tests) don't have to hardcode the list.
ROLLING_CONDITIONS_FEATURES: List[str] = [
    "recent_seg_traversal_s",
    "recent_seg_ratio",
    "seg_n_obs_before",
    "headway_since_last_traversal_s",
    "leader_last_duration_s",
    "route_congestion_index",
    "vehicle_recent_delay_ratio",
]

_REQUIRED_COLUMNS = [
    "trip_id", "route_id", "vehicle_id", "stop_id", "from_stop_id",
    "segment_index", "vp_ts", "actual_arrival",
]


@dataclass(frozen=True)
class RollingConditionsConfig:
    """Tunables for the rolling-conditions features.

    Attributes:
        offset_hours: service-day offset (hours subtracted from `vp_ts` UTC
            before taking `.date()`) used only to derive a stable trip-instance
            key `(trip_id, service_date)` -- MBTA uses 7 (3am ET), bUCR uses 6.
            Required, not defaulted: silently assuming the wrong agency's
            offset would misgroup trip instances across the service-day
            boundary.
        min_observations: a segment/route/vehicle's expanding baseline is
            considered "warm" only once it has seen at least this many prior
            observations; below that, ratio features are NaN rather than
            leaking a noisy or single-sample "mean". Applied independently to
            the segment, route, and vehicle baselines.
        seg_ewma_span: span (pandas `.ewm(span=...)` convention, ~= 2/(span+1)
            smoothing weight on the newest observation) for the per-segment
            recent-traversal-time EWMA -- "the last K vehicles through this
            segment", K ~= span.
        route_ewma_span: span for the per-route congestion-index EWMA.
        vehicle_ewma_span: span for the per-vehicle own-recent-delay EWMA.
    """

    offset_hours: float
    min_observations: int = 3
    seg_ewma_span: float = 5.0
    route_ewma_span: float = 10.0
    vehicle_ewma_span: float = 5.0

    def __post_init__(self) -> None:
        if self.min_observations < 1:
            raise ValueError("min_observations must be >= 1")
        for name in ("seg_ewma_span", "route_ewma_span", "vehicle_ewma_span"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")


def _validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"rolling_conditions: missing required columns: {missing}")
    if not pd.api.types.is_datetime64_any_dtype(df["vp_ts"]):
        raise ValueError("rolling_conditions: vp_ts must be datetime64")
    if not pd.api.types.is_datetime64_any_dtype(df["actual_arrival"]):
        raise ValueError("rolling_conditions: actual_arrival must be datetime64")


def _service_date(ts: pd.Series, offset_hours: float) -> pd.Series:
    return (ts - pd.Timedelta(hours=offset_hours)).dt.date


def _segment_key(from_stop: pd.Series, to_stop: pd.Series) -> pd.Series:
    # Stable, human-readable composite key; str-typed either side already
    # (matches the corpus schema), but be defensive.
    return from_stop.astype(str) + ">" + to_stop.astype(str)


def _build_segment_completions(df: pd.DataFrame, offset_hours: float) -> pd.DataFrame:
    """One row per (trip_id, service_date, segment_index): the traversal that
    completed at `actual_arrival`, with `duration` = time since the same trip
    instance's previous segment arrived (NaN for segment_index 0, where no
    prior arrival is known -- see module docstring). Sorted by `arrival`.
    """
    base = df.loc[:, [
        "trip_id", "route_id", "vehicle_id", "from_stop_id", "stop_id",
        "segment_index", "actual_arrival",
    ]].copy()
    base["service_date"] = _service_date(df["vp_ts"], offset_hours)
    # Multiple VP pings share one (trip, service_date, segment) traversal;
    # actual_arrival/route/vehicle are constant within it -- collapse to one row.
    events = (
        base.drop_duplicates(subset=["trip_id", "service_date", "segment_index"])
        .sort_values(["trip_id", "service_date", "segment_index"], kind="mergesort")
        .reset_index(drop=True)
    )

    grp = events.groupby(["trip_id", "service_date"], sort=False)
    prev_arrival = grp["actual_arrival"].shift(1)
    prev_segment_index = grp["segment_index"].shift(1)
    contiguous = prev_segment_index == (events["segment_index"] - 1)
    duration = (events["actual_arrival"] - prev_arrival).dt.total_seconds()
    events["duration"] = np.where(contiguous, duration, np.nan)
    # Degenerate/non-positive durations (clock skew, duplicate arrivals) are
    # not a plausible traversal time -- treat as missing rather than poisoning
    # the baseline with a false "instant" segment.
    events.loc[events["duration"] <= 0, "duration"] = np.nan

    events["segment_key"] = _segment_key(events["from_stop_id"], events["stop_id"])
    events = events.rename(columns={"actual_arrival": "arrival"})
    events = events.sort_values("arrival", kind="mergesort").reset_index(drop=True)
    return events


def _expanding_stats(events: pd.DataFrame, key: str, span: float,
                      min_observations: int) -> pd.DataFrame:
    """Attach, to each event (a completed traversal, sorted by `arrival`),
    the causal EWMA and expanding mean/count of `duration` for its `key`
    group, **including** this event -- i.e. the state a query strictly after
    this event's `arrival` would legitimately see once merge_asof matches it.
    Also attaches a `ratio_before` computed from the group's history
    *excluding* this event (used only internally to seed the route/vehicle
    congestion indices without an event validating itself).
    """
    # `.expanding()`/`.ewm()` below rely on each group's rows being in
    # chronological order as fed in. Sorting here, unconditionally, makes
    # this function correct regardless of caller order: the first call in
    # the pipeline (segment-level) is fed already arrival-sorted, but the
    # second-order calls (route-level, vehicle-level) are fed from a
    # segment-grouped result and must not be assumed pre-sorted.
    #
    # Uses SeriesGroupBy.transform (not DataFrameGroupBy.apply) deliberately:
    # pandas >=2.2 forces `include_groups=False` on DataFrameGroupBy.apply,
    # which silently drops the grouping column from what the callback sees
    # and from the result -- exactly the columns this function needs to keep.
    # transform() on a single column has no such issue and is simpler besides.
    out = events.sort_values("arrival", kind="mergesort").reset_index(drop=True)
    have_dur = out["duration"].notna()
    g = out.groupby(key, sort=False)["duration"]

    out["n_obs_before"] = g.transform(
        lambda s: s.notna().cumsum().shift(1, fill_value=0)
    )
    out["mean_before"] = g.transform(
        lambda s: s.expanding(min_periods=1).mean().shift(1)
    )
    out["mean_after"] = g.transform(lambda s: s.expanding(min_periods=1).mean())
    out["ewma_after"] = g.transform(
        lambda s: s.ewm(span=span, min_periods=1, adjust=False).mean()
    )

    # "after" count -- how many valid observations (including this one, if it
    # has a valid duration) are behind the state a later query would see once
    # matched to this event. This is what `seg_n_obs_before` in the public
    # output actually reports (the "confidence" of the matched state), as
    # distinct from `n_obs_before` above (strictly prior to this event only,
    # used to gate `ratio_before` so an event never validates itself).
    out["n_obs_after"] = out["n_obs_before"] + have_dur.astype(int)

    warm_before = out["n_obs_before"] >= min_observations
    # `ratio_before`: this SINGLE event's own duration against the group's
    # prior (pre-event) expanding mean -- "how anomalous was this one
    # traversal". Used only as an internal input to the route-/vehicle-level
    # indices (each completion contributes one ratio sample there); using the
    # *prior* mean (not including this event) keeps an event from partially
    # validating itself.
    out["ratio_before"] = np.where(
        warm_before & have_dur, out["duration"] / out["mean_before"], np.nan
    )
    warm_after = out["n_obs_after"] >= min_observations
    # `ewma_ratio_after`: the SMOOTHED recent pace (EWMA, including this
    # event) against the inclusive expanding mean -- "is this segment
    # running slow/fast right now, right up to what we know as of this
    # event". This is the definition the module docstring / roadmap probe
    # calls for and is what `recent_seg_ratio` in the public output reports.
    out["ewma_ratio_after"] = np.where(
        warm_after, out["ewma_after"] / out["mean_after"], np.nan
    )
    # Where duration itself is NaN (segment_index 0), no duration-derived
    # stat can legitimately update -- `mean_after`/`ewma_after` simply carry
    # forward the state from the last valid-duration row in the group,
    # because pandas' `.expanding()`/`.ewm()` treat a NaN input as a skip
    # (they do not reset or count it), so no extra handling is needed here.
    return out


def _asof_join(query: pd.DataFrame, events: pd.DataFrame, by: str,
                value_cols: List[str]) -> pd.DataFrame:
    """Causal point-in-time join: for each query row, attach the `value_cols`
    of the most recent `events` row in the same `by` group with
    `arrival` strictly before the query's `vp_ts`. Rows in `query`/`events`
    may arrive in any order; both are re-sorted internally and the result is
    restored to `query`'s original row order, so the join is deterministic
    regardless of input ordering.
    """
    q = query[["_row_id", by, "vp_ts"]].sort_values("vp_ts", kind="mergesort")
    e = events[[by, "arrival"] + value_cols].sort_values("arrival", kind="mergesort")
    merged = pd.merge_asof(
        q, e,
        left_on="vp_ts", right_on="arrival",
        left_by=by, right_by=by,
        direction="backward", allow_exact_matches=False,
    )
    return merged.set_index("_row_id").reindex(query["_row_id"]).reset_index(drop=True)


def add_rolling_conditions_features(
    df: pd.DataFrame,
    config: RollingConditionsConfig,
) -> pd.DataFrame:
    """Return a copy of `df` with the rolling-conditions feature columns
    (`ROLLING_CONDITIONS_FEATURES`) appended. `df` is not mutated.

    Cold-start / fallback policy: every added feature is `NaN` when its
    causal history is empty or (for ratio features) below
    `config.min_observations`. No synthetic fill (global mean, zero, etc.) is
    applied -- NaN is passed straight through for XGBoost's native
    missing-value routing, matching the existing convention in
    `scratchpad/feature_sched_duration.py`. Any other consumer of these
    features must apply its own explicit imputation; silently substituting a
    value here would hide exactly the "insufficient history" signal the
    guard exists to preserve.
    """
    _validate_input(df)
    out = df.copy()
    out["_row_id"] = np.arange(len(out))
    out["segment_key"] = _segment_key(out["from_stop_id"], out["stop_id"])

    events = _build_segment_completions(df, config.offset_hours)

    # --- per-segment: recent traversal time + ratio to expanding baseline ---
    seg_stats = _expanding_stats(
        events, key="segment_key", span=config.seg_ewma_span,
        min_observations=config.min_observations,
    )
    seg_join = _asof_join(
        out, seg_stats, by="segment_key",
        value_cols=["ewma_after", "ewma_ratio_after", "n_obs_after", "duration"],
    )
    out["recent_seg_traversal_s"] = seg_join["ewma_after"]
    out["recent_seg_ratio"] = seg_join["ewma_ratio_after"]
    out["seg_n_obs_before"] = seg_join["n_obs_after"]
    out["leader_last_duration_s"] = seg_join["duration"]

    # --- headway: time since the segment was last traversed by any vehicle,
    # regardless of whether that traversal's duration is known ---
    headway_join = _asof_join(out, events, by="segment_key", value_cols=[])
    out["headway_since_last_traversal_s"] = (
        out["vp_ts"] - headway_join["arrival"]
    ).dt.total_seconds()

    # --- route-level congestion index: causal EWMA of recent per-segment
    # ratio-to-baseline across the whole route ---
    route_events = seg_stats.dropna(subset=["ratio_before"]).copy()
    route_events["duration"] = route_events["ratio_before"]  # reuse the EWMA helper
    route_stats = _expanding_stats(
        route_events, key="route_id", span=config.route_ewma_span,
        min_observations=1,  # already gated by seg-level min_observations upstream
    )
    route_join = _asof_join(
        out, route_stats, by="route_id", value_cols=["ewma_after"],
    )
    out["route_congestion_index"] = route_join["ewma_after"]

    # --- vehicle-level: this vehicle's own recent delay ratio over its last
    # completed segments (upstream propagation) ---
    vehicle_events = seg_stats.dropna(subset=["ratio_before"]).copy()
    vehicle_events["duration"] = vehicle_events["ratio_before"]
    vehicle_stats = _expanding_stats(
        vehicle_events, key="vehicle_id", span=config.vehicle_ewma_span,
        min_observations=1,
    )
    vehicle_join = _asof_join(
        out, vehicle_stats, by="vehicle_id", value_cols=["ewma_after"],
    )
    out["vehicle_recent_delay_ratio"] = vehicle_join["ewma_after"]

    return out.drop(columns=["_row_id", "segment_key"])
