"""Checks over a trip-update Observation frame alone — no GTFS static feed
needed. Every function takes an already-read `OBSERVATION_TU_SCHEMA` frame
(see `stoptimes.core.readers`) and returns a `Finding | None` (`None` =
nothing to report). See `vp_checks.py` for the vehicle-position counterparts
and `static_checks.py` for checks that need the GTFS static feed.

All checks are vectorized pandas (`groupby`/`diff`/boolean masks) — no
per-row Python loops. Every function must also do something reasonable
(usually: `None`, or an INFO/BLOCKER finding saying so) when the frame is
missing or empty — a newcomer's feed is exactly the case with holes in it.
"""

from __future__ import annotations

import pandas as pd
from stoptimes.core.diagnose._util import bounded_samples
from stoptimes.core.diagnose.models import DiagnoseParams, Finding, Severity




def check_tu_presence(tu_df: pd.DataFrame | None, params: DiagnoseParams) -> Finding | None:
    if tu_df is not None and len(tu_df) > 0:
        return None
    return Finding(
        id="tu_missing",
        severity=Severity.DEGRADED,
        title="No trip updates provided",
        explanation="No trip_update observations were supplied (empty frame or none given).",
        blocks=(
            "There is no independently-published prediction to validate against, and no "
            "TU-reported arrivals to cross-check derived (map-matched) ones. Training from "
            "vehicle positions alone is still possible."
        ),
        recommendation="Record or point diagnose at a trip_updates GTFS-RT feed or archive if you plan to validate or serve TripUpdates.",
    )


def check_tu_prediction_validity(tu_df: pd.DataFrame | None, params: DiagnoseParams) -> Finding | None:
    if tu_df is None or tu_df.empty:
        return None
    have_arrival = "arrival_time" in tu_df.columns
    have_departure = "departure_time" in tu_df.columns
    if not have_arrival and not have_departure:
        return None
    if have_arrival and have_departure:
        missing = tu_df["arrival_time"].isna() & tu_df["departure_time"].isna()
    elif have_arrival:
        missing = tu_df["arrival_time"].isna()
    else:
        missing = tu_df["departure_time"].isna()
    n = int(missing.sum())
    if n == 0:
        return None
    share = n / len(tu_df)
    return Finding(
        id="tu_missing_times",
        severity=Severity.DEGRADED,
        title="Trip-update rows with neither an arrival nor a departure time",
        explanation=f"{n} of {len(tu_df)} stop_time_update rows carry neither an arrival nor a departure time.",
        blocks="These rows carry no usable prediction and cannot serve as a label or a comparison point.",
        recommendation="Confirm the upstream feed actually populates arrival/departure on stop_time_update.",
        count=n,
        total=int(len(tu_df)),
        share=share,
        samples=bounded_samples(tu_df.loc[missing, "trip_id"].dropna(), params) if "trip_id" in tu_df.columns else [],
    )


def check_tu_duplicates(tu_df: pd.DataFrame | None, params: DiagnoseParams) -> Finding | None:
    if tu_df is None or tu_df.empty:
        return None
    key_cols = [c for c in ("trip_id", "start_date", "stop_id", "stop_sequence", "ts") if c in tu_df.columns]
    if len(key_cols) < 3:
        return None
    working = tu_df.dropna(subset=key_cols)
    if working.empty:
        return None
    extra = int(working.duplicated(subset=key_cols, keep="first").sum())
    if extra == 0:
        return None
    dup_mask = working.duplicated(subset=key_cols, keep="first")
    return Finding(
        id="tu_duplicate_predictions",
        severity=Severity.DEGRADED,
        title="Duplicate trip-update predictions",
        explanation=f"{extra} of {len(tu_df)} rows repeat an earlier {key_cols} tuple exactly.",
        blocks="Duplicates double-count predictions in any per-trip/per-stop aggregation or scoring.",
        recommendation=f"De-duplicate on {key_cols} before training or validating.",
        count=extra,
        total=int(len(tu_df)),
        share=extra / len(tu_df),
        samples=bounded_samples(working.loc[dup_mask, "trip_id"], params),
    )


def check_tu_out_of_order(tu_df: pd.DataFrame | None, params: DiagnoseParams) -> Finding | None:
    if tu_df is None or tu_df.empty:
        return None
    required = {"trip_id", "stop_sequence"}
    if not required <= set(tu_df.columns):
        return None
    have_arrival = "arrival_time" in tu_df.columns
    have_departure = "departure_time" in tu_df.columns
    if not have_arrival and not have_departure:
        return None
    working = tu_df.dropna(subset=["trip_id", "stop_sequence"]).copy()
    if have_arrival and have_departure:
        working["_time"] = working["arrival_time"].fillna(working["departure_time"])
    else:
        working["_time"] = working["arrival_time"] if have_arrival else working["departure_time"]
    working = working.dropna(subset=["_time"])
    if working.empty:
        return None
    group_keys = [c for c in ("trip_id", "start_date", "ts") if c in working.columns]
    working = working.sort_values(group_keys + ["stop_sequence"])
    prev_time = working.groupby(group_keys)["_time"].shift(1)
    prev_seq = working.groupby(group_keys)["stop_sequence"].shift(1)
    out_of_order = (working["_time"] < prev_time) & prev_seq.notna()
    n = int(out_of_order.sum())
    if n == 0:
        return None
    return Finding(
        id="tu_out_of_order_arrivals",
        severity=Severity.DEGRADED,
        title="Trip-update predictions going backward in stop order",
        explanation=(
            f"{n} of {len(working)} stop_time_update rows predict an arrival/departure earlier "
            f"than the previous stop_sequence within the same trip/start_date/publish."
        ),
        blocks="A schedule that isn't monotonic along the trip breaks any downstream logic assuming stop order implies time order (horizon computation, next-stop selection).",
        recommendation="Check the upstream predictor for stop_sequence handling bugs, or filter these rows before using them as labels.",
        count=n,
        total=int(len(working)),
        share=n / len(working),
        samples=bounded_samples(working.loc[out_of_order, "trip_id"], params),
    )


def check_tu_staleness(tu_df: pd.DataFrame | None, params: DiagnoseParams) -> Finding | None:
    if tu_df is None or tu_df.empty or "ingested_at" not in tu_df.columns:
        return None
    if tu_df["ingested_at"].isna().all():
        return None
    working = tu_df.dropna(subset=["ts", "ingested_at"])
    if working.empty:
        return None
    skew = (working["ingested_at"] - working["ts"]).dt.total_seconds()
    stale = skew[skew > params.stale_threshold_s]
    if stale.empty:
        return None
    return Finding(
        id="tu_stale_timestamps",
        severity=Severity.DEGRADED,
        title="Stale trip-update timestamps",
        explanation=(
            f"{len(stale)} of {len(working)} rows were ingested more than "
            f"{params.stale_threshold_s:.0f}s after the update's own `ts` "
            f"(median skew {float(skew.median()):.0f}s, max {float(skew.max()):.0f}s)."
        ),
        blocks="A prediction this old when received may already be superseded by a fresher one the poller missed.",
        recommendation="Check the poller's fetch cadence and the upstream feed's own publishing latency.",
        count=int(len(stale)),
        total=int(len(working)),
        share=len(stale) / len(working),
        samples=bounded_samples(working.loc[stale.index, "trip_id"], params) if "trip_id" in working.columns else [],
        detail={"skew_p50_s": float(skew.median()), "skew_max_s": float(skew.max())},
    )
