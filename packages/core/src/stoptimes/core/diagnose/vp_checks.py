"""Checks over a vehicle-position Observation frame alone — no GTFS static
feed needed. Every function takes an already-read `OBSERVATION_VP_SCHEMA`
frame (see `stoptimes.core.readers`) and returns a `Finding | None` (`None`
= nothing to report). See `tu_checks.py` for the trip-update counterparts
and `static_checks.py` for checks that need the GTFS static feed.

All checks are vectorized pandas (`groupby`/`diff`/boolean masks) — no
per-row Python loops — so a check stays sensible on an archive with a month
of positions in it. Every function must also do something reasonable
(usually: `None`, or an INFO/BLOCKER finding saying so) when the frame is
missing or empty — a newcomer's feed is exactly the case with holes in it.
"""

from __future__ import annotations

import pandas as pd
from stoptimes.core.diagnose._util import bounded_samples
from stoptimes.core.diagnose.models import DiagnoseParams, Finding, Severity


def check_vp_presence(vp_df: pd.DataFrame | None, params: DiagnoseParams) -> Finding | None:
    if vp_df is not None and len(vp_df) > 0:
        return None
    return Finding(
        id="vp_missing",
        severity=Severity.BLOCKER,
        title="No vehicle positions provided",
        explanation="No vehicle_position observations were supplied (empty frame or none given).",
        blocks=(
            "Actual stop arrivals cannot be map-matched or interpolated from vehicle "
            "positions; the only possible ground truth is whatever a trip-update feed "
            "already reports (if one is present), which is only as good as its own "
            "upstream predictions."
        ),
        recommendation=(
            "Record or point diagnose at a vehicle_positions GTFS-RT feed or archive "
            "(`stoptimes record`, or `diagnose --vehicle-positions`)."
        ),
    )


def check_vp_position_validity(vp_df: pd.DataFrame | None, params: DiagnoseParams) -> Finding | None:
    if vp_df is None or vp_df.empty or not {"lat", "lon"} <= set(vp_df.columns):
        return None
    missing = vp_df["lat"].isna() | vp_df["lon"].isna()
    n = int(missing.sum())
    if n == 0:
        return None
    share = n / len(vp_df)
    ids = vp_df.loc[missing, "vehicle_id"] if "vehicle_id" in vp_df.columns else pd.Series([], dtype=object)
    return Finding(
        id="vp_missing_lat_lon",
        severity=Severity.DEGRADED,
        title="Vehicle positions with no usable lat/lon",
        explanation=f"{n} of {len(vp_df)} vehicle_position rows have a missing latitude or longitude.",
        blocks="These rows cannot be projected onto a shape and contribute nothing to map-matching.",
        recommendation="Check the source feed/AVL export for rows with blank coordinate fields.",
        count=n,
        total=int(len(vp_df)),
        share=share,
        samples=bounded_samples(ids.dropna(), params),
    )


def check_vp_gaps(vp_df: pd.DataFrame | None, params: DiagnoseParams) -> Finding | None:
    if vp_df is None or vp_df.empty or "ts" not in vp_df.columns:
        return None
    group_col = (
        "vehicle_id" if "vehicle_id" in vp_df.columns and vp_df["vehicle_id"].notna().any() else "trip_id"
    )
    if group_col not in vp_df.columns:
        return None
    working = vp_df.dropna(subset=["ts", group_col]).sort_values([group_col, "ts"]).reset_index(drop=True)
    if len(working) < 2:
        return None
    working["gap_s"] = working.groupby(group_col)["ts"].diff().dt.total_seconds()
    gaps = working["gap_s"].dropna()
    if gaps.empty:
        return None
    over = gaps[gaps > params.gap_threshold_s]

    by_route: dict[str, int] = {}
    if "route_id" in working.columns:
        over_rows = working.loc[over.index]
        by_route = {str(k): int(v) for k, v in over_rows.groupby("route_id").size().items()}
    by_hour = {
        str(h): int(c) for h, c in vp_df["ts"].dropna().dt.hour.value_counts().sort_index().items()
    }

    worst_index = (over.index if len(over) else gaps.index)
    worst = working.loc[worst_index].sort_values("gap_s", ascending=False).head(params.max_samples)
    samples = [
        f"{row[group_col]}: {row['gap_s']:.0f}s gap ending {row['ts'].isoformat()}"
        for _, row in worst.iterrows()
    ]

    detail = {
        "sample_size": int(len(gaps)),
        "p50_s": float(gaps.median()),
        "p95_s": float(gaps.quantile(0.95)),
        "max_s": float(gaps.max()),
        "gaps_over_threshold": int(len(over)),
        "gaps_by_route": by_route,
        "rows_by_hour_utc": by_hour,
    }
    has_gaps = len(over) > 0
    return Finding(
        id="vp_sampling_gaps",
        severity=Severity.DEGRADED if has_gaps else Severity.INFO,
        title=(
            "Vehicle-position sampling gaps"
            if has_gaps
            else "Vehicle-position sampling interval distribution"
        ),
        explanation=(
            f"Consecutive-position intervals per {group_col}: median {detail['p50_s']:.0f}s, "
            f"p95 {detail['p95_s']:.0f}s, longest {detail['max_s']:.0f}s. {len(over)} of "
            f"{len(gaps)} intervals exceed the {params.gap_threshold_s:.0f}s gap threshold."
        ),
        blocks=(
            "Long gaps leave stretches of a trip with no position to map-match against, so "
            "arrivals there fall back to interpolation across a wider span (less accurate) or "
            "are missed entirely."
            if has_gaps
            else "Nothing — no interval exceeds the configured gap threshold."
        ),
        recommendation=(
            "Compare the actual interval to the agency's advertised poll_interval_s, and see "
            "gaps_by_route / rows_by_hour_utc in this finding's detail for where coverage is worst."
            if has_gaps
            else "None needed."
        ),
        count=int(len(over)),
        total=int(len(gaps)),
        share=(len(over) / len(gaps)) if len(gaps) else None,
        samples=samples,
        detail=detail,
    )


def check_vp_duplicates(vp_df: pd.DataFrame | None, params: DiagnoseParams) -> Finding | None:
    if vp_df is None or vp_df.empty:
        return None
    use_vehicle = "vehicle_id" in vp_df.columns and vp_df["vehicle_id"].notna().any()
    key_cols = ["vehicle_id", "ts"] if use_vehicle else ["trip_id", "ts"]
    if not set(key_cols) <= set(vp_df.columns):
        return None
    working = vp_df.dropna(subset=key_cols)
    if working.empty:
        return None
    extra = int(working.duplicated(subset=key_cols, keep="first").sum())
    if extra == 0:
        return None
    dup_mask = working.duplicated(subset=key_cols, keep="first")
    samples = bounded_samples(
        (working.loc[dup_mask, key_cols[0]].astype(str) + "@" + working.loc[dup_mask, "ts"].astype(str)),
        params,
    )
    return Finding(
        id="vp_duplicate_positions",
        severity=Severity.DEGRADED,
        title="Duplicate vehicle-position records",
        explanation=(
            f"{extra} of {len(vp_df)} rows repeat an earlier ({key_cols[0]}, ts) pair exactly."
        ),
        blocks="Duplicates inflate row counts, skew gap-interval statistics, and can double-count arrivals.",
        recommendation="De-duplicate on (vehicle_id, ts) before training or diagnosing further.",
        count=extra,
        total=int(len(vp_df)),
        share=extra / len(vp_df),
        samples=samples,
    )


def check_vp_out_of_order(vp_df: pd.DataFrame | None, params: DiagnoseParams) -> Finding | None:
    if vp_df is None or vp_df.empty or "ingested_at" not in vp_df.columns:
        return None
    if vp_df["ingested_at"].isna().all():
        return None
    group_col = (
        "vehicle_id" if "vehicle_id" in vp_df.columns and vp_df["vehicle_id"].notna().any() else "trip_id"
    )
    if group_col not in vp_df.columns:
        return None
    working = vp_df.dropna(subset=["ts", "ingested_at", group_col]).sort_values([group_col, "ingested_at"])
    if len(working) < 2:
        return None
    prev_ts = working.groupby(group_col)["ts"].shift(1)
    out_of_order = (working["ts"] < prev_ts) & prev_ts.notna()
    n = int(out_of_order.sum())
    if n == 0:
        return None
    return Finding(
        id="vp_out_of_order_timestamps",
        severity=Severity.DEGRADED,
        title="Vehicle positions arriving out of timestamp order",
        explanation=(
            f"{n} of {len(working)} rows have an earlier `ts` than the previous record received "
            f"(sorted by `ingested_at`) for the same {group_col}."
        ),
        blocks="Code that assumes positions are monotonic in time (gap detection, interpolation) will misread these as a jump backward.",
        recommendation="Sort by ts before any sequential processing rather than assuming feed delivery order is time order.",
        count=n,
        total=int(len(working)),
        share=n / len(working),
        samples=bounded_samples(working.loc[out_of_order, group_col], params),
    )


def check_vp_staleness(vp_df: pd.DataFrame | None, params: DiagnoseParams) -> Finding | None:
    if vp_df is None or vp_df.empty or "ingested_at" not in vp_df.columns:
        return None
    if vp_df["ingested_at"].isna().all():
        return Finding(
            id="vp_staleness_unchecked",
            severity=Severity.INFO,
            title="Vehicle-position staleness could not be checked",
            explanation=(
                "No `ingested_at` value is present on any row (common for an AVL CSV import), "
                "so staleness (position time vs. time of receipt) can't be measured."
            ),
            blocks="Nothing directly, but a stale upstream feed would go undetected.",
            recommendation="If this data came from a live poll, keep the reader-populated ingested_at column.",
        )
    working = vp_df.dropna(subset=["ts", "ingested_at"])
    if working.empty:
        return None
    skew = (working["ingested_at"] - working["ts"]).dt.total_seconds()
    stale = skew[skew > params.stale_threshold_s]
    if stale.empty:
        return None
    id_col = "vehicle_id" if "vehicle_id" in working.columns else "trip_id"
    return Finding(
        id="vp_stale_timestamps",
        severity=Severity.DEGRADED,
        title="Stale vehicle-position timestamps",
        explanation=(
            f"{len(stale)} of {len(working)} rows were ingested more than "
            f"{params.stale_threshold_s:.0f}s after the position's own `ts` "
            f"(median skew {float(skew.median()):.0f}s, max {float(skew.max()):.0f}s)."
        ),
        blocks="A position this old when received no longer reflects where the vehicle actually is; ETAs computed from it will be off by roughly the skew.",
        recommendation="Check the poller's fetch cadence and the upstream feed's own publishing latency.",
        count=int(len(stale)),
        total=int(len(working)),
        share=len(stale) / len(working),
        samples=bounded_samples(working.loc[stale.index, id_col].dropna(), params),
        detail={"skew_p50_s": float(skew.median()), "skew_max_s": float(skew.max())},
    )
