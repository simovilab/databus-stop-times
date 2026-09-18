"""Coverage report for `stoptimes data inspect`: rows, date range, trips/routes
seen, share of rows with a usable timestamp/position, gaps in the stream, and
(optionally) how many observed trips match a GTFS static feed.

Pure function over an already-read Observation DataFrame (see
`stoptimes.core.readers`) — no I/O here, so it's cheaply unit-testable
without a real feed or archive.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

DEFAULT_GAP_THRESHOLD_S = 300.0


def build_coverage_report(
    df: pd.DataFrame,
    kind: str,
    *,
    static_trip_ids: set[str] | None = None,
    gap_threshold_s: float = DEFAULT_GAP_THRESHOLD_S,
) -> dict[str, Any]:
    """Summarize an Observation frame's coverage.

    Args:
        df: an Observation DataFrame (`OBSERVATION_VP_SCHEMA`/`_TU_SCHEMA` shape).
        kind: `"vehicle_position"` or `"trip_update"` — position-share only
            applies to vehicle positions.
        static_trip_ids: `trip_id` values from a GTFS static feed, if the
            caller wants "trips matched" reported; `None` skips it.
        gap_threshold_s: a gap between consecutive (sorted) timestamps wider
            than this counts toward `gap_count`.
    """
    report: dict[str, Any] = {"rows": int(len(df))}
    if len(df) == 0:
        report.update(
            {
                "date_range": None,
                "trips_seen": 0,
                "routes_seen": 0,
                "usable_timestamp_share": None,
                "usable_position_share": None,
                "gap_count": 0,
                "max_gap_s": None,
                "trips_matched_static": None,
            }
        )
        return report

    ts = df["ts"] if "ts" in df.columns else pd.Series([], dtype="datetime64[ns, UTC]")
    usable_ts = ts.notna()
    report["usable_timestamp_share"] = float(usable_ts.mean()) if len(ts) else None
    ts_present = ts.dropna()
    report["date_range"] = (
        [str(ts_present.min()), str(ts_present.max())] if not ts_present.empty else None
    )

    trips = set(df["trip_id"].dropna().unique().tolist()) if "trip_id" in df.columns else set()
    report["trips_seen"] = len(trips)
    routes = set(df["route_id"].dropna().unique().tolist()) if "route_id" in df.columns else set()
    report["routes_seen"] = len(routes)

    if kind == "vehicle_position" and {"lat", "lon"} <= set(df.columns):
        usable_position = df["lat"].notna() & df["lon"].notna()
        report["usable_position_share"] = float(usable_position.mean())
    else:
        report["usable_position_share"] = None

    sorted_ts = ts.dropna().sort_values()
    if len(sorted_ts) > 1:
        gaps_s = sorted_ts.diff().dropna().dt.total_seconds()
        big_gaps = gaps_s[gaps_s > gap_threshold_s]
        report["gap_count"] = int(len(big_gaps))
        report["max_gap_s"] = float(gaps_s.max())
    else:
        report["gap_count"] = 0
        report["max_gap_s"] = None

    if static_trip_ids is not None and trips:
        matched = trips & static_trip_ids
        report["trips_matched_static"] = {
            "matched": len(matched),
            "total": len(trips),
            "share": len(matched) / len(trips),
        }
    else:
        report["trips_matched_static"] = None

    return report
