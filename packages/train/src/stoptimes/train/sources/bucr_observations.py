"""bUCR navsat pings -> GTFS-RT-shaped VehiclePosition archive, trip ids from the timetable.

Pipeline (one service day per vehicle, America/Costa_Rica from agency.txt)::

    raw navsat parquet -> normalize -> clean -> adapt (canonical VP)
        -> corridor_legs.segment_legs (turning-point legs, geometric variant prior)
        -> schedule_match (stop passages -> one-to-one scheduled trip per leg)
        -> archive parquet: year=/month=/day=/route_id=/part.parquet with the scheduled
           trip_id, direction_id and start_date, i.e. what a GTFS-RT feed would carry

so ``stoptimes data frame --live --observations <out>`` builds the same 19-feature frame as
for MBTA. Matches whose ``margin_s`` is below ``--min-margin-s`` are dropped (ambiguous).
A negative margin means the one-to-one assignment gave this leg's best-fitting trip to
another leg, so the default of 0 keeps only uncontested matches (semester: 5,844 of 6,110).
Also writes ``legs.parquet`` (one row per leg with its match) and ``report.json``.

    python -m stoptimes.train.sources.bucr_observations \\
        --navsat-dir <dir of daily parquet> --gtfs-zip <gtfs.zip> --out <dir> [--min-margin-s 60]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from collections.abc import Sequence
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from stoptimes.core.spatial.polyline import project_point_to_polyline
from stoptimes.train.sources.bucr_gtfs import (
    load_gtfs,
    route_direction_candidates,
    route_directions,
)
from stoptimes.train.sources.corridor_legs import (
    CorridorLayout,
    LegParams,
    corridor_layout,
    segment_legs,
)
from stoptimes.train.sources.navsat_adapter import navsat_to_vehicle_positions
from stoptimes.train.sources.navsat_cleaning import clean_navsat_trace
from stoptimes.train.sources.navsat_normalize import normalize_raw_navsat
from stoptimes.train.sources.schedule_match import (
    MatchParams,
    match_instances,
    scheduled_stop_times,
    stop_passages,
)

__all__ = ["active_trips_fn", "legs_from_pings", "build_observations", "main"]

_STOPPED = "STOPPED_AT"


def _read_zip_table(gtfs_zip: Path, name: str) -> pd.DataFrame:
    with zipfile.ZipFile(gtfs_zip) as zf:
        if name not in zf.namelist():
            return pd.DataFrame()
        return pd.read_csv(io.BytesIO(zf.read(name)), dtype=str)


def active_trips_fn(gtfs_zip: Path, trips: pd.DataFrame):
    """``service_date (YYYYMMDD) -> trip_ids`` from calendar.txt and calendar_dates.txt."""
    cal = _read_zip_table(gtfs_zip, "calendar.txt")
    exc = _read_zip_table(gtfs_zip, "calendar_dates.txt")
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    by_service = (
        trips.groupby("service_id")["trip_id"].apply(lambda s: set(s.astype(str))).to_dict()
    )

    def active(service_date: str) -> set[str]:
        d = date(int(service_date[:4]), int(service_date[4:6]), int(service_date[6:]))
        services = {
            str(r.service_id)
            for r in cal.itertuples()
            if r.start_date <= service_date <= r.end_date and getattr(r, days[d.weekday()]) == "1"
        }
        for r in exc[exc.get("date", pd.Series(dtype=str)) == service_date].itertuples():
            (services.add if r.exception_type == "1" else services.discard)(str(r.service_id))
        return set().union(*(by_service.get(s, set()) for s in services))

    return active


def _canonical(raw: pd.DataFrame) -> pd.DataFrame:
    cleaned, _ = clean_navsat_trace(normalize_raw_navsat(raw))
    vp = navsat_to_vehicle_positions(cleaned)
    vp = vp.sort_values(["vehicle_id", "ts"]).drop_duplicates(["vehicle_id", "ts"], keep="last")
    vp["stopped"] = (vp["current_status"] == _STOPPED) | (vp["speed"].fillna(0) <= 0)
    return vp.reset_index(drop=True)


def legs_from_pings(
    vp: pd.DataFrame, layout: CorridorLayout, tz: str, params: LegParams
) -> pd.DataFrame:
    """Canonical VP (all vehicles, any days) -> pings tagged with ``instance_id``, ``shape_id``."""
    vp = vp.assign(local_date=vp["ts"].dt.tz_convert(tz).dt.strftime("%Y%m%d"))
    parts = []
    for (vehicle, day), g in vp.groupby(["vehicle_id", "local_date"], sort=True):
        g = g.reset_index(drop=True)
        for leg in segment_legs(g, layout, params):
            seg = g.iloc[leg.start : leg.end + 1]
            start = seg["ts"].iloc[0].tz_convert("UTC").strftime("%Y%m%dT%H%M%SZ")
            parts.append(
                seg.assign(
                    instance_id=f"bucr:{vehicle}:{leg.direction_id}:{leg.shape_id}:{start}",
                    service_date=day,
                    direction_id=leg.direction_id,
                    shape_id=leg.shape_id,
                )
            )
    return pd.concat(parts, ignore_index=True) if parts else vp.iloc[0:0]


def _passages(pings: pd.DataFrame, layout: CorridorLayout, tz: str) -> pd.DataFrame:
    rows = []
    for inst, g in pings.groupby("instance_id", sort=False):
        cand = layout.candidates[str(g["shape_id"].iloc[0])]
        midnight = pd.Timestamp(str(g["service_date"].iloc[0])).tz_localize(tz).timestamp()
        got = stop_passages(
            g,
            [(s.stop_id, s.progress_m) for s in cand.stops],
            lambda a, b, poly=cand.polyline: project_point_to_polyline(a, b, poly).progress_m,
        )
        rows += [(inst, stop, t - midnight) for stop, t in got.items()]
    return pd.DataFrame(rows, columns=["instance_id", "stop_id", "passage_s"])


def build_observations(
    navsat_files: Sequence[Path],
    gtfs_zip: Path,
    *,
    leg_params: LegParams | None = None,
    match_params: MatchParams | None = None,
    min_margin_s: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """``(observations, legs, report)``; observations are archive-shaped VP rows."""
    leg_params, match_params = leg_params or LegParams(), match_params or MatchParams()
    gtfs = load_gtfs(gtfs_zip)
    candidates = [
        c for r, d in route_directions(gtfs) for c in route_direction_candidates(gtfs, r, d)
    ]
    layout = corridor_layout(candidates)
    agency = _read_zip_table(gtfs_zip, "agency.txt")
    tz = str(agency["agency_timezone"].iloc[0])
    raw = pd.concat([pd.read_parquet(f) for f in navsat_files], ignore_index=True)
    vp = _canonical(raw)
    pings = legs_from_pings(vp, layout, tz, leg_params)
    instances = pings.drop_duplicates("instance_id")[
        ["instance_id", "service_date", "direction_id", "shape_id"]
    ]
    trips = gtfs.trips.astype({"trip_id": str})
    matches = match_instances(
        _passages(pings, layout, tz),
        instances,
        scheduled_stop_times(gtfs.stop_times, trips),
        active_trips_fn(gtfs_zip, trips),
        match_params,
    )
    keep = matches[matches["matched"] & (matches["margin_s"] >= min_margin_s)]
    obs = pings.merge(keep[["instance_id", "trip_id"]], on="instance_id", how="inner")
    route = trips.set_index("trip_id")["route_id"]
    obs = pd.DataFrame(
        {
            "feed_name": "bucr_navsat_inferred",
            "vehicle_id": obs["vehicle_id"].astype(str),
            "ts": obs["ts"],
            "lat": obs["lat"],
            "lon": obs["lon"],
            "bearing": np.nan,
            "speed": obs["speed"],
            "trip_id": obs["trip_id"],
            "route_id": obs["trip_id"].map(route),
            "direction_id": obs["direction_id"].astype(int),
            "start_date": obs["service_date"],
            "current_stop_sequence": pd.NA,
            "current_status": obs["current_status"],
            "stop_id": pd.NA,
            "ingested_at": obs["ts"],  # derived rows: the fix time stands in for ingestion
        }
    )
    legs = instances.merge(
        matches.drop(columns=["service_date", "direction_id"]), on="instance_id", how="left"
    )
    report = _report(raw, vp, pings, matches, keep, layout, leg_params, match_params, min_margin_s)
    return obs, legs, report


def _report(raw, vp, pings, matches, keep, layout, leg_params, match_params, min_margin_s) -> dict:
    m = matches[matches["matched"]]
    weekdays = m["service_date"].nunique()
    return {
        "raw_pings": int(len(raw)),
        "clean_pings": int(len(vp)),
        "pings_in_legs": int(len(pings)),
        "pings_in_legs_share": round(len(pings) / max(len(vp), 1), 4),
        "legs": int(len(matches)),
        "legs_matched": int(len(m)),
        "legs_kept": int(len(keep)),
        "pattern_agreement": round(float(m["pattern_agrees"].mean()), 4) if len(m) else None,
        "offset_s_quantiles": {
            str(q): round(float(m["offset_s"].quantile(q)), 1)
            for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        }
        if len(m)
        else {},
        "median_spread_s": round(float(m["spread_s"].median()), 1) if len(m) else None,
        "matched_trips_per_service_day": round(len(m) / max(weekdays, 1), 1),
        "service_days_matched": int(weekdays),
        "terminals": {"low": list(layout.low_terminals), "high": list(layout.high_terminals)},
        "detour_stops": list(layout.detour_stops),
        "leg_params": asdict(leg_params),
        "match_params": asdict(match_params),
        "min_margin_s": min_margin_s,
    }


def _write_archive(obs: pd.DataFrame, out: Path) -> int:
    n = 0
    for (day, route), g in obs.groupby(["start_date", "route_id"], sort=True):
        d = out / f"year={day[:4]}" / f"month={day[4:6]}" / f"day={day[6:]}" / f"route_id={route}"
        d.mkdir(parents=True, exist_ok=True)
        g.drop(columns=["route_id"]).sort_values(["vehicle_id", "ts"]).to_parquet(
            d / "part.parquet", index=False
        )
        n += 1
    return n


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--navsat-dir",
        type=Path,
        required=True,
        help="directory searched recursively for *.parquet",
    )
    p.add_argument("--gtfs-zip", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--min-margin-s",
        type=float,
        default=0.0,
        help="drop matches whose margin_s is below this; 0 (default) drops contested ones",
    )
    args = p.parse_args(argv)
    files = sorted(args.navsat_dir.rglob("*.parquet"))
    if not files:
        print(f"no *.parquet under {args.navsat_dir}", file=sys.stderr)
        return 1
    obs, legs, report = build_observations(files, args.gtfs_zip, min_margin_s=args.min_margin_s)
    args.out.mkdir(parents=True, exist_ok=True)
    report["archive_files"] = _write_archive(obs, args.out / "archive")
    legs.to_parquet(args.out / "legs.parquet", index=False)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in list(report)[:12]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
