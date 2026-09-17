"""S3-backed build driver for the MBTA segment ETA dataset (deliverable B).

Pulls raw MBTA VehiclePositions for a small window (routes 222/15/37),
resolves the static GTFS snapshot in effect for that window, and builds the
segment dataset via ``feature_engineering.mbta_segments.build_mbta_segment_dataset``
-- same schema as the bUCR segment corpus.

READ-ONLY on S3: only ``read_vehicle_positions``, ``list_snapshots``,
``get_snapshot`` are used. Nothing is written back to S3.

Run (from repo root)::

    set -a; . gtfs-rt-pipeline/.env; set +a
    PYTHONPATH=.:gtfs-rt-pipeline uv run --group bucr --with duckdb \
        python scratchpad/build_mbta_segments.py
"""

from __future__ import annotations

import datetime as dt
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "gtfs-rt-pipeline"))

import pandas as pd

from feature_engineering.bucr_gtfs import load_gtfs
from feature_engineering.mbta_segments import build_mbta_segment_dataset
from rt_pipeline.storage.s3_writer import read_vehicle_positions
from rt_pipeline.storage.static_gtfs import get_snapshot, latest_snapshot_on_or_before

ROUTE_IDS = ["222", "15", "37"]
# Snapshots only exist from 2026-08-14 onward -- keep the window small and
# entirely after that date. Do NOT reach back into the July archive (predates
# all snapshots; latest_snapshot_on_or_before returns None there).
WINDOW_START = dt.datetime(2026, 8, 17, 0, 0, 0, tzinfo=dt.timezone.utc)
WINDOW_END = dt.datetime(2026, 8, 19, 0, 0, 0, tzinfo=dt.timezone.utc)
WINDOW_LABEL = "20260817_20260819"

OUT_PATH = REPO_ROOT / "datasets" / f"mbta_segments_{WINDOW_LABEL}.parquet"


def main() -> None:
    print(f"Reading MBTA VehiclePositions routes={ROUTE_IDS} "
          f"[{WINDOW_START.isoformat()}, {WINDOW_END.isoformat()})")
    vps = read_vehicle_positions(route_ids=ROUTE_IDS, start=WINDOW_START, end=WINDOW_END)
    print(f"  -> {len(vps):,} raw VP rows")
    if vps.empty:
        print("No VPs in window -- aborting.")
        return

    window_date = WINDOW_START.date()
    snapshot_date = latest_snapshot_on_or_before("mbta", window_date)
    if snapshot_date is None:
        print(f"No MBTA GTFS snapshot on or before {window_date} -- aborting.")
        return
    print(f"Static GTFS snapshot used: {snapshot_date.isoformat()}")

    zip_bytes = get_snapshot("mbta", snapshot_date)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp.write(zip_bytes)
        tmp_path = tmp.name

    try:
        gtfs = load_gtfs(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    print(f"GTFS loaded: {len(gtfs.routes)} routes, {len(gtfs.trips)} trips, "
          f"{len(gtfs.stops)} stops, {gtfs.shapes['shape_id'].nunique()} shapes")

    segments = build_mbta_segment_dataset(vps, gtfs)
    print(f"Segments built: {len(segments):,}")
    if segments.empty:
        print("No segments produced -- aborting write.")
        return

    n_trips = segments["trip_id"].nunique()
    n_days = pd.to_datetime(segments["vp_ts"], utc=True).dt.date.nunique()
    tta = segments["time_to_arrival_seconds"]
    print(f"n_segments={len(segments):,}  n_trips={n_trips}  n_days={n_days}")
    print(
        "traversal seconds -- median={:.1f}  mean={:.1f}  p95={:.1f}".format(
            tta.median(), tta.mean(), tta.quantile(0.95)
        )
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    segments.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} ({len(segments):,} rows)")


if __name__ == "__main__":
    main()
