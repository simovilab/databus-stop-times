"""Multi-snapshot MBTA segment corpus build (bigger, day-grouped-ready).

Builds the MBTA segment ETA dataset over a multi-week window that spans several
weekly static-GTFS snapshots. Each VP sub-window is built against the snapshot
in effect at that time, then all sub-windows are concatenated. Same 35-column
schema as the bUCR corpus, so the shared trainers / leakage probe / metrics all
apply.

Window: 2026-08-14 -> 2026-08-30 (16 continuous days, routes 222/15/37).
Snapshot epochs (from list_snapshots): 08-14, 08-17, 08-24. Each VP is built
against the latest snapshot on or before its own date.

READ-ONLY on S3 (read_vehicle_positions / list_snapshots / get_snapshot only).

Run (from repo root)::

    set -a; . gtfs-rt-pipeline/.env; set +a
    PYTHONPATH=.:gtfs-rt-pipeline uv run --group bucr --with duckdb \
        python scratchpad/build_mbta_segments_corpus.py
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
from rt_pipeline.storage.static_gtfs import get_snapshot, list_snapshots

ROUTE_IDS = ["222", "15", "37"]
WINDOW_START = dt.datetime(2026, 8, 14, 0, 0, 0, tzinfo=dt.timezone.utc)
WINDOW_END = dt.datetime(2026, 8, 30, 0, 0, 0, tzinfo=dt.timezone.utc)

OUT_PATH = REPO_ROOT / "datasets" / "mbta_segments_corpus.parquet"


def _snapshot_epochs(snaps: list[dt.date]) -> list[tuple[dt.datetime, dt.datetime, dt.date]]:
    """Split [WINDOW_START, WINDOW_END) into [epoch_start, epoch_end) sub-windows,
    each tagged with the snapshot date in effect for it. Snapshots before the
    window collapse into the first epoch (their effective start is clamped up)."""
    snaps = sorted(snaps)
    # effective start of each snapshot's epoch, clamped to the window start
    starts = []
    for s in snaps:
        s_dt = dt.datetime(s.year, s.month, s.day, tzinfo=dt.timezone.utc)
        starts.append(max(s_dt, WINDOW_START))
    epochs = []
    for i, s in enumerate(snaps):
        e_start = starts[i]
        e_end = starts[i + 1] if i + 1 < len(snaps) else WINDOW_END
        if e_start >= e_end:
            continue  # snapshot fully before the window or zero-width
        epochs.append((e_start, e_end, s))
    return epochs


def main() -> None:
    snaps = [s for s in list_snapshots("mbta") if s <= WINDOW_END.date()]
    if not snaps:
        print("No MBTA snapshots available -- aborting.")
        return
    epochs = _snapshot_epochs(snaps)
    print(f"snapshot epochs ({len(epochs)}):")
    for a, b, s in epochs:
        print(f"  [{a.date()} .. {b.date()})  ->  snapshot {s.isoformat()}")

    frames = []
    for e_start, e_end, snap_date in epochs:
        print(f"\n=== epoch [{e_start.date()} .. {e_end.date()})  snapshot {snap_date} ===")
        vps = read_vehicle_positions(route_ids=ROUTE_IDS, start=e_start, end=e_end)
        print(f"  VP rows: {len(vps):,}")
        if vps.empty:
            continue
        zip_bytes = get_snapshot("mbta", snap_date)
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp.write(zip_bytes)
            tmp_path = tmp.name
        try:
            gtfs = load_gtfs(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        seg = build_mbta_segment_dataset(vps, gtfs)
        print(f"  segments: {len(seg):,}")
        if not seg.empty:
            frames.append(seg)

    if not frames:
        print("\nNo segments produced across any epoch -- aborting.")
        return

    corpus = pd.concat(frames, ignore_index=True)
    n_trips = corpus["trip_id"].nunique()
    n_days = pd.to_datetime(corpus["vp_ts"], utc=True).dt.date.nunique()
    tta = corpus["time_to_arrival_seconds"]
    print(f"\n=== CORPUS ===")
    print(f"n_segments={len(corpus):,}  n_trips={n_trips}  n_days={n_days}")
    print("traversal seconds -- median={:.1f}  mean={:.1f}  p95={:.1f}".format(
        tta.median(), tta.mean(), tta.quantile(0.95)))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} ({len(corpus):,} rows)")


if __name__ == "__main__":
    main()
