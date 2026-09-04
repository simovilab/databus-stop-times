"""MBTA segment leaderboard vs the offset schedule (day-grouped holdout).

Mirrors scratchpad/baseline_compare_corpus.py (bUCR) for the MBTA segment
corpus. Because MBTA rows carry *real* GTFS trip_ids, the schedule baseline is
exact: the scheduled duration of each adjacent segment (from_stop -> to_stop) is
sched_arrival(to_stop) - sched_arrival(from_stop), read straight from the
snapshot's stop_times. This is offset-invariant (uses scheduled *durations*, not
absolute schedule-vs-actual), so a trip running late as a whole is not penalised.

Split: day-grouped walk-forward. MBTA service day boundary ~3am ET (=07:00 UTC in
EDT), so cr_day = (vp_ts - 7h).date keeps whole trips inside one day (trip-safe).

HARD CONSTRAINT: save_model=False everywhere -- nothing is written to
models/trained/ or registry.json. This is a research comparison, not a deploy.

GTFS-RT (agency TripUpdates) baseline is intentionally absent: MBTA TripUpdates
are not archived (roadmap 0.5), so that comparison is only available live via
etaval, not offline here.

Run (from repo root)::

    PYTHONPATH=. uv run --group bucr --with duckdb python scratchpad/baseline_compare_mbta.py
"""
from __future__ import annotations

import datetime as dt
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "gtfs-rt-pipeline"))
sys.path.insert(0, str((REPO_ROOT / "models").resolve()))

from historical_mean.train import train_historical_mean  # noqa: E402
from polyreg_distance.train import train_polyreg_distance  # noqa: E402
from polyreg_time.train import train_polyreg_time  # noqa: E402
from ewma.train import train_ewma  # noqa: E402
from xgb.train import train_xgboost  # noqa: E402

from feature_engineering.bucr_gtfs import load_gtfs  # noqa: E402
from rt_pipeline.storage.static_gtfs import get_snapshot, latest_snapshot_on_or_before  # noqa: E402

DATASET = "mbta_segments_corpus"
DS_PATH = str(REPO_ROOT / "datasets" / f"{DATASET}.parquet")
MIN_SEGS_PER_DAY = 200
N_TEST_DAYS = 3
N_VAL_DAYS = 3
SERVICE_DAY_OFFSET_H = 7  # 3am ET in EDT -> 07:00 UTC


def _arr_seconds(hhmmss: str) -> float:
    try:
        h, m, s = str(hhmmss).split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)
    except Exception:
        return float("nan")


def scheduled_segment_durations(snapshot_date: dt.date) -> dict:
    """(trip_id, from_stop_id, to_stop_id) -> scheduled duration seconds, from
    the given snapshot's stop_times (adjacent stops only, positive durations)."""
    zip_bytes = get_snapshot("mbta", snapshot_date)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp.write(zip_bytes)
        tmp_path = tmp.name
    try:
        g = load_gtfs(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    st = g.stop_times.copy()
    st["trip_id"] = st["trip_id"].astype(str)
    st["stop_id"] = st["stop_id"].astype(str)
    st["_sec"] = st["arrival_time"].map(_arr_seconds)
    st = st.sort_values(["trip_id", "stop_sequence"])
    lookup: dict = {}
    for trip_id, grp in st.groupby("trip_id", sort=False):
        secs = grp["_sec"].to_numpy()
        ids = grp["stop_id"].to_numpy()
        for k in range(len(grp) - 1):
            dur = float(secs[k + 1] - secs[k])
            if dur > 0:
                lookup[(trip_id, ids[k], ids[k + 1])] = dur
    return lookup


def dist_stats(err: np.ndarray) -> dict:
    a = np.abs(err[~np.isnan(err)])
    return {
        "mae": float(a.mean()),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
        "w30": float((a <= 30).mean() * 100),
        "w60": float((a <= 60).mean() * 100),
        "n": int(a.size),
    }


def main() -> None:
    df = pd.read_parquet(DS_PATH)
    df["trip_id"] = df["trip_id"].astype(str)
    df["from_stop_id"] = df["from_stop_id"].astype(str)
    df["stop_id"] = df["stop_id"].astype(str)
    df["cr_day"] = (df["vp_ts"] - pd.Timedelta(hours=SERVICE_DAY_OFFSET_H)).dt.date

    counts = df.groupby("cr_day").size()
    usable = sorted(counts[counts >= MIN_SEGS_PER_DAY].index)
    print(f"usable CR-days (>={MIN_SEGS_PER_DAY} seg): {len(usable)}  ({usable[0]} .. {usable[-1]})")

    test_days = usable[-N_TEST_DAYS:]
    val_days = usable[-(N_TEST_DAYS + N_VAL_DAYS):-N_TEST_DAYS]
    train_days = usable[:-(N_TEST_DAYS + N_VAL_DAYS)]

    train_df = df[df["cr_day"].isin(train_days)].copy()
    val_df = df[df["cr_day"].isin(val_days)].copy()
    test_df = df[df["cr_day"].isin(test_days)].copy()
    print(f"train {len(train_df):,} ({len(train_days)}d)  "
          f"val {len(val_df):,} ({len(val_days)}d)  test {len(test_df):,} ({len(test_days)}d)")
    print(f"test days: {test_days}")

    # ---- schedule baseline on the test set (exact, per real trip_id) ----
    snap = latest_snapshot_on_or_before("mbta", max(test_days))
    print(f"\nschedule baseline uses snapshot {snap}")
    sched = scheduled_segment_durations(snap)
    test_df["sched_dur"] = [
        sched.get((t, f, s))
        for t, f, s in zip(test_df["trip_id"], test_df["from_stop_id"], test_df["stop_id"])
    ]
    cov = test_df["sched_dur"].notna().mean()
    base_err = (test_df["sched_dur"] - test_df["time_to_arrival_seconds"]).to_numpy(dtype=float)
    b = dist_stats(base_err)
    print(f"BASELINE(schedule)  coverage={cov:.0%}  n={b['n']}  "
          f"MAE {b['mae']:.1f}s  RMSE {float(np.sqrt(np.nanmean(base_err**2))):.1f}s  "
          f"within +-30s {b['w30']:.0f}%  +-60s {b['w60']:.0f}%")

    # ---- model leaderboard (day-grouped split, no registry writes) ----
    pre = (train_df, val_df, test_df)
    cp = lambda: tuple(x.copy() for x in pre)  # noqa: E731
    trainers = {
        "historical_mean": lambda: train_historical_mean(dataset_name=DATASET, save_model=False, pre_split=cp()),
        "polyreg_distance": lambda: train_polyreg_distance(dataset_name=DATASET, degree=2, save_model=False, pre_split=cp()),
        "polyreg_time": lambda: train_polyreg_time(dataset_name=DATASET, poly_degree=2, include_temporal=True, save_model=False, pre_split=cp()),
        "ewma": lambda: train_ewma(dataset_name=DATASET, alpha=0.3, save_model=False, pre_split=cp()),  # ablation only (retired)
        "xgboost": lambda: train_xgboost(dataset_name=DATASET, route_id=None, save_model=False, pre_split=cp()),
    }
    print(f"\n{'model':18s} {'test MAE':>9s} {'RMSE':>8s} {'R2':>7s}   vs schedule")
    print("-" * 62)
    print(f"{'BASELINE(schedule)':18s} {b['mae']:8.1f}s {float(np.sqrt(np.nanmean(base_err**2))):7.1f}s {'--':>7s}")
    xgb_model = None
    for name, fn in trainers.items():
        try:
            out = fn()
            m = out.get("metrics", {})
            mae = m.get("test_mae_seconds") or (m.get("test_mae_minutes", 0) * 60)
            rmse = m.get("test_rmse_seconds")
            r2 = m.get("test_r2")
            if name == "xgboost":
                xgb_model = out.get("model")
            delta = (b["mae"] - mae) / b["mae"] * 100 if b["mae"] else 0
            flag = "BEATS" if mae < b["mae"] else "worse"
            print(f"{name:18s} {mae:8.1f}s {rmse:7.1f}s {r2:7.3f}   {flag} ({delta:+.0f}%)")
        except Exception as e:
            print(f"{name:18s} FAILED: {e}")

    # ---- error distribution: xgboost vs schedule on the test set ----
    xgb_stats = None
    if xgb_model is not None:
        try:
            preds = np.asarray(xgb_model.predict(test_df.copy()), dtype=float)
            xgb_err = preds - test_df["time_to_arrival_seconds"].to_numpy(dtype=float)
            xgb_stats = dist_stats(xgb_err)
            signed = xgb_err[~np.isnan(xgb_err)]
            print(f"\nxgboost signed-error: mean={signed.mean():+.1f}s  median={np.median(signed):+.1f}s")
        except Exception as e:
            print(f"\nxgboost prediction for distribution FAILED: {e}")

    print("\n=== per-segment absolute-error distribution (test) ===")
    hdr = f"{'stat':10s} {'xgboost':>10s} {'schedule':>10s}" if xgb_stats else f"{'stat':10s} {'schedule':>10s}"
    print(hdr)
    for k in ("mae", "median", "p90", "p95", "p99", "max", "w30", "w60"):
        unit = "%" if k in ("w30", "w60") else "s"
        if xgb_stats:
            print(f"{k:10s} {xgb_stats[k]:9.1f}{unit} {b[k]:9.1f}{unit}")
        else:
            print(f"{k:10s} {b[k]:9.1f}{unit}")


if __name__ == "__main__":
    main()
