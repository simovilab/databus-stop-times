"""Baseline comparison on the full corpus: models vs offset schedule.

Day-grouped walk-forward holdout: sort the usable CR-days (>=50 segments),
train = all but last 10, val = next-to-last 5 days, test = last 5 days. A trip
lives within one day, so day-grouping is trip-safe (no trip straddles splits).
Baseline = schedule's own stop-to-stop durations (offset-invariant).
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))
sys.path.append(str(Path("models").resolve()))
from historical_mean.train import train_historical_mean
from polyreg_distance.train import train_polyreg_distance
from polyreg_time.train import train_polyreg_time
from ewma.train import train_ewma
from xgb.train import train_xgboost

from feature_engineering.bucr_gtfs import load_gtfs, route_directions, route_direction_candidates
from feature_engineering.segment_dataset_builder import _parse_trip_id

DATASET = "bucr_segments_corpus"
DS_PATH = "datasets/bucr_segments_corpus.parquet"
MIN_SEGS_PER_DAY = 50
N_TEST_DAYS = 5
N_VAL_DAYS = 5


def _arr_seconds(hhmmss):
    h, m, s = str(hhmmss).split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def scheduled_segment_durations(gtfs_path):
    g = load_gtfs(gtfs_path)
    lookup = {}
    for route_id, direction_id in route_directions(g):
        for cand in route_direction_candidates(g, route_id, direction_id):
            st = (g.stop_times[g.stop_times["trip_id"] == cand.representative_trip_id]
                  .sort_values("stop_sequence").reset_index(drop=True))
            secs = st["arrival_time"].map(_arr_seconds).to_numpy()
            ids = st["stop_id"].astype(str).to_numpy()
            for k in range(len(st) - 1):
                dur = float(secs[k + 1] - secs[k])
                if dur > 0:
                    lookup[(cand.shape_id, ids[k], ids[k + 1])] = dur
    return lookup


def mae_rmse(err):
    err = err[~np.isnan(err)]
    return float(np.abs(err).mean()), float(np.sqrt((err ** 2).mean())), len(err)


def main():
    df = pd.read_parquet(DS_PATH)
    df["cr_day"] = (df["vp_ts"] - pd.Timedelta(hours=6)).dt.date
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

    # schedule baseline on test
    sched = scheduled_segment_durations("scratchpad/gtfs.zip")
    def shape_of(tid):
        p = _parse_trip_id(tid); return p[3] if p else None
    test_df["shape_id"] = test_df["trip_id"].map(shape_of)
    test_df["sched_dur"] = [sched.get((sh, fr, to)) for sh, fr, to in
                            zip(test_df["shape_id"], test_df["from_stop_id"], test_df["stop_id"])]
    base_err = (test_df["sched_dur"] - test_df["time_to_arrival_seconds"]).to_numpy(dtype=float)
    b_mae, b_rmse, b_n = mae_rmse(base_err)
    within60 = np.nanmean(np.abs(base_err) <= 60) * 100
    print(f"\nBASELINE (offset schedule)  coverage={test_df['sched_dur'].notna().mean():.0%}  n={b_n}")
    print(f"  MAE {b_mae:.1f}s ({b_mae/60:.2f}m)  RMSE {b_rmse:.1f}s  within +-60s {within60:.0f}%")

    pre = (train_df, val_df, test_df)
    cp = lambda: tuple(x.copy() for x in pre)
    trainers = {
        "historical_mean": lambda: train_historical_mean(dataset_name=DATASET, save_model=True, pre_split=cp()),
        "polyreg_distance": lambda: train_polyreg_distance(dataset_name=DATASET, degree=2, save_model=True, pre_split=cp()),
        "polyreg_time": lambda: train_polyreg_time(dataset_name=DATASET, poly_degree=2, include_temporal=True, save_model=True, pre_split=cp()),
        "ewma": lambda: train_ewma(dataset_name=DATASET, alpha=0.3, save_model=True, pre_split=cp()),
        "xgboost": lambda: train_xgboost(dataset_name=DATASET, route_id=None, save_model=True, pre_split=cp()),
    }
    print(f"\n{'model':18s} {'test MAE':>9s} {'RMSE':>8s} {'R2':>7s}   vs baseline")
    print("-" * 62)
    print(f"{'BASELINE(schedule)':18s} {b_mae:8.1f}s {b_rmse:7.1f}s {'--':>7s}")
    for name, fn in trainers.items():
        try:
            m = fn().get("metrics", {})
            mae = m.get("test_mae_seconds") or (m.get("test_mae_minutes", 0) * 60)
            rmse = m.get("test_rmse_seconds"); r2 = m.get("test_r2")
            delta = (b_mae - mae) / b_mae * 100 if b_mae else 0
            flag = "BEATS" if mae < b_mae else "worse"
            print(f"{name:18s} {mae:8.1f}s {rmse:7.1f}s {r2:7.3f}   {flag} ({delta:+.0f}%)")
        except Exception as e:
            print(f"{name:18s} FAILED: {e}")


if __name__ == "__main__":
    main()
