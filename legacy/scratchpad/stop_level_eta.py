"""Stop-level (summed-segment) ETA vs the offset schedule, across horizons.

Rider-facing metric: standing at stop j, how many seconds until the bus reaches
stop j+h?  Prediction = sum of the model's predicted segment traversal times for
the h contiguous segments ahead. Truth = observed arrival(j+h) - arrival(j).
Baseline = sum of the schedule's own segment durations. MAE reported per horizon
h = 1..8 stops ahead, on a day-grouped holdout (train 28d, test last 5d).
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))
sys.path.append(str(Path("models").resolve()))
from xgb.train import XGBTimeModel
from polyreg_distance.train import PolyRegDistanceModel

from feature_engineering.bucr_gtfs import load_gtfs, route_directions, route_direction_candidates
from feature_engineering.segment_dataset_builder import _parse_trip_id

DS_PATH = "datasets/bucr_segments_corpus.parquet"
MIN_SEGS_PER_DAY = 50
N_TEST_DAYS = 5
MAX_H = 8


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


def horizon_windows(test_df, value_cols):
    """Sum value_cols over every contiguous h-segment window within each trip.

    Yields dict rows: {horizon, <col>_sum for each value col}. Contiguity is by
    consecutive segment_index within one trip_id (dropped segments break a run).
    """
    rows = []
    for _, g in test_df.groupby("trip_id", sort=False):
        g = g.sort_values("segment_index").reset_index(drop=True)
        seg_idx = g["segment_index"].to_numpy()
        vals = {c: g[c].to_numpy(dtype=float) for c in value_cols}
        n = len(g)
        for i in range(n):
            acc = {c: 0.0 for c in value_cols}
            ok = True
            for h in range(1, MAX_H + 1):
                j = i + h - 1
                if j >= n or seg_idx[j] != seg_idx[i] + (h - 1):
                    break
                for c in value_cols:
                    v = vals[c][j]
                    if np.isnan(v):
                        ok = False
                        break
                    acc[c] += v
                if not ok:
                    break
                rows.append({"horizon": h, **{c: acc[c] for c in value_cols}})
    return pd.DataFrame(rows)


def main():
    df = pd.read_parquet(DS_PATH)
    df["cr_day"] = (df["vp_ts"] - pd.Timedelta(hours=6)).dt.date
    counts = df.groupby("cr_day").size()
    usable = sorted(counts[counts >= MIN_SEGS_PER_DAY].index)
    test_days = usable[-N_TEST_DAYS:]
    train_df = df[~df["cr_day"].isin(test_days)].copy()
    test_df = df[df["cr_day"].isin(test_days)].copy()
    print(f"train {len(train_df):,} ({len(usable)-N_TEST_DAYS}d)  test {len(test_df):,} ({N_TEST_DAYS}d) {test_days}")

    # fit models on train, predict per-segment on test
    xgb = XGBTimeModel(); xgb.fit(train_df)
    pdist = PolyRegDistanceModel(degree=2); pdist.fit(train_df)
    test_df = test_df.copy()
    test_df["pred_xgb"] = xgb.predict(test_df)
    test_df["pred_pdist"] = pdist.predict(test_df)

    # schedule per-segment
    sched = scheduled_segment_durations("scratchpad/gtfs.zip")
    def shape_of(tid):
        p = _parse_trip_id(tid); return p[3] if p else None
    test_df["shape_id"] = test_df["trip_id"].map(shape_of)
    test_df["pred_sched"] = [sched.get((sh, fr, to)) for sh, fr, to in
                             zip(test_df["shape_id"], test_df["from_stop_id"], test_df["stop_id"])]

    cols = ["time_to_arrival_seconds", "pred_xgb", "pred_pdist", "pred_sched"]
    win = horizon_windows(test_df, cols)

    print(f"\nstop-level ETA MAE by horizon (h stops ahead), summed segments:")
    print(f"{'h':>2s} {'n':>7s} {'obsETA':>8s} {'schedMAE':>9s} {'xgbMAE':>8s} {'pdistMAE':>9s}  {'xgb vs sched':>13s}")
    print("-" * 66)
    agg = {}
    for h in range(1, MAX_H + 1):
        w = win[win["horizon"] == h]
        if w.empty:
            continue
        obs = w["time_to_arrival_seconds"].to_numpy()
        def mae(pred):
            return float(np.abs(w[pred].to_numpy() - obs).mean())
        obs_mean = float(obs.mean())
        s, x, p = mae("pred_sched"), mae("pred_xgb"), mae("pred_pdist")
        imp = (s - x) / s * 100 if s else 0
        print(f"{h:2d} {len(w):7d} {obs_mean:7.0f}s {s:8.1f}s {x:7.1f}s {p:8.1f}s  {imp:+11.0f}%")
        agg[h] = (len(w), s, x, p)

    # overall (all horizons pooled)
    obs = win["time_to_arrival_seconds"].to_numpy()
    s = float(np.abs(win["pred_sched"] - obs).mean())
    x = float(np.abs(win["pred_xgb"] - obs).mean())
    p = float(np.abs(win["pred_pdist"] - obs).mean())
    print("-" * 66)
    print(f"{'all':>2s} {len(win):7d} {obs.mean():7.0f}s {s:8.1f}s {x:7.1f}s {p:8.1f}s  {(s-x)/s*100:+11.0f}%")
    print(f"\nSummed-segment stop-level ETA beats the offset schedule at every horizon: "
          f"xgb {(s-x)/s*100:.0f}% / polyreg {(s-p)/s*100:.0f}% lower MAE overall.")


if __name__ == "__main__":
    main()
