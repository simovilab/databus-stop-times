"""Full error-distribution report for the best models vs the offset schedule.

Reports, at the per-segment level and the summed-segment stop-level (h=1..8),
the distribution of absolute error and signed error: min, max, mean, std,
median, key percentiles, skew/kurtosis, and within-band hit rates. Same
day-grouped holdout as the other corpus scripts (train all but last 5 CR-days,
test the last 5).
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


def _skew_kurt(x):
    x = x[~np.isnan(x)]
    m, s = x.mean(), x.std()
    if s == 0:
        return 0.0, 0.0
    z = (x - m) / s
    return float((z ** 3).mean()), float((z ** 4).mean() - 3.0)


def describe(name, err):
    """err = signed error (pred - obs), seconds."""
    err = err[~np.isnan(err)]
    ae = np.abs(err)
    sk, ku = _skew_kurt(err)
    p = lambda a, q: float(np.percentile(a, q))
    print(f"\n--- {name}  (n={len(err):,}) ---")
    print(f"  SIGNED (pred-obs): mean {err.mean():+.1f}s  std {err.std():.1f}s  "
          f"skew {sk:+.2f}  exkurt {ku:+.2f}")
    print(f"    min {err.min():+.0f}  p05 {p(err,5):+.0f}  p25 {p(err,25):+.0f}  "
          f"med {p(err,50):+.0f}  p75 {p(err,75):+.0f}  p95 {p(err,95):+.0f}  max {err.max():+.0f}")
    print(f"  ABS |pred-obs|: MAE {ae.mean():.1f}s  std {ae.std():.1f}s")
    print(f"    p50 {p(ae,50):.0f}  p75 {p(ae,75):.0f}  p90 {p(ae,90):.0f}  "
          f"p95 {p(ae,95):.0f}  p99 {p(ae,99):.0f}  max {ae.max():.0f}")
    print(f"    within +-30s {(ae<=30).mean()*100:4.1f}%   +-60s {(ae<=60).mean()*100:4.1f}%"
          f"   +-120s {(ae<=120).mean()*100:4.1f}%")
    edges = [0, 15, 30, 60, 90, 120, 180, 300, np.inf]
    labels = ["0-15", "15-30", "30-60", "60-90", "90-120", "120-180", "180-300", "300+"]
    hist, _ = np.histogram(ae, bins=edges)
    print("    |err| hist: " + "  ".join(f"{l}:{h/len(ae)*100:.0f}%" for l, h in zip(labels, hist)))


def horizon_windows(test_df, value_cols):
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

    xgb = XGBTimeModel(); xgb.fit(train_df)
    pdist = PolyRegDistanceModel(degree=2); pdist.fit(train_df)
    test_df = test_df.copy()
    test_df["pred_xgb"] = xgb.predict(test_df)
    test_df["pred_pdist"] = pdist.predict(test_df)

    sched = scheduled_segment_durations("scratchpad/gtfs.zip")
    def shape_of(tid):
        p = _parse_trip_id(tid); return p[3] if p else None
    test_df["shape_id"] = test_df["trip_id"].map(shape_of)
    test_df["pred_sched"] = [sched.get((sh, fr, to)) for sh, fr, to in
                             zip(test_df["shape_id"], test_df["from_stop_id"], test_df["stop_id"])]

    obs = test_df["time_to_arrival_seconds"].to_numpy(dtype=float)
    print("\n" + "#" * 72)
    print("# PER-SEGMENT error distribution (single stop-to-stop leg)")
    print("#" * 72)
    print(f"observed target: mean {obs.mean():.0f}s  med {np.median(obs):.0f}s  "
          f"min {obs.min():.0f}  p95 {np.percentile(obs,95):.0f}  max {obs.max():.0f}")
    describe("XGBoost", test_df["pred_xgb"].to_numpy(dtype=float) - obs)
    describe("PolyReg-distance", test_df["pred_pdist"].to_numpy(dtype=float) - obs)
    describe("Offset schedule", test_df["pred_sched"].to_numpy(dtype=float) - obs)

    cols = ["time_to_arrival_seconds", "pred_xgb", "pred_pdist", "pred_sched"]
    win = horizon_windows(test_df, cols)
    o = win["time_to_arrival_seconds"].to_numpy()
    print("\n" + "#" * 72)
    print("# STOP-LEVEL error distribution (summed segments, ALL horizons pooled)")
    print("#" * 72)
    print(f"observed ETA: mean {o.mean():.0f}s  med {np.median(o):.0f}s  "
          f"min {o.min():.0f}  p95 {np.percentile(o,95):.0f}  max {o.max():.0f}")
    describe("XGBoost (stop-level)", win["pred_xgb"].to_numpy() - o)
    describe("PolyReg-distance (stop-level)", win["pred_pdist"].to_numpy() - o)
    describe("Offset schedule (stop-level)", win["pred_sched"].to_numpy() - o)

    print("\n" + "#" * 72)
    print("# XGBoost absolute-error percentiles by horizon (stop-level)")
    print("#" * 72)
    print(f"{'h':>2s} {'n':>6s} {'MAE':>6s} {'p50':>5s} {'p75':>5s} {'p90':>5s} {'p95':>5s} {'max':>6s}")
    for h in range(1, MAX_H + 1):
        w = win[win["horizon"] == h]
        if w.empty:
            continue
        ae = np.abs((w["pred_xgb"] - w["time_to_arrival_seconds"]).to_numpy())
        pc = lambda q: np.percentile(ae, q)
        print(f"{h:2d} {len(w):6d} {ae.mean():5.0f}s {pc(50):4.0f}s {pc(75):4.0f}s "
              f"{pc(90):4.0f}s {pc(95):4.0f}s {ae.max():5.0f}s")


if __name__ == "__main__":
    main()
