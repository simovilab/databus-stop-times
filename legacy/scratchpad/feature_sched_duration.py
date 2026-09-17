"""Does scheduled segment duration break the MBTA plateau? (roadmap 5.x feature work)

MBTA XGBoost plateaus at ~24.6s / R2 0.65 across every hyperparameter config,
which says the remaining error is feature-limited. The model currently uses only
geometry + kinematics + time-of-day; it has no per-segment identity and no prior
on how long *this* segment should take. Scheduled segment duration
(sched_arrival(to) - sched_arrival(from), read from the GTFS snapshot) is that
prior -- and it is serveable online (GTFS is present in prod, per-trip, no
cross-vehicle state), so it passes the train/serve-parity filter that killed
headway.

Design: train the tuned MBTA XGBoost config (+ early stopping) twice on the
**identical row set** and the **same day-grouped split** -- base features vs
base + sched_dur. Uncovered rows keep sched_dur = NaN (XGBoost routes NaN
natively), so both models see exactly the same rows and the only difference is
the feature. Test is evaluated once.

save_model=False throughout; nothing touches the registry.

Run (from repo root)::

    PYTHONPATH=.:gtfs-rt-pipeline uv run --group bucr --with duckdb \
        python scratchpad/feature_sched_duration.py
"""
from __future__ import annotations

import datetime as dt
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "gtfs-rt-pipeline"))

from feature_engineering.bucr_gtfs import load_gtfs  # noqa: E402
from rt_pipeline.storage.static_gtfs import (  # noqa: E402
    get_snapshot,
    latest_snapshot_on_or_before,
)

DATASET = "mbta_segments_corpus"
DS_PATH = str(REPO_ROOT / "datasets" / f"{DATASET}.parquet")
MIN_SEGS_PER_DAY = 200
N_TEST_DAYS = 3
N_VAL_DAYS = 3
SERVICE_DAY_OFFSET_H = 7  # 3am ET in EDT -> 07:00 UTC
TARGET = "time_to_arrival_seconds"

# Feature list exactly as XGBTimeModel._get_feature_groups() would assemble it
# (core -> spatial_numeric -> temporal), so the base model matches production.
BASE_FEATURES = [
    "distance_to_stop",
    "distance_to_next_stop", "shape_distance_to_stop", "shape_progress",
    "cross_track_error", "progress_ratio", "stops_ahead",
    "current_speed_kmh", "bearing_to_stop", "bearing_diff", "is_at_stop",
    "hour", "day_of_week", "is_weekend", "is_peak_hour",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]

# Tuned MBTA config from the 9-param + early-stopping HPO study (RESULTS s06).
TUNED = dict(
    max_depth=12, learning_rate=0.0062, subsample=0.82, colsample_bytree=1.0,
    min_child_weight=1.5, reg_alpha=0.02, reg_lambda=10.9, gamma=1.8,
    n_estimators=3000, objective="reg:squarederror", random_state=42, n_jobs=4,
    early_stopping_rounds=50, eval_metric="mae",
)


def _arr_seconds(hhmmss: str) -> float:
    try:
        h, m, s = str(hhmmss).split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)
    except Exception:
        return float("nan")


def _sched_lookup_for_snapshot(snapshot_date: dt.date) -> dict:
    """(trip_id, from_stop_id, to_stop_id) -> scheduled adjacent-segment seconds."""
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


def attach_sched_dur(df: pd.DataFrame) -> pd.Series:
    """Per service-day, resolve the snapshot on-or-before that day and map each
    row's (trip, from, to) to its scheduled duration. Serving-realistic: at
    inference the current snapshot's stop_times are available for the live trip."""
    snap_cache: dict = {}
    out = np.full(len(df), np.nan, dtype=float)
    for cr_day, idx in df.groupby("cr_day").groups.items():
        snap = latest_snapshot_on_or_before("mbta", cr_day)
        if snap not in snap_cache:
            snap_cache[snap] = _sched_lookup_for_snapshot(snap)
        lut = snap_cache[snap]
        sub = df.loc[idx]
        vals = [
            lut.get((t, f, s), np.nan)
            for t, f, s in zip(sub["trip_id"], sub["from_stop_id"], sub["stop_id"])
        ]
        out[df.index.get_indexer(idx)] = vals
    return pd.Series(out, index=df.index, name="sched_dur")


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    a = np.abs(err)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mae": float(a.mean()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": 1.0 - ss_res / ss_tot,
        "w60": float((a <= 60).mean() * 100),
    }


def fit_eval(feats, tr, va, te):
    Xtr, ytr = tr[feats].to_numpy(dtype=float), tr[TARGET].to_numpy(dtype=float)
    Xva, yva = va[feats].to_numpy(dtype=float), va[TARGET].to_numpy(dtype=float)
    Xte, yte = te[feats].to_numpy(dtype=float), te[TARGET].to_numpy(dtype=float)
    model = XGBRegressor(**TUNED)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    imp = dict(zip(feats, model.feature_importances_))
    return metrics(yte, model.predict(Xte)), model.best_iteration, imp


def main() -> None:
    df = pd.read_parquet(DS_PATH)
    for c in ("trip_id", "from_stop_id", "stop_id"):
        df[c] = df[c].astype(str)
    df["cr_day"] = (df["vp_ts"] - pd.Timedelta(hours=SERVICE_DAY_OFFSET_H)).dt.date

    counts = df.groupby("cr_day").size()
    usable = sorted(counts[counts >= MIN_SEGS_PER_DAY].index)
    test_days = usable[-N_TEST_DAYS:]
    val_days = usable[-(N_TEST_DAYS + N_VAL_DAYS):-N_TEST_DAYS]
    train_days = usable[:-(N_TEST_DAYS + N_VAL_DAYS)]

    # Same row set for both models: drop only on base features + target.
    df = df.dropna(subset=BASE_FEATURES + [TARGET]).copy()
    df["sched_dur"] = attach_sched_dur(df)

    tr = df[df["cr_day"].isin(train_days)].copy()
    va = df[df["cr_day"].isin(val_days)].copy()
    te = df[df["cr_day"].isin(test_days)].copy()
    print(f"rows: train {len(tr):,} / val {len(va):,} / test {len(te):,}")
    for name, part in (("train", tr), ("val", va), ("test", te)):
        cov = part["sched_dur"].notna().mean()
        print(f"  sched_dur coverage {name}: {cov:.0%}")

    base_m, base_bi, _ = fit_eval(BASE_FEATURES, tr, va, te)
    aug_feats = BASE_FEATURES + ["sched_dur"]
    aug_m, aug_bi, aug_imp = fit_eval(aug_feats, tr, va, te)

    print(f"\n{'model':22s} {'test MAE':>9s} {'RMSE':>8s} {'R2':>7s} {'+-60s':>7s} {'trees':>6s}")
    print("-" * 64)
    print(f"{'base (19 feats)':22s} {base_m['mae']:8.2f}s {base_m['rmse']:7.1f}s "
          f"{base_m['r2']:7.3f} {base_m['w60']:6.1f}% {base_bi:6d}")
    print(f"{'base + sched_dur':22s} {aug_m['mae']:8.2f}s {aug_m['rmse']:7.1f}s "
          f"{aug_m['r2']:7.3f} {aug_m['w60']:6.1f}% {aug_bi:6d}")
    d_mae = (base_m["mae"] - aug_m["mae"]) / base_m["mae"] * 100
    print(f"\nsched_dur: test MAE {d_mae:+.1f}%   R2 {aug_m['r2'] - base_m['r2']:+.3f}")
    top = sorted(aug_imp.items(), key=lambda x: x[1], reverse=True)[:5]
    print("top-5 features (augmented, by importance):")
    for f, v in top:
        print(f"  {f:24s} {v:.3f}")


if __name__ == "__main__":
    main()
