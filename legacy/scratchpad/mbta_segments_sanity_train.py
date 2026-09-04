"""Leakage/sanity check: train XGBoost on the MBTA segment dataset.

Adapted from scratchpad/baseline_compare_corpus.py's day-grouped split
approach, but for the small ~2-day MBTA window this repo just built
(datasets/mbta_segments_20260817_20260819.parquet) a day-grouped walk-forward
split isn't practical (only 2 days) -- use a random 70/15/15 split instead,
grouped by trip_id so no trip straddles splits (mirrors the trip-safety
property day-grouping gives the bUCR corpus, just at trip granularity since
there's no multi-day service pattern here).

HARD CONSTRAINT: save_model=False always -- nothing written to
models/trained/ or registry.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path.cwd()))
sys.path.append(str(Path("models").resolve()))
from xgb.train import train_xgboost  # noqa: E402

DATASET = "mbta_segments_20260817_20260819"
DS_PATH = f"datasets/{DATASET}.parquet"
SEED = 42


def main() -> None:
    df = pd.read_parquet(DS_PATH)
    print(f"loaded {len(df):,} segments, {df['trip_id'].nunique()} trips")

    trip_ids = df["trip_id"].unique()
    rng = np.random.default_rng(SEED)
    rng.shuffle(trip_ids)
    n = len(trip_ids)
    n_test = max(1, int(n * 0.15))
    n_val = max(1, int(n * 0.15))
    test_trips = set(trip_ids[:n_test])
    val_trips = set(trip_ids[n_test:n_test + n_val])
    train_trips = set(trip_ids[n_test + n_val:])

    train_df = df[df["trip_id"].isin(train_trips)].copy()
    val_df = df[df["trip_id"].isin(val_trips)].copy()
    test_df = df[df["trip_id"].isin(test_trips)].copy()
    print(f"train {len(train_df):,} ({len(train_trips)} trips)  "
          f"val {len(val_df):,} ({len(val_trips)} trips)  "
          f"test {len(test_df):,} ({len(test_trips)} trips)")

    result = train_xgboost(
        dataset_name=DATASET,
        route_id=None,
        save_model=False,
        pre_split=(train_df, val_df, test_df),
    )
    metrics = result.get("metrics", {})
    print("\n=== MBTA segment XGBoost sanity result ===")
    for k in ("test_mae_seconds", "test_rmse_seconds", "test_r2", "val_mae_seconds", "val_r2"):
        if k in metrics:
            print(f"  {k}: {metrics[k]}")


if __name__ == "__main__":
    main()
