"""Per-route vs global XGBoost comparison (roadmap open item, never done).

``train_all_models.py --by-route`` has supported per-route training since
it was written, but nothing in this repo has ever actually compared a
per-route XGBoost against the global one on the **segment** corpus (the
current primary formulation) with a significance test. This script does
that once, cheaply, for MBTA (routes 222/15/37) and checks whether bUCR
has more than one route before attempting the same there.

Design, mirroring the roadmap's ask:

  1. Day-grouped split (``models.common.splits.day_grouped_split``) on the
     full corpus -- this is the SAME split every route's rows are drawn
     from, so "same test rows" is literal: the pooled per-route prediction
     vector and the global prediction vector are both aligned against the
     one global test_df (each per-route model only fills in the rows for
     its own route).
  2. One global XGBoost (all routes) and one XGBoost per route, all using
     the roadmap-5.1-tuned MBTA hyperparameters (the best-val-MAE row of
     scratchpad/hpo_trials_mbta_segments_corpus.csv), trained via
     ``models.xgb.train.train_xgboost(..., save_model=False, pre_split=...)``
     -- nothing is written to the registry.
  3. Score both on the same test rows, per-route and pooled, with
     ``models.evaluation.stats.compare_all``/``paired_test`` (groups=trip_id)
     for a CI + p-value on the pooled global-vs-per-route claim.
  4. Report rows-per-route alongside the deltas, since a per-route model
     trained on ~1/3 the data can lose purely for lack of data -- this
     output is what lets a reader tell "route heterogeneity" from "sample
     size" apart.

A caveat surfaced while building this: ``day_grouped_split(...,
group_cols=("trip_id",))`` with its default ``purge=True`` drops the
overwhelming majority of train/val rows here (18,651 train / 189 val vs.
92,618 / 32,534 with purge=False) because the same ``trip_id`` values
recur across many service days (they're trip *templates*, not per-day
instances) -- trip-purging under that assumption treats nearly the whole
corpus as one giant "group". The instructed/expected split sizes
(92,618 / 32,534 / 28,346) only reproduce with ``purge=False``, matching
what the original ``scratchpad/hpo_xgboost.py::day_grouped_split`` (which
never purged) actually did. This script uses ``purge=False`` to match the
tuned hyperparameters' training regime and the requested split sizes, but
this is a real leakage caveat worth flagging, not a settled decision: a
trip-safe split (dedupe by e.g. (trip_id, service_day) or truly unique
trip instances) is future work, not this script's job to fix.

Run (repo root)::

    PYTHONPATH=.:models /path/to/.venv-dl/bin/python scratchpad/by_route_comparison.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "models"))

import numpy as np
import pandas as pd

from models.common.splits import day_grouped_split
from models.evaluation.stats import compare_all, paired_test
from models.xgb.train import train_xgboost

TARGET_COL = "time_to_arrival_seconds"

# Roadmap-5.1-tuned MBTA config: best-val-MAE row (trial 87, val MAE
# 24.26s) of scratchpad/hpo_trials_mbta_segments_corpus.csv.
MBTA_TUNED_PARAMS = dict(
    max_depth=12,
    learning_rate=0.006246,
    subsample=0.818663,
    colsample_bytree=0.999495,
    min_child_weight=1.511085,
    reg_alpha=0.023292,
    reg_lambda=10.873937,
    gamma=1.825873,
    n_estimators=3000,           # cap; early stopping picks the real count
    early_stopping_rounds=50,
)

# bUCR-tuned config (from the prompt); used only if bUCR turns out to have
# more than one route (see main()).
BUCR_TUNED_PARAMS = dict(
    max_depth=11,
    learning_rate=0.012,
    reg_lambda=69.0,
    n_estimators=3000,
    early_stopping_rounds=50,
)


def _filter_split(pre_split, route_id: str):
    return tuple(df[df["route_id"] == route_id].copy() for df in pre_split)


def _fit_global(dataset_name: str, pre_split, params: dict):
    out = train_xgboost(
        dataset_name=dataset_name, save_model=False, pre_split=pre_split, **params
    )
    return out["model"], out["metrics"]


def _fit_per_route(dataset_name: str, pre_split, routes, params: dict):
    """One model per route. Returns {route: (model, metrics, row_counts)}."""
    models = {}
    for route in routes:
        route_split = _filter_split(pre_split, route)
        row_counts = tuple(len(df) for df in route_split)
        out = train_xgboost(
            dataset_name=dataset_name,
            route_id=route,
            save_model=False,
            pre_split=route_split,
            **params,
        )
        models[route] = (out["model"], out["metrics"], row_counts)
    return models


def _pooled_per_route_predictions(test_df: pd.DataFrame, per_route_models: dict) -> np.ndarray:
    """Fill one prediction per test row using the model trained for that
    row's route. Same test rows as the global model -- just each row's
    prediction comes from its own route's specialist."""
    preds = np.full(len(test_df), np.nan, dtype=float)
    for route, (model, _, _) in per_route_models.items():
        mask = (test_df["route_id"] == route).to_numpy()
        if mask.sum() == 0:
            continue
        preds[mask] = model.predict(test_df.loc[mask])
    if np.isnan(preds).any():
        missing_routes = sorted(test_df.loc[np.isnan(preds), "route_id"].unique())
        raise ValueError(
            f"No per-route model covers test rows for route(s): {missing_routes}"
        )
    return preds


def run_comparison(dataset_name: str, offset_hours: int, min_rows_per_day: int,
                    test_days: int, val_days: int, params: dict) -> dict:
    df = pd.read_parquet(REPO_ROOT / "datasets" / f"{dataset_name}.parquet")
    routes = sorted(df["route_id"].unique())

    if len(routes) <= 1:
        return {"dataset": dataset_name, "skipped": True, "routes": routes}

    split = day_grouped_split(
        df, ts_col="vp_ts", offset_hours=offset_hours,
        test_days=test_days, val_days=val_days,
        min_rows_per_day=min_rows_per_day, group_cols=("trip_id",), purge=False,
    )
    train_df, val_df, test_df = split.train, split.val, split.test
    pre_split = (train_df, val_df, test_df)
    print(f"\n=== {dataset_name}: routes {routes} ===")
    print(f"split: train {len(train_df):,} / val {len(val_df):,} / test {len(test_df):,}")

    rows_per_route = {
        "train": train_df["route_id"].value_counts().to_dict(),
        "val": val_df["route_id"].value_counts().to_dict(),
        "test": test_df["route_id"].value_counts().to_dict(),
    }
    print("rows per route:", rows_per_route)

    global_model, global_metrics = _fit_global(dataset_name, pre_split, params)
    print(f"global: val MAE {global_metrics.get('val_mae_seconds'):.2f}s | "
          f"test MAE {global_metrics.get('test_mae_seconds'):.2f}s")

    per_route_models = _fit_per_route(dataset_name, pre_split, routes, params)
    for route, (_, metrics, counts) in per_route_models.items():
        print(f"route {route} (train/val/test rows {counts}): "
              f"val MAE {metrics.get('val_mae_seconds'):.2f}s | "
              f"test MAE {metrics.get('test_mae_seconds'):.2f}s")

    y_true = test_df[TARGET_COL].to_numpy(dtype=float)
    pred_global = global_model.predict(test_df)
    pred_per_route_pooled = _pooled_per_route_predictions(test_df, per_route_models)
    groups = test_df["trip_id"].to_numpy()

    pooled_table = compare_all(
        y_true,
        {"global": pred_global, "per_route": pred_per_route_pooled},
        baseline="global",
        groups=groups,
        paired_method="bootstrap",
    )
    print("\nPooled comparison (baseline=global):")
    print(pooled_table.to_string(index=False))

    per_route_breakdown = {}
    for route in routes:
        mask = (test_df["route_id"] == route).to_numpy()
        if mask.sum() == 0:
            continue
        test_r = paired_test(
            y_true[mask], pred_per_route_pooled[mask], pred_global[mask],
            method="bootstrap", groups=groups[mask],
        )
        per_route_breakdown[route] = {
            "n_test_rows": int(mask.sum()),
            "n_train_rows": rows_per_route["train"].get(route, 0),
            "mae_per_route_model": test_r.mae_a,
            "mae_global_model": test_r.mae_b,
            "diff_seconds": test_r.diff_seconds,
            "diff_percent": test_r.diff_percent,
            "p_value": test_r.p_value,
        }
        print(f"\nroute {route} (n_test={mask.sum()}, n_train={rows_per_route['train'].get(route, 0)}): "
              f"per-route MAE {test_r.mae_a:.2f}s vs global MAE {test_r.mae_b:.2f}s "
              f"(diff {test_r.diff_seconds:+.2f}s / {test_r.diff_percent:+.1f}%, p={test_r.p_value:.4f})")

    return {
        "dataset": dataset_name,
        "skipped": False,
        "routes": routes,
        "rows_per_route": rows_per_route,
        "global_metrics": global_metrics,
        "per_route_metrics": {r: m for r, (_, m, _) in per_route_models.items()},
        "pooled_table": pooled_table,
        "per_route_breakdown": per_route_breakdown,
    }


def main() -> None:
    mbta_result = run_comparison(
        "mbta_segments_corpus", offset_hours=7, min_rows_per_day=200,
        test_days=3, val_days=3, params=MBTA_TUNED_PARAMS,
    )

    bucr_path = REPO_ROOT / "datasets" / "bucr_segments_corpus.parquet"
    if bucr_path.exists():
        bucr_df = pd.read_parquet(bucr_path)
        bucr_routes = sorted(bucr_df["route_id"].unique())
        if len(bucr_routes) <= 1:
            print(f"\n=== bucr_segments_corpus: routes {bucr_routes} -- "
                  f"single-route corpus, per-route vs global comparison is "
                  f"meaningless here (per-route == global). Skipping. ===")
        else:
            run_comparison(
                "bucr_segments_corpus", offset_hours=6, min_rows_per_day=50,
                test_days=5, val_days=5, params=BUCR_TUNED_PARAMS,
            )
    else:
        print("\nbucr_segments_corpus.parquet not found -- skipping bUCR arm.")

    print("\nDONE")


if __name__ == "__main__":
    main()
