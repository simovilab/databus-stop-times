"""Optuna hyperparameter tuning for the XGBoost segment model (roadmap 5.1).

Tunes the five exposed XGBoost hyperparameters against the **validation** split
of a segment corpus (day-grouped walk-forward, trip-safe), selects the best trial
by val MAE, then reports the **test** metrics ONCE for that config -- test is
never seen during the search. Every trial is recorded to a CSV.

No changes to models/xgb/train.py: train_xgboost already accepts the five params
and returns val_/test_ metrics for a supplied pre_split, with save_model=False so
nothing is written to the registry.

Run (from repo root)::

    PYTHONPATH=. uv run --group bucr --with optuna --with duckdb \
        python scratchpad/hpo_xgboost.py --dataset mbta_segments_corpus --trials 50
    PYTHONPATH=. uv run --group bucr --with optuna \
        python scratchpad/hpo_xgboost.py --dataset bucr_segments_corpus --trials 50 \
            --service-offset-h 6 --min-segs 50 --test-days 5 --val-days 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str((REPO_ROOT / "models").resolve()))

import optuna  # noqa: E402
from xgb.train import train_xgboost  # noqa: E402

# Default config currently baked into train_xgboost (the baseline to beat).
DEFAULTS = dict(max_depth=5, n_estimators=200, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_weight=1.0, reg_alpha=0.0, reg_lambda=1.0, gamma=0.0)


def day_grouped_split(dataset: str, service_offset_h: int, min_segs: int,
                      test_days: int, val_days: int):
    df = pd.read_parquet(REPO_ROOT / "datasets" / f"{dataset}.parquet")
    df["cr_day"] = (df["vp_ts"] - pd.Timedelta(hours=service_offset_h)).dt.date
    counts = df.groupby("cr_day").size()
    usable = sorted(counts[counts >= min_segs].index)
    test = usable[-test_days:]
    val = usable[-(test_days + val_days):-test_days]
    train = usable[:-(test_days + val_days)]
    return (
        df[df["cr_day"].isin(train)].copy(),
        df[df["cr_day"].isin(val)].copy(),
        df[df["cr_day"].isin(test)].copy(),
        (len(train), len(val), len(test)),
    )


def _fit(dataset, params, pre_split):
    cp = tuple(x.copy() for x in pre_split)
    out = train_xgboost(dataset_name=dataset, save_model=False, pre_split=cp, **params)
    return out.get("metrics", {})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--service-offset-h", type=int, default=7)  # MBTA 3am ET
    ap.add_argument("--min-segs", type=int, default=200)
    ap.add_argument("--test-days", type=int, default=3)
    ap.add_argument("--val-days", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--early-stop-rounds", type=int, default=0,
                    help="If >0, drop n_estimators from the search, cap it high, "
                         "and let early stopping on val pick the tree count.")
    args = ap.parse_args()
    es = args.early_stop_rounds or None
    es_ceiling = 3000  # hard cap when early stopping chooses the tree count

    train_df, val_df, test_df, sizes = day_grouped_split(
        args.dataset, args.service_offset_h, args.min_segs, args.test_days, args.val_days
    )
    pre = (train_df, val_df, test_df)
    print(f"{args.dataset}: train {sizes[0]:,} / val {sizes[1]:,} / test {sizes[2]:,}")

    def val_mae(metrics):
        return metrics.get("val_mae_seconds") or (metrics.get("val_mae_minutes", 0) * 60)

    def test_mae(metrics):
        return metrics.get("test_mae_seconds") or (metrics.get("test_mae_minutes", 0) * 60)

    # Baseline (current defaults) for reference.
    base_m = _fit(args.dataset, DEFAULTS, pre)
    print(f"DEFAULT config -> val MAE {val_mae(base_m):.2f}s | "
          f"test MAE {test_mae(base_m):.2f}s | test R² {base_m.get('test_r2'):.3f}")

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            max_depth=trial.suggest_int("max_depth", 3, 12),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 50.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 100.0, log=True),
            gamma=trial.suggest_float("gamma", 1e-4, 5.0, log=True),
        )
        if es is not None:
            # Early stopping picks the tree count; don't also tune it.
            params["n_estimators"] = es_ceiling
            params["early_stopping_rounds"] = es
        else:
            params["n_estimators"] = trial.suggest_int("n_estimators", 100, 1000, step=50)
        m = _fit(args.dataset, params, pre)
        return val_mae(m)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        study_name=f"xgb_{args.dataset}",
    )
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    best = dict(study.best_params)
    if es is not None:
        best["n_estimators"] = es_ceiling
        best["early_stopping_rounds"] = es
    print("\n=== best params (by val MAE) ===")
    for k, v in best.items():
        print(f"  {k}: {v}")

    # Final: retrain best, report TEST once.
    best_m = _fit(args.dataset, best, pre)
    if es is not None:
        # best_iteration is on the model, not in metrics; refit exposes it via metadata.
        bi = train_xgboost(dataset_name=args.dataset, save_model=False,
                           pre_split=tuple(x.copy() for x in pre),
                           **best).get("metadata", {}).get("best_iteration")
        print(f"early stopping selected best_iteration = {bi} "
              f"(ceiling {es_ceiling}, patience {es})")
    b_val, b_test, b_r2 = val_mae(best_m), test_mae(best_m), best_m.get("test_r2")
    d_val, d_test, d_r2 = val_mae(base_m), test_mae(base_m), base_m.get("test_r2")
    print("\n=== tuned vs default (test reported once) ===")
    print(f"{'':10s} {'val MAE':>9s} {'test MAE':>9s} {'test R²':>8s}")
    print(f"{'default':10s} {d_val:8.2f}s {d_test:8.2f}s {d_r2:8.3f}")
    print(f"{'tuned':10s} {b_val:8.2f}s {b_test:8.2f}s {b_r2:8.3f}")
    print(f"test MAE improvement: {(d_test - b_test) / d_test * 100:+.1f}%   "
          f"R² {b_r2 - d_r2:+.3f}")

    out_csv = REPO_ROOT / "scratchpad" / f"hpo_trials_{args.dataset}.csv"
    study.trials_dataframe().to_csv(out_csv, index=False)
    print(f"\nrecorded {len(study.trials)} trials -> {out_csv}")


if __name__ == "__main__":
    main()
