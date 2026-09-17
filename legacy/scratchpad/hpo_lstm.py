"""Optuna hyperparameter tuning for the segment-sequence LSTM/GRU model
(roadmap 5.2), mirroring `scratchpad/hpo_xgboost.py`'s structure.

Tunes architecture/optimization hyperparameters against the **validation**
split of a segment corpus (day-grouped, trip-safe -- same `day_grouped_split`
as hpo_xgboost.py, copied here verbatim so the splits are byte-identical and
results are directly comparable), selects the best trial by val MAE, then
reports **test** metrics ONCE for that config. Every trial is recorded to a
CSV.

Unlike hpo_xgboost.py, trials here use Optuna's `MedianPruner` with an
epoch-level intermediate-value callback: `train_lstm_pruned` (a thin
epoch-loop copy of `lstm.train.train_lstm`'s training loop) reports val MAE
after every epoch so unpromising trials can be pruned early -- this is what
makes a CPU-only RNN sweep affordable in the time budget.

Run (from repo root)::

    PYTHONPATH=.:models .venv-dl/bin/python scratchpad/hpo_lstm.py \\
        --dataset mbta_segments_corpus --trials 50
    PYTHONPATH=.:models .venv-dl/bin/python scratchpad/hpo_lstm.py \\
        --dataset bucr_segments_corpus --trials 50 \\
        --service-offset-h 6 --min-segs 50 --test-days 5 --val-days 5
"""
from __future__ import annotations

import argparse
import copy
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str((REPO_ROOT / "models").resolve()))

import torch  # noqa: E402
import optuna  # noqa: E402

from lstm.data import (  # noqa: E402
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    apply_preprocessor,
    build_sequences,
    coerce_feature_dtypes,
    fit_preprocessor,
    iter_batches,
)
from lstm.model import SegmentSequenceModel, masked_loss  # noqa: E402
from lstm.train import (  # noqa: E402
    _clean_bounds,
    _fit_target_stats,
    _run_eval,
    _seed_everything,
    _transform_sequence_targets,
    train_lstm,
)
from common.utils import clip_predictions  # noqa: E402


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


def train_lstm_pruned(
    pre_split: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    dataset_name: str,
    service_offset_h: float,
    params: Dict,
    seed: int,
    max_epochs: int,
    patience: int,
    trial: Optional[optuna.Trial] = None,
) -> Dict:
    """Thin copy of `lstm.train.train_lstm`'s training loop with an
    epoch-level Optuna report/prune hook. Returns the same
    {"metrics": {...}, "metadata": {...}} shape train_lstm does (minus
    save_model support -- HPO trials never persist to the registry)."""
    _seed_everything(seed)
    start_time = time.time()

    train_df, val_df, test_df = (df.copy() for df in pre_split)
    train_df = _clean_bounds(coerce_feature_dtypes(train_df))
    val_df = _clean_bounds(coerce_feature_dtypes(val_df))
    test_df = _clean_bounds(coerce_feature_dtypes(test_df))

    train_seqs_raw = build_sequences(train_df, FEATURE_COLUMNS, TARGET_COLUMN, service_offset_h)
    val_seqs_raw = build_sequences(val_df, FEATURE_COLUMNS, TARGET_COLUMN, service_offset_h)
    test_seqs_raw = build_sequences(test_df, FEATURE_COLUMNS, TARGET_COLUMN, service_offset_h)
    if not train_seqs_raw:
        raise ValueError("No training sequences -- check service_offset_h / data")

    preprocessor = fit_preprocessor(train_seqs_raw, FEATURE_COLUMNS)
    train_seqs = apply_preprocessor(train_seqs_raw, preprocessor)
    val_seqs = apply_preprocessor(val_seqs_raw, preprocessor) if val_seqs_raw else []
    test_seqs = apply_preprocessor(test_seqs_raw, preprocessor) if test_seqs_raw else []

    target_transform = params.get("target_transform", "none")
    target_stats = _fit_target_stats(train_seqs, target_transform)
    train_seqs_loss = _transform_sequence_targets(train_seqs, target_transform, target_stats)

    n_features = len(preprocessor.feature_cols)
    model = SegmentSequenceModel(
        n_features=n_features,
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        rnn_type=params["rnn_type"],
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"]
    )

    rng = np.random.RandomState(seed)
    best_val_mae = float("inf")
    best_state = None
    best_epoch = -1
    epochs_since_improve = 0
    epochs_trained = 0
    pruned = False

    for epoch in range(max_epochs):
        model.train()
        for batch in iter_batches(train_seqs_loss, batch_size=params["batch_size"], shuffle=True, rng=rng):
            optimizer.zero_grad()
            preds = model(batch.x, batch.lengths)
            batch_loss = masked_loss(preds, batch.y, batch.mask, loss=params["loss"])
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        epochs_trained = epoch + 1

        if val_seqs:
            y_true, y_pred, _ = _run_eval(model, val_seqs, params["batch_size"], target_transform, target_stats)
            val_mae = float(np.mean(np.abs(y_true - clip_predictions(y_pred))))
        else:
            val_mae = float("nan")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                break

        if trial is not None:
            trial.report(val_mae, epoch)
            if trial.should_prune():
                pruned = True
                break

    if pruned:
        raise optuna.TrialPruned()

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epochs_trained - 1

    wall_clock_seconds = time.time() - start_time

    from common.metrics import compute_all_metrics

    def _jsonable(d):
        return {k: float(v) for k, v in d.items()}

    val_metrics = {}
    if val_seqs:
        y_val, val_preds, _ = _run_eval(model, val_seqs, params["batch_size"], target_transform, target_stats)
        val_metrics = _jsonable(compute_all_metrics(y_val, clip_predictions(val_preds), prefix="val_"))

    test_metrics = {}
    if test_seqs:
        y_test, test_preds, _ = _run_eval(model, test_seqs, params["batch_size"], target_transform, target_stats)
        test_metrics = _jsonable(compute_all_metrics(y_test, clip_predictions(test_preds), prefix="test_"))

    return {
        "metrics": {**val_metrics, **test_metrics},
        "metadata": {
            "epochs_trained": epochs_trained,
            "best_epoch": best_epoch,
            "best_val_mae_seconds": best_val_mae,
            "train_wall_clock_seconds": wall_clock_seconds,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--service-offset-h", type=int, default=7)
    ap.add_argument("--min-segs", type=int, default=200)
    ap.add_argument("--test-days", type=int, default=3)
    ap.add_argument("--val-days", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--timeout", type=float, default=None,
                     help="Wall-clock budget in seconds for study.optimize (in addition to "
                          "--trials -- whichever limit is hit first stops the study). CPU-only "
                          "RNN training has widely variable per-trial cost (hidden_size/"
                          "num_layers/batch_size all move it), so a timeout is the reliable way "
                          "to keep a sweep inside a fixed wall-clock budget.")
    ap.add_argument("--n-startup-trials", type=int, default=8,
                     help="Trials run to completion (unpruned) before MedianPruner starts acting.")
    ap.add_argument("--n-warmup-steps", type=int, default=8,
                     help="Epochs run before a trial becomes eligible for pruning.")
    args = ap.parse_args()

    train_df, val_df, test_df, sizes = day_grouped_split(
        args.dataset, args.service_offset_h, args.min_segs, args.test_days, args.val_days
    )
    pre = (train_df, val_df, test_df)
    print(f"{args.dataset}: train {sizes[0]:,} / val {sizes[1]:,} / test {sizes[2]:,}")

    def val_mae(metrics):
        return metrics.get("val_mae_seconds")

    def test_mae(metrics):
        return metrics.get("test_mae_seconds")

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            hidden_size=trial.suggest_int("hidden_size", 32, 256, log=True),
            num_layers=trial.suggest_int("num_layers", 1, 3),
            dropout=trial.suggest_float("dropout", 0.0, 0.4),
            rnn_type=trial.suggest_categorical("rnn_type", ["lstm", "gru"]),
            learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            weight_decay=trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
            loss=trial.suggest_categorical("loss", ["huber", "mse"]),
            target_transform=trial.suggest_categorical("target_transform", ["none", "log1p", "standardize"]),
        )
        out = train_lstm_pruned(
            pre, args.dataset, args.service_offset_h, params,
            seed=args.seed, max_epochs=args.max_epochs, patience=args.patience, trial=trial,
        )
        return val_mae(out["metrics"])

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=args.n_warmup_steps, n_startup_trials=args.n_startup_trials),
        study_name=f"lstm_{args.dataset}",
    )
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout, show_progress_bar=False,
                    catch=(RuntimeError,))

    best = dict(study.best_params)
    print("\n=== best params (by val MAE) ===")
    for k, v in best.items():
        print(f"  {k}: {v}")

    # Final: retrain best config once more (unpruned, full patience) and report TEST once.
    best_out = train_lstm_pruned(
        pre, args.dataset, args.service_offset_h, best,
        seed=args.seed, max_epochs=args.max_epochs, patience=args.patience, trial=None,
    )
    b_val, b_test, b_r2 = val_mae(best_out["metrics"]), test_mae(best_out["metrics"]), best_out["metrics"].get("test_r2")
    print("\n=== tuned LSTM (test reported once) ===")
    print(f"val MAE {b_val:.2f}s | test MAE {b_test:.2f}s | test R² {b_r2:.3f} | "
          f"best_epoch {best_out['metadata']['best_epoch']} epochs_trained {best_out['metadata']['epochs_trained']}")

    out_csv = REPO_ROOT / "scratchpad" / f"hpo_trials_lstm_{args.dataset}.csv"
    study.trials_dataframe().to_csv(out_csv, index=False)
    print(f"\nrecorded {len(study.trials)} trials -> {out_csv}")
    n_pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    n_complete = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    print(f"pruned {n_pruned} / complete {n_complete} / total {len(study.trials)}")


if __name__ == "__main__":
    main()
