"""Final LSTM training run per agency (roadmap 5.2): retrain the val-best
config found by `scratchpad/hpo_lstm.py` with a fixed seed, full patience,
`save_model=True`, and verify the registry round-trips (reload -> identical
test predictions).

Splits are the same day-grouped split used throughout (`day_grouped_split`,
copied verbatim from `hpo_xgboost.py`/`hpo_lstm.py` so results are directly
comparable to the tuned-XGBoost numbers in docs/RESEARCH_ROADMAP.md).

Run (from repo root)::

    PYTHONPATH=.:models .venv-dl/bin/python scratchpad/train_lstm_final.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str((REPO_ROOT / "models").resolve()))

from common.registry import get_registry  # noqa: E402
from lstm.train import train_lstm  # noqa: E402


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
        (len(df[df["cr_day"].isin(train)]), len(df[df["cr_day"].isin(val)]), len(df[df["cr_day"].isin(test)])),
    )


# Val-best configs from scratchpad/hpo_trials_lstm_{dataset}.csv (see
# scratchpad/logs/hpo_lstm_{mbta,bucr}.log for the full study output).
AGENCIES = {
    "mbta": dict(
        dataset="mbta_segments_corpus", service_offset_h=7, min_segs=200,
        test_days=3, val_days=3,
        params=dict(hidden_size=99, num_layers=1, dropout=0.38783385110582347,
                    rnn_type="gru", learning_rate=0.006161049539380964,
                    batch_size=64, weight_decay=1.6832027985721884e-08,
                    loss="mse", target_transform="log1p"),
    ),
    "bucr": dict(
        dataset="bucr_segments_corpus", service_offset_h=6, min_segs=50,
        test_days=5, val_days=5,
        params=dict(hidden_size=42, num_layers=2, dropout=0.21087533839745937,
                    rnn_type="lstm", learning_rate=0.0059075347183753625,
                    batch_size=64, weight_decay=4.844142187948514e-06,
                    loss="huber", target_transform="standardize"),
    ),
}


def main() -> None:
    registry = get_registry()
    results = {}

    for agency, cfg in AGENCIES.items():
        print(f"\n{'=' * 70}\n{agency.upper()} final training\n{'=' * 70}")
        train_df, val_df, test_df, sizes = day_grouped_split(
            cfg["dataset"], cfg["service_offset_h"], cfg["min_segs"], cfg["test_days"], cfg["val_days"]
        )
        print(f"{cfg['dataset']}: train {sizes[0]:,} / val {sizes[1]:,} / test {sizes[2]:,} rows")

        t0 = time.time()
        out = train_lstm(
            dataset_name=cfg["dataset"],
            pre_split=(train_df, val_df, test_df),
            service_offset_h=cfg["service_offset_h"],
            max_epochs=300,
            patience=25,
            save_model=True,
            seed=42,
            **cfg["params"],
        )
        wall = time.time() - t0
        m = out["metrics"]
        model_key = out["metadata"]["model_key"]
        print(f"wall_clock={wall:.0f}s epochs_trained={out['metadata']['epochs_trained']} "
              f"best_epoch={out['metadata']['best_epoch']}")
        print(f"val_mae={m.get('val_mae_seconds'):.2f}s test_mae={m.get('test_mae_seconds'):.2f}s "
              f"test_r2={m.get('test_r2'):.3f}")
        print(f"model_key={model_key}")

        # --- Registry round trip: reload and verify identical test predictions. ---
        loaded = registry.load_model(model_key)
        preds_direct = out["model"].predict(test_df)
        preds_loaded = loaded.predict(test_df)
        max_diff = float(np.max(np.abs(preds_direct - preds_loaded))) if len(preds_direct) else 0.0
        assert np.allclose(preds_direct, preds_loaded, rtol=1e-6, atol=1e-6), (
            f"{agency}: registry round-trip predictions diverge (max diff {max_diff})"
        )
        print(f"registry round-trip OK (max |diff| = {max_diff:.2e}, n={len(preds_direct)})")

        results[agency] = dict(
            model_key=model_key,
            wall_clock_seconds=wall,
            val_mae_seconds=m.get("val_mae_seconds"),
            test_mae_seconds=m.get("test_mae_seconds"),
            test_r2=m.get("test_r2"),
            epochs_trained=out["metadata"]["epochs_trained"],
            best_epoch=out["metadata"]["best_epoch"],
        )

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for agency, r in results.items():
        print(f"{agency}: test MAE {r['test_mae_seconds']:.2f}s | test R² {r['test_r2']:.3f} | "
              f"wall {r['wall_clock_seconds']:.0f}s | key={r['model_key']}")


if __name__ == "__main__":
    main()
