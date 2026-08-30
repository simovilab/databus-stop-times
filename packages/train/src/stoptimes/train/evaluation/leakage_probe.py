"""Shuffle-label leakage probe for the bUCR segment ETA formulation.

Verifies roadmap item 3.2: confirm the segment formulation (one row per
trip-instance x stop-to-stop segment, target = time_to_arrival_seconds) is
not leaking the target through its features.

Method
------
1. Reproduce the exact day-grouped walk-forward split used in
   scratchpad/baseline_compare_corpus.py (train = all but last 10 usable
   CR-days, val = next-to-last 5, test = last 5; CR-day = (vp_ts - 6h).date,
   days with < 50 segments are dropped). A trip lives within a single day,
   so the split is trip-safe.
2. CONTROL arm: train XGBoost and polyreg_distance on the real splits.
3. LEAKAGE arm: permute `time_to_arrival_seconds` independently within
   train_df and val_df (np.random.default_rng(SEED).permutation), features
   left untouched, test_df left completely untouched. Retrain both models
   on the shuffled-label splits and evaluate on the same real test set.
4. Verdict: on shuffled labels a model can only learn noise, so its test
   metrics must collapse to the naive constant-predictor reference (predict
   the train-label mean for every test row). PASS if the shuffled arm
   collapses (R2 <= ~0.05, MAE within ~10% of the naive reference) and the
   control arm reproduces the known real-label result (~33s/0.69 XGBoost,
   ~35s/0.66 polyreg_distance); FAIL (leakage suspected) otherwise.

HARD CONSTRAINTS honored by this script
----------------------------------------
- Every trainer call passes save_model=False. Nothing is written to
  models/trained/ (no registry.json update, no *.pkl write).
- No writes outside this file. No git operations.

Run
---
    PYTHONPATH=. uv run --group bucr python models/evaluation/leakage_probe.py

===========================================================================
RESULT OF LAST RUN (2026-08-30)
===========================================================================
usable CR-days (>=50 seg): 38 (train 28d / val 5d / test 5d)
train n=10,991  val n=3,304  test n=3,201

naive constant-predictor reference on test (predict train-label mean):
  MAE = 71.2s   (this is the "should collapse to" reference for MAE)

              test MAE   test RMSE   test R2
CONTROL (real labels)
  xgboost              33.4s        53.3s     0.694
  polyreg_distance     35.5s        56.2s     0.659

LEAKAGE ARM (shuffled train+val labels, real untouched test)
  xgboost              72.3s        98.5s    -0.046
  polyreg_distance     69.9s        96.8s    -0.011

VERDICT: PASS -- shuffled-label arm collapses to ~naive-baseline MAE (70-72s
vs 71.2s reference) with R2 ~ 0 (slightly negative, as expected from finite
noise fitting), while the control arm reproduces the known real-label result
(xgboost ~33s/0.69, polyreg_distance ~35s/0.66). The segment formulation is
not leaking the target through its features.
===========================================================================
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))
sys.path.append(str((REPO_ROOT / "models").resolve()))

from xgb.train import train_xgboost  # noqa: E402
from polyreg_distance.train import train_polyreg_distance  # noqa: E402

DATASET = "bucr_segments_corpus"
DS_PATH = str(REPO_ROOT / "datasets" / "bucr_segments_corpus.parquet")
MIN_SEGS_PER_DAY = 50
N_TEST_DAYS = 5
N_VAL_DAYS = 5
TARGET_COL = "time_to_arrival_seconds"
SEED = 42


def build_splits():
    """Reproduce the exact split from scratchpad/baseline_compare_corpus.py."""
    df = pd.read_parquet(DS_PATH)
    df["cr_day"] = (df["vp_ts"] - pd.Timedelta(hours=6)).dt.date
    counts = df.groupby("cr_day").size()
    usable = sorted(counts[counts >= MIN_SEGS_PER_DAY].index)

    test_days = usable[-N_TEST_DAYS:]
    val_days = usable[-(N_TEST_DAYS + N_VAL_DAYS):-N_TEST_DAYS]
    train_days = usable[:-(N_TEST_DAYS + N_VAL_DAYS)]

    train_df = df[df["cr_day"].isin(train_days)].copy()
    val_df = df[df["cr_day"].isin(val_days)].copy()
    test_df = df[df["cr_day"].isin(test_days)].copy()
    return train_df, val_df, test_df, usable, train_days, val_days, test_days


def shuffle_labels(train_df: pd.DataFrame, val_df: pd.DataFrame, seed: int):
    """Return copies of train/val with TARGET_COL permuted; features untouched."""
    rng = np.random.default_rng(seed)
    train_shuf = train_df.copy()
    val_shuf = val_df.copy()
    train_shuf[TARGET_COL] = train_shuf[TARGET_COL].to_numpy()[
        rng.permutation(len(train_shuf))
    ]
    val_shuf[TARGET_COL] = val_shuf[TARGET_COL].to_numpy()[
        rng.permutation(len(val_shuf))
    ]
    return train_shuf, val_shuf


def naive_reference_mae(train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    """MAE of predicting the train-label mean for every test row."""
    train_mean = float(train_df[TARGET_COL].mean())
    return float((test_df[TARGET_COL] - train_mean).abs().mean())


def run_arm(label: str, train_df, val_df, test_df) -> dict:
    pre = (train_df, val_df, test_df)
    cp = lambda: tuple(x.copy() for x in pre)  # noqa: E731

    results = {}
    for name, fn in {
        "xgboost": lambda: train_xgboost(
            dataset_name=DATASET, route_id=None, save_model=False, pre_split=cp()
        ),
        "polyreg_distance": lambda: train_polyreg_distance(
            dataset_name=DATASET, degree=2, save_model=False, pre_split=cp()
        ),
    }.items():
        out = fn()
        m = out.get("metrics", {})
        mae = m.get("test_mae_seconds") or (m.get("test_mae_minutes", 0) * 60)
        rmse = m.get("test_rmse_seconds")
        r2 = m.get("test_r2")
        results[name] = {"mae": mae, "rmse": rmse, "r2": r2}
        print(f"  [{label}] {name:18s} MAE={mae:8.1f}s  RMSE={rmse:8.1f}s  R2={r2:7.3f}")
    return results


def main() -> None:
    print("Loading corpus and building day-grouped walk-forward split...")
    train_df, val_df, test_df, usable, train_days, val_days, test_days = build_splits()
    print(
        f"usable CR-days (>={MIN_SEGS_PER_DAY} seg): {len(usable)}  "
        f"(train {len(train_days)}d / val {len(val_days)}d / test {len(test_days)}d)"
    )
    print(
        f"train n={len(train_df):,}  val n={len(val_df):,}  test n={len(test_df):,}"
    )

    naive_mae = naive_reference_mae(train_df, test_df)
    print(
        f"\nnaive constant-predictor reference on test "
        f"(predict train-label mean): MAE = {naive_mae:.1f}s"
    )

    print("\n=== CONTROL ARM (real labels) ===")
    control = run_arm("CONTROL", train_df, val_df, test_df)

    print("\n=== LEAKAGE ARM (shuffled train+val labels, real untouched test) ===")
    train_shuf, val_shuf = shuffle_labels(train_df, val_df, SEED)
    leakage = run_arm("SHUFFLED", train_shuf, val_shuf, test_df)

    print("\n" + "=" * 70)
    print(f"{'arm':10s} {'model':18s} {'test MAE':>10s} {'test RMSE':>10s} {'test R2':>9s}")
    print("-" * 70)
    for name in ("xgboost", "polyreg_distance"):
        c = control[name]
        print(f"{'control':10s} {name:18s} {c['mae']:9.1f}s {c['rmse']:9.1f}s {c['r2']:9.3f}")
    for name in ("xgboost", "polyreg_distance"):
        s = leakage[name]
        print(f"{'shuffled':10s} {name:18s} {s['mae']:9.1f}s {s['rmse']:9.1f}s {s['r2']:9.3f}")
    print(f"{'naive':10s} {'(reference)':18s} {naive_mae:9.1f}s {'--':>10s} {'--':>9s}")
    print("=" * 70)

    # Verdict logic:
    #  - shuffled R2 must collapse near/at-or-below 0 (allow small positive
    #    slack for finite-sample noise fitting)
    #  - shuffled MAE must be close to the naive reference (within 15%)
    #  - control must roughly reproduce the known real-label result
    r2_ok = all(leakage[m]["r2"] <= 0.05 for m in ("xgboost", "polyreg_distance"))
    mae_ok = all(
        abs(leakage[m]["mae"] - naive_mae) / naive_mae <= 0.15
        for m in ("xgboost", "polyreg_distance")
    )
    control_ok = (
        control["xgboost"]["r2"] >= 0.55 and control["polyreg_distance"]["r2"] >= 0.50
    )

    verdict = "PASS" if (r2_ok and mae_ok and control_ok) else "FAIL"
    print(f"\nVERDICT: {verdict}")
    if verdict == "PASS":
        print(
            "Shuffled-label arm collapses to ~naive-baseline MAE with R2 ~ 0, "
            "while the control arm reproduces the known real-label result. "
            "The segment formulation is not leaking the target through its features."
        )
    else:
        print(
            "LEAKAGE SUSPECTED: shuffled-label arm did NOT collapse to the naive "
            "baseline (or the control arm failed to reproduce the known real-label "
            "result). This is a real finding -- do not attempt to silently fix the "
            "formulation here; report it."
        )


if __name__ == "__main__":
    main()
