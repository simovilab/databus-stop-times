"""Score an etaval live-run Parquet with the statistical layer (roadmap 4.3).

etaval (sibling repo) replays or streams a real GTFS-RT feed, asks every
registered predictor for arrival times, detects actual arrivals independently,
and writes one row per (trip, stop, predictor) with `error_s` and `horizon_s`.
Its own `etaval report` prints point estimates only; this adds the uncertainty
and significance the paper needs, stratified by prediction horizon -- which is
the axis the served models turn out to fail on.

Errors within one trip are strongly correlated, so CIs use the cluster
bootstrap over `trip_key`, and model-vs-model claims use a paired test on the
rows where both predictors answered the same (trip, stop).

Run (from repo root)::

    PYTHONPATH=.:models .venv-dl/bin/python scratchpad/analyze_live_validation.py \
        /path/to/run.parquet --baseline gtfsrt
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str((REPO_ROOT / "models").resolve()))

from evaluation.stats import bootstrap_metric_ci, correct_pvalues, paired_test  # noqa: E402

# Horizon buckets. etaval's own report uses <2m / 2-10m / >10m; keep them
# identical so the two outputs can be read side by side.
BUCKETS = [(-np.inf, 120, "<2m"), (120, 600, "2-10m"), (600, np.inf, ">10m")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--baseline", default="gtfsrt",
                    help="predictor_id every other predictor is tested against")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--min-n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    df = pd.read_parquet(args.parquet)
    df["abs_err"] = df["error_s"].abs()
    df["bucket"] = pd.cut(
        df["horizon_s"],
        [b[0] for b in BUCKETS] + [BUCKETS[-1][1]],
        labels=[b[2] for b in BUCKETS],
    )

    for label in [b[2] for b in BUCKETS]:
        sub = df[df["bucket"] == label]
        if sub.empty:
            continue
        print(f"\n=== horizon {label} ===")

        for pid in sorted(sub["predictor_id"].unique()):
            s = sub[sub["predictor_id"] == pid]
            if len(s) < args.min_n:
                continue
            # bootstrap_metric_ci computes MAE(y_true, y_pred); the errors are
            # already residuals, so compare them against zero.
            r = bootstrap_metric_ci(
                np.zeros(len(s)), s["abs_err"].to_numpy(),
                groups=s["trip_key"].to_numpy(), n_boot=args.n_boot, seed=args.seed,
            )
            print(f"  {pid:28s} MAE {r.estimate:9.1f}s  "
                  f"95% CI [{r.ci_lower:8.1f}, {r.ci_upper:8.1f}]  "
                  f"n={len(s):5d} trips={r.n_clusters}")

        # Paired tests against the baseline, on rows where both answered.
        base = sub[sub["predictor_id"] == args.baseline].set_index(["trip_key", "stop_id"])
        if base.empty:
            continue
        rows, pvals = [], []
        for pid in sorted(sub["predictor_id"].unique()):
            if pid == args.baseline:
                continue
            other = sub[sub["predictor_id"] == pid].set_index(["trip_key", "stop_id"])
            j = other[["abs_err"]].join(base[["abs_err"]], how="inner",
                                        lsuffix="_m", rsuffix="_b")
            if len(j) < args.min_n:
                continue
            t = paired_test(np.zeros(len(j)), j["abs_err_m"].to_numpy(),
                            j["abs_err_b"].to_numpy(), method="wilcoxon")
            rows.append((pid, t))
            pvals.append(t.p_value)

        if rows:
            adj = correct_pvalues(np.array(pvals), method="holm")
            print(f"  -- paired vs {args.baseline} (Holm-corrected) --")
            for (pid, t), p_adj, rej in zip(rows, adj["pvals_adjusted"], adj["reject"]):
                verdict = "significant" if rej else "not significant"
                print(f"  {pid:28s} diff {t.mean_diff:+9.1f}s ({t.diff_percent:+7.1f}%)  "
                      f"n={t.n:5d}  p={t.p_value:.3g}  p_adj={p_adj:.3g}  {verdict}")


if __name__ == "__main__":
    main()
