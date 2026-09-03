"""
Statistical layer for model evaluation (roadmap 4.3).

Provides the machinery needed to turn a leaderboard of point-estimate metrics
into publication-grade claims:

- Bootstrap confidence intervals on MAE (i.i.d. and cluster/block bootstrap).
- Paired model-vs-model significance tests on the per-observation absolute
  error differential (Wilcoxon signed-rank, Diebold-Mariano, paired bootstrap).
- Multiple-comparison correction (Holm-Bonferroni, Benjamini-Hochberg).
- A leaderboard-level entry point (`compare_all`) that assembles all of the
  above into one DataFrame, suitable for a paper table.
- Thin, DataFrame-returning wrappers around the existing (but previously
  unused) `error_analysis` and `prediction_intervals` helpers in
  `models/common/metrics.py`, with a stratification argument.

Everything here builds on `models/common/metrics.py` rather than
reimplementing it: `mae_seconds`, `error_analysis`, and `prediction_intervals`
are imported, not duplicated.

Why the cluster bootstrap matters
----------------------------------
Rows in this corpus are per-(trip, segment) observations: many rows come from
the same trip instance and are strongly correlated (the same vehicle running
a few minutes early/late drags every segment's error the same direction). The
plain i.i.d. bootstrap resamples *rows*, treating them as if they were
independent draws, which understates the true sampling variance of MAE and
of paired differences. The cluster (block) bootstrap instead resamples whole
*groups* (e.g. `trip_id`, or a service day) with replacement and keeps every
row of a chosen group together. Pass `groups=` to any function below that
accepts it to get the more honest, wider interval. Given the strong
within-trip correlation in this domain, prefer the cluster bootstrap over
the i.i.d. one whenever `trip_id` (or another natural cluster key) is
available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from common.metrics import (
    mae_seconds,
    error_analysis as _error_analysis,
    prediction_intervals as _prediction_intervals,
)

__all__ = [
    "BootstrapResult",
    "PairedTestResult",
    "bootstrap_metric_ci",
    "paired_test",
    "correct_pvalues",
    "compare_all",
    "compare_all_to_markdown",
    "error_analysis_by",
    "prediction_intervals_by",
]


# ---------------------------------------------------------------------------
# Bootstrap CIs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    """Result of `bootstrap_metric_ci`."""

    estimate: float
    ci_lower: float
    ci_upper: float
    se: float
    alpha: float
    n_boot: int
    n: int
    clustered: bool
    n_clusters: Optional[int] = None

    def as_dict(self) -> Dict[str, float]:
        return {
            "estimate": self.estimate,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "se": self.se,
            "alpha": self.alpha,
            "n_boot": self.n_boot,
            "n": self.n,
            "clustered": self.clustered,
            "n_clusters": self.n_clusters,
        }


def _resample_indices_iid(n: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Vectorized i.i.d. bootstrap: (n_boot, n) matrix of row indices."""
    return rng.integers(0, n, size=(n_boot, n))


def _resample_indices_clustered(
    groups: np.ndarray, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Vectorized cluster/block bootstrap: resample whole groups with
    replacement, n_clusters draws per replicate, keeping every row of a
    chosen group together. Returns a (n_boot, n_rows_total_per_rep) matrix
    of row indices into the *original* (unsorted) array — but since group
    sizes vary, we instead return a list-like via a fixed-size approach:
    we resample `n_clusters` groups per replicate and concatenate their row
    indices. To keep this vectorized we precompute, per unique group, the
    array of row positions, then build each replicate by concatenation.
    This is inherently ragged (unequal group sizes) so full vectorization
    across replicates is not possible without padding; we instead vectorize
    the *group draw* (cheap) and only loop over the n_boot replicates for
    the concatenation step, which is fast since n_boot loop bodies are just
    array concatenations, not python-level per-row work.
    """
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    n_clusters = len(unique_groups)
    # row positions per group, precomputed once
    order = np.argsort(inverse, kind="stable")
    sorted_inv = inverse[order]
    boundaries = np.searchsorted(sorted_inv, np.arange(n_clusters + 1))
    group_rows = [order[boundaries[i]:boundaries[i + 1]] for i in range(n_clusters)]

    # (n_boot, n_clusters) matrix of which groups get drawn per replicate
    group_draws = rng.integers(0, n_clusters, size=(n_boot, n_clusters))

    replicate_indices = []
    for b in range(n_boot):
        parts = [group_rows[g] for g in group_draws[b]]
        replicate_indices.append(np.concatenate(parts))
    return replicate_indices, n_clusters


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] = mae_seconds,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
    groups: Optional[np.ndarray] = None,
    block: Optional[bool] = None,
) -> BootstrapResult:
    """
    Bootstrap confidence interval for a metric (default: MAE in seconds).

    Two resampling schemes:
      - i.i.d. bootstrap over rows (default, `groups=None`).
      - cluster/block bootstrap that resamples whole *groups* with
        replacement (e.g. `groups=df["trip_id"].values`) — pass `groups=`
        to reach it. This is the recommended mode whenever rows are not
        independent draws (e.g. many rows per trip instance): the i.i.d.
        bootstrap understates the true sampling variance in that case, and
        the cluster bootstrap gives an honest, typically wider, CI.

    `block=True/False` can force the mode explicitly (True requires
    `groups`); by default the cluster bootstrap is used iff `groups` is
    given.

    Vectorized: the i.i.d. path builds one (n_boot, n) index matrix and
    evaluates the metric via a fully array-based MAE (no python loop over
    replicates for the metric itself). The cluster path loops over
    replicates only for cheap array concatenation (group sizes are ragged),
    which is still fast — 10k reps over ~30k rows in low single-digit
    seconds.

    Returns
    -------
    BootstrapResult(estimate, ci_lower, ci_upper, se, alpha, n_boot, n,
                     clustered, n_clusters)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if len(y_pred) != n:
        raise ValueError("y_true and y_pred must be the same length")

    use_cluster = bool(groups is not None) if block is None else bool(block)
    if use_cluster and groups is None:
        raise ValueError("block=True requires groups")

    rng = np.random.default_rng(seed)
    point = float(metric(y_true, y_pred))

    if use_cluster:
        groups = np.asarray(groups)
        if len(groups) != n:
            raise ValueError("groups must be the same length as y_true")
        replicate_indices, n_clusters = _resample_indices_clustered(groups, n_boot, rng)
        boot_stats = np.empty(n_boot, dtype=float)
        for b, idx in enumerate(replicate_indices):
            boot_stats[b] = metric(y_true[idx], y_pred[idx])
    else:
        n_clusters = None
        idx_matrix = _resample_indices_iid(n, n_boot, rng)
        # Fast-path vectorized MAE without a python loop; fall back to a
        # loop for arbitrary metric callables (still fast: n_boot calls).
        if metric is mae_seconds:
            abs_err = np.abs(y_true - y_pred)
            boot_stats = abs_err[idx_matrix].mean(axis=1)
        else:
            boot_stats = np.empty(n_boot, dtype=float)
            yt = y_true[idx_matrix]
            yp = y_pred[idx_matrix]
            for b in range(n_boot):
                boot_stats[b] = metric(yt[b], yp[b])

    lo, hi = np.percentile(boot_stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    se = float(np.std(boot_stats, ddof=1))

    return BootstrapResult(
        estimate=point,
        ci_lower=float(lo),
        ci_upper=float(hi),
        se=se,
        alpha=alpha,
        n_boot=n_boot,
        n=n,
        clustered=use_cluster,
        n_clusters=n_clusters,
    )


# ---------------------------------------------------------------------------
# Paired model-vs-model tests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedTestResult:
    """Result of `paired_test`."""

    method: str
    mean_diff: float
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    statistic: float
    p_value: float
    n: int
    mae_a: float
    mae_b: float
    diff_seconds: float
    diff_percent: float

    def as_dict(self) -> Dict[str, Union[str, float, int]]:
        return {
            "method": self.method,
            "mean_diff": self.mean_diff,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "n": self.n,
            "mae_a": self.mae_a,
            "mae_b": self.mae_b,
            "diff_seconds": self.diff_seconds,
            "diff_percent": self.diff_percent,
        }


def _newey_west_variance(d: np.ndarray, lags: int) -> float:
    """
    Newey-West (HAC) long-run variance estimate of the mean of `d`.

    var(mean(d)) ~= (1/n) * [ gamma_0 + 2 * sum_{k=1..lags} w_k * gamma_k ]
    with Bartlett weights w_k = 1 - k/(lags+1), gamma_k the (biased, /n)
    sample autocovariance at lag k. This is the standard Diebold-Mariano
    variance estimator for a possibly autocorrelated loss differential.
    """
    n = len(d)
    d_c = d - d.mean()
    gamma0 = np.dot(d_c, d_c) / n
    var = gamma0
    for k in range(1, lags + 1):
        gamma_k = np.dot(d_c[k:], d_c[:-k]) / n
        w_k = 1.0 - k / (lags + 1)
        var += 2 * w_k * gamma_k
    return var / n


def _diebold_mariano(d: np.ndarray, alpha: float) -> PairedTestResult:
    """
    Diebold-Mariano test on a loss differential `d = |e_a| - |e_b|`.

    Lag rule: `lags = floor(4 * (n/100)^(2/9))`, the standard Newey-West
    rule-of-thumb (Newey & West 1994) used because it scales the truncation
    lag sublinearly with n and is the conventional default for DM tests on
    macro/finance-style time series without a known autocorrelation order.
    We use it here because the paired series is a time-ordered sequence of
    per-observation (or rolling-window) errors that can carry short-range
    autocorrelation (e.g. consecutive segments of the same trip); assuming
    i.i.d. errors (lags=0) would understate the variance and overstate
    significance.
    """
    n = len(d)
    lags = max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))
    lags = min(lags, n - 2) if n > 2 else 0
    var_mean = _newey_west_variance(d, lags)
    mean_d = float(d.mean())
    se = float(np.sqrt(max(var_mean, 0.0)))
    if se == 0.0:
        dm_stat = 0.0 if mean_d == 0 else np.inf * np.sign(mean_d)
        p_value = 1.0 if mean_d == 0 else 0.0
    else:
        dm_stat = mean_d / se
        p_value = 2 * (1 - sp_stats.norm.cdf(abs(dm_stat)))
    z = sp_stats.norm.ppf(1 - alpha / 2)
    ci_lo = mean_d - z * se
    ci_hi = mean_d + z * se
    return mean_d, ci_lo, ci_hi, float(dm_stat), float(p_value), lags


def paired_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    method: str = "wilcoxon",
    groups: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    n_boot: int = 10_000,
    seed: int = 42,
) -> PairedTestResult:
    """
    Paired test on the per-observation absolute-error differential
    `d_i = |y_i - a_i| - |y_i - b_i|`. A negative mean difference means
    model A has lower error than model B (A is better).

    method:
      - "wilcoxon": paired Wilcoxon signed-rank test on `d` (scipy). Good
        default for a static test set; distribution-free, no independence-
        beyond-pairing assumption needed, but treats observations as
        exchangeable (no explicit correlation adjustment).
      - "dm": Diebold-Mariano test with a Newey-West/HAC variance estimate
        on `d`, appropriate when `d` is a time-ordered sequence (e.g.
        rolling-window errors) that may be autocorrelated. See
        `_diebold_mariano` for the lag rule.
      - "bootstrap": paired bootstrap CI on `mean(d)`; cluster-aware via
        `groups` (recommended when rows are not independent, e.g. many
        rows per trip).

    Returns a `PairedTestResult` with the mean difference, its CI (except
    for a plain Wilcoxon test, which has no natural CI and returns None),
    the test statistic, p-value, n, each model's MAE, and the practical
    effect size (difference in MAE seconds and percent of model B's MAE).
    """
    y_true = np.asarray(y_true, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    n = len(y_true)
    if not (len(pred_a) == n and len(pred_b) == n):
        raise ValueError("y_true, pred_a, pred_b must be the same length")

    abs_err_a = np.abs(y_true - pred_a)
    abs_err_b = np.abs(y_true - pred_b)
    d = abs_err_a - abs_err_b
    mae_a = float(abs_err_a.mean())
    mae_b = float(abs_err_b.mean())
    diff_seconds = mae_a - mae_b
    diff_percent = 100.0 * diff_seconds / mae_b if mae_b != 0 else float("nan")

    if method == "wilcoxon":
        if np.allclose(d, 0.0):
            stat, p_value = 0.0, 1.0
        else:
            stat, p_value = sp_stats.wilcoxon(d, zero_method="wilcox", mode="auto")
        return PairedTestResult(
            method="wilcoxon",
            mean_diff=float(d.mean()),
            ci_lower=None,
            ci_upper=None,
            statistic=float(stat),
            p_value=float(p_value),
            n=n,
            mae_a=mae_a,
            mae_b=mae_b,
            diff_seconds=diff_seconds,
            diff_percent=diff_percent,
        )

    if method == "dm":
        mean_d, ci_lo, ci_hi, stat, p_value, lags = _diebold_mariano(d, alpha)
        result = PairedTestResult(
            method="dm",
            mean_diff=mean_d,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            statistic=stat,
            p_value=p_value,
            n=n,
            mae_a=mae_a,
            mae_b=mae_b,
            diff_seconds=diff_seconds,
            diff_percent=diff_percent,
        )
        return result

    if method == "bootstrap":
        rng = np.random.default_rng(seed)
        if groups is not None:
            groups_arr = np.asarray(groups)
            replicate_indices, _ = _resample_indices_clustered(groups_arr, n_boot, rng)
            boot_means = np.empty(n_boot, dtype=float)
            for b, idx in enumerate(replicate_indices):
                boot_means[b] = d[idx].mean()
        else:
            idx_matrix = _resample_indices_iid(n, n_boot, rng)
            boot_means = d[idx_matrix].mean(axis=1)
        mean_d = float(d.mean())
        lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        # two-sided bootstrap p-value: proportion of replicates on the
        # opposite side of zero from the observed sign, doubled.
        if mean_d >= 0:
            p_value = 2 * min((boot_means <= 0).mean(), 0.5)
        else:
            p_value = 2 * min((boot_means >= 0).mean(), 0.5)
        p_value = float(min(max(p_value, 0.0), 1.0))
        se = float(np.std(boot_means, ddof=1))
        stat = mean_d / se if se > 0 else 0.0
        return PairedTestResult(
            method="bootstrap",
            mean_diff=mean_d,
            ci_lower=float(lo),
            ci_upper=float(hi),
            statistic=float(stat),
            p_value=p_value,
            n=n,
            mae_a=mae_a,
            mae_b=mae_b,
            diff_seconds=diff_seconds,
            diff_percent=diff_percent,
        )

    raise ValueError(f"unknown method: {method!r} (expected wilcoxon|dm|bootstrap)")


# ---------------------------------------------------------------------------
# Multiple-comparison correction
# ---------------------------------------------------------------------------


def correct_pvalues(
    pvals: Sequence[float], method: str = "holm", alpha: float = 0.05
) -> Dict[str, np.ndarray]:
    """
    Multiple-comparison correction on a family of p-values.

    method="holm": Holm-Bonferroni step-down procedure (controls FWER).
    method="bh": Benjamini-Hochberg step-up procedure (controls FDR).

    Returns {"pvals_adjusted": np.ndarray, "reject": np.ndarray[bool]},
    in the original (unsorted) order of `pvals`.
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return {"pvals_adjusted": np.array([]), "reject": np.array([], dtype=bool)}

    order = np.argsort(p)
    p_sorted = p[order]

    if method == "holm":
        # adjusted_i = max_{j<=i} (m - j) * p_sorted[j], cumulative max, clipped to 1
        multipliers = m - np.arange(m)
        adj_sorted = np.maximum.accumulate(multipliers * p_sorted)
        adj_sorted = np.clip(adj_sorted, 0, 1)
    elif method == "bh":
        # adjusted_i = min_{j>=i} (m/(j+1)) * p_sorted[j], reverse cumulative min, clipped to 1
        ranks = np.arange(1, m + 1)
        raw = p_sorted * m / ranks
        adj_sorted = np.minimum.accumulate(raw[::-1])[::-1]
        adj_sorted = np.clip(adj_sorted, 0, 1)
    else:
        raise ValueError(f"unknown method: {method!r} (expected holm|bh)")

    pvals_adjusted = np.empty(m, dtype=float)
    pvals_adjusted[order] = adj_sorted
    reject = pvals_adjusted <= alpha
    return {"pvals_adjusted": pvals_adjusted, "reject": reject}


# ---------------------------------------------------------------------------
# Leaderboard-level entry point
# ---------------------------------------------------------------------------


def compare_all(
    y_true: np.ndarray,
    preds: Dict[str, np.ndarray],
    *,
    baseline: str,
    groups: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    seed: int = 42,
    n_boot: int = 10_000,
    paired_method: str = "wilcoxon",
    correction: str = "holm",
) -> pd.DataFrame:
    """
    Build the leaderboard-with-uncertainty table: one row per model with its
    MAE, bootstrap CI (cluster bootstrap if `groups` is given, else i.i.d.),
    and a paired comparison against `baseline` (mean diff, CI, raw p,
    Holm/BH-adjusted p across the family of non-baseline comparisons, and a
    significance flag at `alpha`).

    Rows are sorted by MAE ascending (best model first).

    Columns: model, mae_seconds, ci_lower, ci_upper, se, n,
             diff_vs_baseline_seconds, diff_vs_baseline_percent,
             test_method, test_statistic, p_value, p_value_adjusted,
             significant, is_baseline.
    """
    if baseline not in preds:
        raise ValueError(f"baseline {baseline!r} not found in preds keys {list(preds)}")

    y_true = np.asarray(y_true, dtype=float)
    rows = []
    raw_pvals = []
    non_baseline_names = [name for name in preds if name != baseline]

    for name, y_pred in preds.items():
        y_pred = np.asarray(y_pred, dtype=float)
        ci = bootstrap_metric_ci(
            y_true, y_pred, metric=mae_seconds, n_boot=n_boot, alpha=alpha,
            seed=seed, groups=groups,
        )
        row = {
            "model": name,
            "mae_seconds": ci.estimate,
            "ci_lower": ci.ci_lower,
            "ci_upper": ci.ci_upper,
            "se": ci.se,
            "n": ci.n,
            "is_baseline": name == baseline,
        }
        if name == baseline:
            row.update({
                "diff_vs_baseline_seconds": 0.0,
                "diff_vs_baseline_percent": 0.0,
                "test_method": None,
                "test_statistic": None,
                "p_value": None,
                "p_value_adjusted": None,
                "significant": False,
            })
        else:
            test = paired_test(
                y_true, y_pred, np.asarray(preds[baseline], dtype=float),
                method=paired_method, groups=groups, alpha=alpha, n_boot=n_boot, seed=seed,
            )
            row.update({
                "diff_vs_baseline_seconds": test.diff_seconds,
                "diff_vs_baseline_percent": test.diff_percent,
                "test_method": test.method,
                "test_statistic": test.statistic,
                "p_value": test.p_value,
                "p_value_adjusted": None,  # filled below
                "significant": False,  # filled below
            })
            raw_pvals.append(test.p_value)
        rows.append(row)

    if raw_pvals:
        corrected = correct_pvalues(raw_pvals, method=correction, alpha=alpha)
        adj_by_name = dict(zip(non_baseline_names, corrected["pvals_adjusted"]))
        reject_by_name = dict(zip(non_baseline_names, corrected["reject"]))
        for row in rows:
            if not row["is_baseline"]:
                row["p_value_adjusted"] = float(adj_by_name[row["model"]])
                row["significant"] = bool(reject_by_name[row["model"]])

    df = pd.DataFrame(rows)
    df = df.sort_values("mae_seconds", kind="stable").reset_index(drop=True)
    return df


def compare_all_to_markdown(df: pd.DataFrame, decimals: int = 2) -> str:
    """
    Render a `compare_all` result as a paper-ready markdown table.

    Hand-rolled (not `DataFrame.to_markdown`, which needs the optional
    `tabulate` package that is not part of this project's environment).
    """
    fmt = df.copy()
    for col in ["mae_seconds", "ci_lower", "ci_upper", "se",
                "diff_vs_baseline_seconds", "diff_vs_baseline_percent"]:
        if col in fmt:
            fmt[col] = fmt[col].map(
                lambda v: "" if pd.isna(v) else f"{v:.{decimals}f}"
            )
    for col in ["p_value", "p_value_adjusted"]:
        if col in fmt:
            fmt[col] = fmt[col].map(lambda v: "" if pd.isna(v) else f"{v:.4g}")
    display_cols = [
        "model", "mae_seconds", "ci_lower", "ci_upper",
        "diff_vs_baseline_seconds", "diff_vs_baseline_percent",
        "test_method", "p_value", "p_value_adjusted", "significant",
    ]
    display_cols = [c for c in display_cols if c in fmt.columns]

    def _cell(v) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        return str(v)

    header = display_cols
    rows = [[_cell(v) for v in row] for row in fmt[display_cols].itertuples(index=False, name=None)]
    widths = [
        max(len(header[i]), *(len(row[i]) for row in rows)) if rows else len(header[i])
        for i in range(len(header))
    ]
    lines = [
        "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(header)) + " |",
        "|" + "|".join("-" * (widths[i] + 2) for i in range(len(header))) + "|",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)) + " |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Revived error_analysis / prediction_intervals wrappers
# ---------------------------------------------------------------------------


def error_analysis_by(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_df: Optional[pd.DataFrame] = None,
    stratify_by: Optional[Union[str, Sequence[str]]] = None,
    horizon_bucket_col: Optional[str] = None,
    horizon_bins: Sequence[float] = (0, 60, 180, 360, 600, np.inf),
    horizon_labels: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Tidy wrapper around `common.metrics.error_analysis`, adding a
    stratification argument so the same call can slice by route, hour-of-day,
    or a horizon bucket (built from a raw seconds-to-arrival column).

    Args:
        y_true, y_pred: true/predicted values.
        feature_df: DataFrame aligned (by position) with y_true/y_pred,
            holding the columns to stratify by (e.g. "route_id", "hour").
        stratify_by: a column name in `feature_df`, or a list of column
            names to run separately (one output block per column). If a
            horizon bucket is desired, include `horizon_bucket_col`'s
            derived bucket column name ("horizon_bucket") here too.
        horizon_bucket_col: optional column in `feature_df` holding a raw
            horizon (e.g. seconds to arrival); if given, a "horizon_bucket"
            column is derived via `pd.cut(horizon_bins)` and can be named in
            `stratify_by`.
        horizon_bins, horizon_labels: bin edges / labels for the horizon
            bucket, forwarded to `pd.cut`.

    Returns a tidy DataFrame with an added "stratify_by" column identifying
    which grouping column produced each block of rows (or "overall" if none
    given), on top of the group index, count/mae/median_ae/std/max columns
    that `common.metrics.error_analysis` already produces.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if stratify_by is None:
        out = _error_analysis(y_true, y_pred)
        out = out.reset_index(drop=True)
        out.insert(0, "stratify_by", "overall")
        out.insert(1, "group", "overall")
        return out

    if feature_df is None:
        raise ValueError("feature_df is required when stratify_by is given")

    work_df = feature_df.reset_index(drop=True).copy()
    if horizon_bucket_col is not None:
        work_df["horizon_bucket"] = pd.cut(
            work_df[horizon_bucket_col], bins=list(horizon_bins), labels=horizon_labels
        )

    cols = [stratify_by] if isinstance(stratify_by, str) else list(stratify_by)
    blocks = []
    for col in cols:
        block = _error_analysis(y_true, y_pred, feature_df=work_df, group_by=col)
        block = block.reset_index().rename(columns={col: "group"})
        block.insert(0, "stratify_by", col)
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def prediction_intervals_by(
    y_pred: np.ndarray,
    residuals: np.ndarray,
    confidence: float = 0.95,
    feature_df: Optional[pd.DataFrame] = None,
    stratify_by: Optional[str] = None,
) -> pd.DataFrame:
    """
    Tidy wrapper around `common.metrics.prediction_intervals`.

    Without `stratify_by`, computes a single global residual-quantile
    interval and applies it to every prediction (as the underlying function
    does), returning a DataFrame with y_pred/lower/upper columns.

    With `stratify_by` (a column in `feature_df`, e.g. "route_id" or a
    precomputed "horizon_bucket"), the residual distribution — and hence the
    interval half-widths — is estimated *per group* and applied to that
    group's own predictions, which is the useful mode when error variance
    differs materially by route or by horizon.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = np.asarray(residuals, dtype=float)
    if len(y_pred) != len(residuals):
        raise ValueError("y_pred and residuals must be the same length")

    if stratify_by is None:
        lower, upper = _prediction_intervals(y_pred, residuals, confidence=confidence)
        return pd.DataFrame({
            "group": "overall",
            "y_pred": y_pred,
            "lower": lower,
            "upper": upper,
        })

    if feature_df is None:
        raise ValueError("feature_df is required when stratify_by is given")

    work_df = feature_df.reset_index(drop=True).copy()
    groups_col = work_df[stratify_by].reset_index(drop=True)

    out_lower = np.empty(len(y_pred), dtype=float)
    out_upper = np.empty(len(y_pred), dtype=float)
    for g, idx in groups_col.groupby(groups_col).groups.items():
        idx = np.asarray(idx)
        lower, upper = _prediction_intervals(
            y_pred[idx], residuals[idx], confidence=confidence
        )
        out_lower[idx] = lower
        out_upper[idx] = upper

    return pd.DataFrame({
        "group": groups_col.values,
        "y_pred": y_pred,
        "lower": out_lower,
        "upper": out_upper,
    })
