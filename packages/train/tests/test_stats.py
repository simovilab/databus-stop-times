"""
Tests for models/evaluation/stats.py (roadmap 4.3): bootstrap CIs, paired
model-vs-model significance tests, multiple-comparison correction, and the
leaderboard-level compare_all entry point.

All synthetic and fast -- deterministic under fixed seeds, no real datasets
loaded here (see scratchpad/ for the real-data demonstration on
datasets/mbta_segments_corpus.parquet).

Run with:
    PYTHONPATH=.:models .venv-dl/bin/python -m pytest models/tests/test_stats.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.evaluation.stats import (
    BootstrapResult,
    PairedTestResult,
    bootstrap_metric_ci,
    compare_all,
    compare_all_to_markdown,
    correct_pvalues,
    error_analysis_by,
    paired_test,
    prediction_intervals_by,
)
from models.common.metrics import mae_seconds


# --------------------------------------------------------------------------
# bootstrap_metric_ci
# --------------------------------------------------------------------------

def test_bootstrap_ci_covers_true_mae_iid():
    # Known distribution: errors ~ Laplace-ish via abs(normal), true MAE of
    # y_pred = y_true + noise is E|noise|. With noise ~ N(0, sigma^2),
    # E|noise| = sigma * sqrt(2/pi). Use a large n and check the CI covers
    # this true value at a fixed seed (deterministic, not a Monte Carlo
    # coverage sweep).
    rng = np.random.default_rng(123)
    n = 5000
    sigma = 20.0
    y_true = rng.normal(300, 50, size=n)
    noise = rng.normal(0, sigma, size=n)
    y_pred = y_true + noise
    true_mae = sigma * np.sqrt(2 / np.pi)

    result = bootstrap_metric_ci(y_true, y_pred, n_boot=2000, seed=42)
    assert isinstance(result, BootstrapResult)
    assert result.ci_lower < true_mae < result.ci_upper
    assert result.ci_lower < result.estimate < result.ci_upper
    assert result.se > 0
    assert not result.clustered


def test_bootstrap_cluster_ci_wider_than_iid_under_correlation():
    # Construct strongly within-cluster-correlated errors: each of 200
    # "trips" gets one shared random offset applied to all its rows'
    # errors, plus small row-level noise. The i.i.d. bootstrap (which
    # ignores the cluster structure) should understate the true sampling
    # variance relative to the cluster bootstrap that resamples whole
    # trips.
    rng = np.random.default_rng(7)
    n_clusters = 200
    rows_per_cluster = 30
    n = n_clusters * rows_per_cluster

    cluster_ids = np.repeat(np.arange(n_clusters), rows_per_cluster)
    cluster_offset = rng.normal(0, 40, size=n_clusters)[cluster_ids]
    row_noise = rng.normal(0, 3, size=n)

    y_true = rng.normal(300, 50, size=n)
    y_pred = y_true + cluster_offset + row_noise

    iid_result = bootstrap_metric_ci(y_true, y_pred, n_boot=2000, seed=42)
    cluster_result = bootstrap_metric_ci(
        y_true, y_pred, n_boot=2000, seed=42, groups=cluster_ids
    )

    iid_width = iid_result.ci_upper - iid_result.ci_lower
    cluster_width = cluster_result.ci_upper - cluster_result.ci_lower
    assert cluster_result.clustered
    assert cluster_result.n_clusters == n_clusters
    assert cluster_width > iid_width
    # Sanity: not a marginal difference -- correlation is strong by
    # construction, so the effect should be substantial.
    assert cluster_width > 1.5 * iid_width


def test_bootstrap_ci_deterministic_under_fixed_seed():
    rng = np.random.default_rng(1)
    y_true = rng.normal(300, 50, size=500)
    y_pred = y_true + rng.normal(0, 15, size=500)

    r1 = bootstrap_metric_ci(y_true, y_pred, n_boot=500, seed=99)
    r2 = bootstrap_metric_ci(y_true, y_pred, n_boot=500, seed=99)
    assert r1.ci_lower == r2.ci_lower
    assert r1.ci_upper == r2.ci_upper
    assert r1.se == r2.se


def test_bootstrap_ci_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        bootstrap_metric_ci(np.arange(10), np.arange(9))


def test_bootstrap_block_true_requires_groups():
    with pytest.raises(ValueError):
        bootstrap_metric_ci(np.arange(10), np.arange(10), block=True)


# --------------------------------------------------------------------------
# paired_test
# --------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["wilcoxon", "dm", "bootstrap"])
def test_paired_test_identical_predictions_gives_null_result(method):
    rng = np.random.default_rng(5)
    y_true = rng.normal(300, 50, size=1000)
    pred = y_true + rng.normal(0, 20, size=1000)

    result = paired_test(y_true, pred, pred, method=method, n_boot=2000, seed=1)
    assert isinstance(result, PairedTestResult)
    assert result.mean_diff == pytest.approx(0.0, abs=1e-9)
    assert result.diff_seconds == pytest.approx(0.0, abs=1e-9)
    assert result.p_value > 0.9


@pytest.mark.parametrize("method", ["wilcoxon", "dm", "bootstrap"])
def test_paired_test_large_difference_is_significant(method):
    rng = np.random.default_rng(6)
    n = 2000
    y_true = rng.normal(300, 50, size=n)
    # Model A: small noise. Model B: much larger noise -> clearly worse.
    pred_a = y_true + rng.normal(0, 10, size=n)
    pred_b = y_true + rng.normal(0, 60, size=n)

    result = paired_test(y_true, pred_a, pred_b, method=method, n_boot=3000, seed=2)
    assert result.p_value < 0.001
    assert result.mean_diff < 0  # A has lower error than B
    assert result.diff_seconds < 0
    assert result.mae_a < result.mae_b


def test_paired_test_unknown_method_raises():
    y = np.arange(10, dtype=float)
    with pytest.raises(ValueError):
        paired_test(y, y, y, method="not-a-method")


def test_paired_test_dm_ci_present_bootstrap_ci_present_wilcoxon_ci_none():
    rng = np.random.default_rng(9)
    y_true = rng.normal(300, 50, size=300)
    pred_a = y_true + rng.normal(0, 15, size=300)
    pred_b = y_true + rng.normal(0, 25, size=300)

    r_wilcoxon = paired_test(y_true, pred_a, pred_b, method="wilcoxon")
    r_dm = paired_test(y_true, pred_a, pred_b, method="dm")
    r_boot = paired_test(y_true, pred_a, pred_b, method="bootstrap", n_boot=1000)

    assert r_wilcoxon.ci_lower is None and r_wilcoxon.ci_upper is None
    assert r_dm.ci_lower is not None and r_dm.ci_upper is not None
    assert r_boot.ci_lower is not None and r_boot.ci_upper is not None


def test_paired_test_bootstrap_cluster_aware():
    rng = np.random.default_rng(11)
    n_clusters = 100
    rows_per_cluster = 10
    n = n_clusters * rows_per_cluster
    cluster_ids = np.repeat(np.arange(n_clusters), rows_per_cluster)

    y_true = rng.normal(300, 50, size=n)
    pred_a = y_true + rng.normal(0, 15, size=n)
    pred_b = y_true + rng.normal(0, 15, size=n) + 5  # slightly worse, on avg

    r_iid = paired_test(y_true, pred_a, pred_b, method="bootstrap", n_boot=1500, seed=3)
    r_cluster = paired_test(
        y_true, pred_a, pred_b, method="bootstrap", n_boot=1500, seed=3, groups=cluster_ids
    )
    assert r_iid.ci_lower is not None and r_cluster.ci_lower is not None
    # Just check both run and produce a CI; widths need not compare here
    # since correlation isn't specifically injected on the differential.
    assert r_cluster.n == n


# --------------------------------------------------------------------------
# correct_pvalues
# --------------------------------------------------------------------------

def test_holm_matches_hand_computed_values():
    # p = [0.01, 0.02, 0.03, 0.04], m=4
    # sorted: 0.01, 0.02, 0.03, 0.04 ; multipliers (m-j): 4,3,2,1
    # raw: 0.04, 0.06, 0.06, 0.04 ; cummax: 0.04, 0.06, 0.06, 0.06
    pvals = [0.01, 0.02, 0.03, 0.04]
    out = correct_pvalues(pvals, method="holm")
    expected = np.array([0.04, 0.06, 0.06, 0.06])
    np.testing.assert_allclose(out["pvals_adjusted"], expected, atol=1e-12)


def test_bh_matches_hand_computed_values():
    # p = [0.01, 0.02, 0.03, 0.04], m=4
    # sorted ranks 1..4: raw = p*m/rank = 0.04, 0.04, 0.04, 0.04
    # reverse cummin -> all 0.04
    pvals = [0.01, 0.02, 0.03, 0.04]
    out = correct_pvalues(pvals, method="bh")
    expected = np.array([0.04, 0.04, 0.04, 0.04])
    np.testing.assert_allclose(out["pvals_adjusted"], expected, atol=1e-12)


def test_bh_matches_hand_computed_values_uneven():
    # p = [0.005, 0.01, 0.03, 0.20], m=4
    # raw = p*4/rank: 0.02, 0.02, 0.04, 0.20
    # reverse cummin: min(0.20)=0.20 -> min(0.04,0.20)=0.04 -> min(0.02,0.04)=0.02 -> min(0.02,0.02)=0.02
    pvals = [0.005, 0.01, 0.03, 0.20]
    out = correct_pvalues(pvals, method="bh")
    expected = np.array([0.02, 0.02, 0.04, 0.20])
    np.testing.assert_allclose(out["pvals_adjusted"], expected, atol=1e-12)


def test_holm_more_conservative_than_bh():
    rng = np.random.default_rng(0)
    pvals = rng.uniform(0, 0.2, size=8)
    holm = correct_pvalues(pvals, method="holm")["pvals_adjusted"]
    bh = correct_pvalues(pvals, method="bh")["pvals_adjusted"]
    assert np.all(holm >= bh - 1e-12)
    assert holm.sum() > bh.sum()


def test_correct_pvalues_reject_mask_matches_alpha():
    pvals = [0.001, 0.2, 0.5]
    out = correct_pvalues(pvals, method="holm", alpha=0.05)
    assert out["reject"][0] == (out["pvals_adjusted"][0] <= 0.05)
    assert not out["reject"][1]
    assert not out["reject"][2]


def test_correct_pvalues_empty():
    out = correct_pvalues([], method="holm")
    assert len(out["pvals_adjusted"]) == 0
    assert len(out["reject"]) == 0


def test_correct_pvalues_unknown_method_raises():
    with pytest.raises(ValueError):
        correct_pvalues([0.1, 0.2], method="bogus")


# --------------------------------------------------------------------------
# compare_all
# --------------------------------------------------------------------------

def _synthetic_preds(seed=42, n=2000):
    rng = np.random.default_rng(seed)
    y_true = rng.normal(300, 50, size=n)
    preds = {
        "good_model": y_true + rng.normal(0, 10, size=n),
        "baseline": y_true + rng.normal(0, 30, size=n),
        "bad_model": y_true + rng.normal(0, 50, size=n),
    }
    return y_true, preds


def test_compare_all_shape_and_columns():
    y_true, preds = _synthetic_preds()
    df = compare_all(y_true, preds, baseline="baseline", n_boot=1000, seed=1)
    assert len(df) == 3
    expected_cols = {
        "model", "mae_seconds", "ci_lower", "ci_upper", "se", "n",
        "diff_vs_baseline_seconds", "diff_vs_baseline_percent",
        "test_method", "test_statistic", "p_value", "p_value_adjusted",
        "significant", "is_baseline",
    }
    assert expected_cols.issubset(set(df.columns))


def test_compare_all_orders_by_mae_ascending():
    y_true, preds = _synthetic_preds()
    df = compare_all(y_true, preds, baseline="baseline", n_boot=1000, seed=1)
    maes = df["mae_seconds"].tolist()
    assert maes == sorted(maes)
    assert df.iloc[0]["model"] == "good_model"
    assert df.iloc[-1]["model"] == "bad_model"


def test_compare_all_baseline_row_has_null_test_fields():
    y_true, preds = _synthetic_preds()
    df = compare_all(y_true, preds, baseline="baseline", n_boot=1000, seed=1)
    base_row = df[df["model"] == "baseline"].iloc[0]
    assert base_row["is_baseline"]
    assert base_row["diff_vs_baseline_seconds"] == 0.0
    assert pd.isna(base_row["p_value"])
    assert base_row["significant"] == False  # noqa: E712


def test_compare_all_significant_for_clear_difference():
    y_true, preds = _synthetic_preds()
    df = compare_all(y_true, preds, baseline="baseline", n_boot=2000, seed=1)
    bad_row = df[df["model"] == "bad_model"].iloc[0]
    assert bad_row["p_value"] < 0.01
    assert bad_row["significant"] == True  # noqa: E712


def test_compare_all_deterministic_under_fixed_seed():
    y_true, preds = _synthetic_preds()
    df1 = compare_all(y_true, preds, baseline="baseline", n_boot=500, seed=7)
    df2 = compare_all(y_true, preds, baseline="baseline", n_boot=500, seed=7)
    pd.testing.assert_frame_equal(
        df1.drop(columns=["test_method"]), df2.drop(columns=["test_method"])
    )


def test_compare_all_unknown_baseline_raises():
    y_true, preds = _synthetic_preds()
    with pytest.raises(ValueError):
        compare_all(y_true, preds, baseline="nope")


def test_compare_all_to_markdown_runs():
    y_true, preds = _synthetic_preds()
    df = compare_all(y_true, preds, baseline="baseline", n_boot=500, seed=1)
    md = compare_all_to_markdown(df)
    assert isinstance(md, str)
    assert "good_model" in md
    assert "baseline" in md


def test_compare_all_with_groups_cluster_bootstrap():
    rng = np.random.default_rng(13)
    n_clusters = 100
    rows_per_cluster = 20
    n = n_clusters * rows_per_cluster
    cluster_ids = np.repeat(np.arange(n_clusters), rows_per_cluster)
    y_true = rng.normal(300, 50, size=n)
    preds = {
        "a": y_true + rng.normal(0, 15, size=n),
        "b": y_true + rng.normal(0, 40, size=n),
    }
    df = compare_all(y_true, preds, baseline="b", groups=cluster_ids, n_boot=500, seed=1)
    assert len(df) == 2
    assert df["n"].iloc[0] == n


# --------------------------------------------------------------------------
# error_analysis_by / prediction_intervals_by (revived from common.metrics)
# --------------------------------------------------------------------------

def test_error_analysis_by_overall():
    rng = np.random.default_rng(20)
    y_true = rng.normal(300, 50, size=200)
    y_pred = y_true + rng.normal(0, 15, size=200)
    out = error_analysis_by(y_true, y_pred)
    assert len(out) == 1
    assert out.iloc[0]["stratify_by"] == "overall"
    assert out.iloc[0]["count"] == 200


def test_error_analysis_by_stratify_single_column():
    rng = np.random.default_rng(21)
    n = 300
    y_true = rng.normal(300, 50, size=n)
    y_pred = y_true + rng.normal(0, 15, size=n)
    feature_df = pd.DataFrame({"route_id": rng.choice(["1", "2", "3"], size=n)})
    out = error_analysis_by(y_true, y_pred, feature_df=feature_df, stratify_by="route_id")
    assert set(out["stratify_by"]) == {"route_id"}
    assert set(out["group"]) == {"1", "2", "3"}
    assert out["count"].sum() == n


def test_error_analysis_by_multiple_stratifications_and_horizon_bucket():
    rng = np.random.default_rng(22)
    n = 400
    y_true = rng.normal(300, 50, size=n)
    y_pred = y_true + rng.normal(0, 15, size=n)
    feature_df = pd.DataFrame({
        "route_id": rng.choice(["1", "2"], size=n),
        "raw_horizon": rng.uniform(0, 700, size=n),
    })
    out = error_analysis_by(
        y_true, y_pred, feature_df=feature_df,
        stratify_by=["route_id", "horizon_bucket"],
        horizon_bucket_col="raw_horizon",
    )
    assert set(out["stratify_by"]) == {"route_id", "horizon_bucket"}
    # every row accounted for within each stratification block
    for name, block in out.groupby("stratify_by"):
        assert block["count"].sum() == n


def test_error_analysis_by_requires_feature_df_when_stratifying():
    y_true = np.arange(10, dtype=float)
    y_pred = y_true + 1
    with pytest.raises(ValueError):
        error_analysis_by(y_true, y_pred, stratify_by="route_id")


def test_prediction_intervals_by_overall():
    rng = np.random.default_rng(23)
    residuals = rng.normal(0, 20, size=1000)
    y_pred = rng.normal(300, 50, size=1000)
    out = prediction_intervals_by(y_pred, residuals, confidence=0.9)
    assert len(out) == 1000
    assert (out["upper"] >= out["lower"]).all()
    assert (out["group"] == "overall").all()


def test_prediction_intervals_by_stratified_differs_per_group():
    rng = np.random.default_rng(24)
    n = 2000
    group = np.repeat(["tight", "wide"], n // 2)
    residuals = np.where(group == "tight", rng.normal(0, 5, n), rng.normal(0, 50, n))
    y_pred = rng.normal(300, 50, size=n)
    feature_df = pd.DataFrame({"g": group})

    out = prediction_intervals_by(
        y_pred, residuals, confidence=0.9, feature_df=feature_df, stratify_by="g"
    )
    widths = (out["upper"] - out["lower"])
    tight_width = widths[out["group"] == "tight"].iloc[0]
    wide_width = widths[out["group"] == "wide"].iloc[0]
    assert wide_width > tight_width


def test_prediction_intervals_by_requires_feature_df_when_stratifying():
    y_pred = np.arange(10, dtype=float)
    residuals = np.zeros(10)
    with pytest.raises(ValueError):
        prediction_intervals_by(y_pred, residuals, stratify_by="g")


# --------------------------------------------------------------------------
# consistency with common.metrics.mae_seconds
# --------------------------------------------------------------------------

def test_bootstrap_point_estimate_matches_plain_mae():
    rng = np.random.default_rng(30)
    y_true = rng.normal(300, 50, size=500)
    y_pred = y_true + rng.normal(0, 15, size=500)
    result = bootstrap_metric_ci(y_true, y_pred, n_boot=200, seed=1)
    assert result.estimate == pytest.approx(mae_seconds(y_true, y_pred))
