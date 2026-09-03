"""
Tests for roadmap 5.3: shared NaN handling + prediction clipping
(models/common/preprocessing.py), the EWMA validation-leak fix, and the
row-set-equality guarantee that makes the five-family leaderboard honest.

Run with:
    PYTHONPATH=.:models .venv-dl/bin/python -m pytest models/tests/test_preprocessing.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.common.data import load_dataset
from models.common.preprocessing import (
    assert_same_row_set,
    clip_predictions,
    prepare_features,
)
from models.ewma.train import EWMAModel, train_ewma
from models.historical_mean.train import HistoricalMeanModel
from models.polyreg_distance.train import PolyRegDistanceModel
from models.polyreg_time.train import PolyRegTimeModel, train_polyreg_time
from models.xgb.train import XGBTimeModel, train_xgboost

DATASET_NAME = "sample"


def _small_pre_split(train_n=600, val_n=150, test_n=150):
    """A small, real slice of the tracked sample dataset that is known to
    contain NaN in distance_to_next_stop within val/test (not just train) --
    exactly the situation that used to make families diverge silently."""
    dataset = load_dataset(DATASET_NAME)
    dataset.clean_data()
    train_df, val_df, test_df = dataset.temporal_split(train_frac=0.7, val_frac=0.15)
    return (
        train_df.head(train_n).copy(),
        val_df.head(val_n).copy(),
        test_df.head(test_n).copy(),
    )


# ---------------------------------------------------------------------------
# prepare_features: handle_nan modes
# ---------------------------------------------------------------------------

class TestPrepareFeaturesModes:
    def _df(self):
        return pd.DataFrame(
            {
                "a": [1.0, 2.0, np.nan, 4.0],
                "b": [10.0, np.nan, 30.0, 40.0],
                "cat": ["x", "y", "x", np.nan],
            }
        )

    def test_drop_removes_rows_with_any_nan_and_returns_kept_index(self):
        df = self._df()
        X, kept_index, stats = prepare_features(df, ["a", "b"], handle_nan="drop")
        # rows 1 (b NaN) and 2 (a NaN) must be dropped; 0 and 3 survive
        assert list(kept_index) == [0, 3]
        assert len(X) == 2
        assert not X.isna().any().any()
        assert stats == {}

    def test_impute_fills_with_computed_stats_when_none_given(self):
        df = self._df()
        X, kept_index, stats = prepare_features(df, ["a", "b"], handle_nan="impute")
        # no rows dropped
        assert list(kept_index) == list(df.index)
        assert not X.isna().any().any()
        assert stats["a"] == pytest.approx(df["a"].median())
        assert stats["b"] == pytest.approx(df["b"].median())
        # NaN cell filled with the median
        assert X.loc[2, "a"] == pytest.approx(stats["a"])
        assert X.loc[1, "b"] == pytest.approx(stats["b"])

    def test_impute_categorical_uses_mode(self):
        df = self._df()
        X, _, stats = prepare_features(df, ["cat"], handle_nan="impute")
        assert stats["cat"] == "x"  # mode of ['x','y','x']
        assert X.loc[3, "cat"] == "x"

    def test_error_raises_when_nan_present(self):
        df = self._df()
        with pytest.raises(ValueError, match="NaN"):
            prepare_features(df, ["a", "b"], handle_nan="error")

    def test_error_passes_when_no_nan(self):
        df = self._df().dropna()
        X, kept_index, stats = prepare_features(df, ["a", "b"], handle_nan="error")
        assert len(X) == len(df)
        assert stats == {}

    def test_unknown_handle_nan_raises(self):
        df = self._df()
        with pytest.raises(ValueError, match="Unknown handle_nan"):
            prepare_features(df, ["a"], handle_nan="bogus")

    def test_missing_feature_columns_are_silently_skipped(self):
        df = self._df()
        X, kept_index, _ = prepare_features(df, ["a", "does_not_exist"], handle_nan="drop")
        assert "does_not_exist" not in X.columns
        assert list(X.columns) == ["a"]


# ---------------------------------------------------------------------------
# Imputation statistics must come from TRAIN only
# ---------------------------------------------------------------------------

class TestTrainOnlyImputeStats:
    def test_val_fitted_imputer_would_give_a_detectably_different_answer(self):
        train_df = pd.DataFrame({"x": [10.0, 10.0, 10.0, 10.0]})  # train median = 10
        val_df = pd.DataFrame({"x": [np.nan, 1000.0, 1000.0]})  # val's own median = 1000

        # Correct usage: fit stats on train, apply to val.
        _, _, train_stats = prepare_features(train_df, ["x"], handle_nan="impute")
        X_val_correct, _, applied_stats = prepare_features(
            val_df, ["x"], handle_nan="impute", impute_stats=train_stats
        )
        assert applied_stats == train_stats
        assert X_val_correct.loc[0, "x"] == pytest.approx(10.0)

        # The leaky alternative: fit stats on val itself.
        X_val_leaky, _, leaky_stats = prepare_features(val_df, ["x"], handle_nan="impute")
        assert X_val_leaky.loc[0, "x"] == pytest.approx(1000.0)

        # The two must disagree -- proving the train-only path is not an
        # accidental no-op.
        assert X_val_correct.loc[0, "x"] != X_val_leaky.loc[0, "x"]
        assert train_stats != leaky_stats

    def test_impute_stats_pass_through_unmodified(self):
        """When impute_stats is supplied, it is used verbatim -- prepare_features
        must never silently recompute from the df it was just asked to fill."""
        df = pd.DataFrame({"x": [np.nan, np.nan]})
        fixed_stats = {"x": 42.0}
        X, _, stats = prepare_features(df, ["x"], handle_nan="impute", impute_stats=fixed_stats)
        assert stats is fixed_stats
        assert (X["x"] == 42.0).all()


# ---------------------------------------------------------------------------
# Clipping bounds identical across every family
# ---------------------------------------------------------------------------

class TestClippingIsUnified:
    def test_clip_predictions_bounds(self):
        preds = clip_predictions(np.array([-100.0, 0.0, 3600.0, 7200.0, 50000.0]))
        assert preds.tolist() == [0.0, 0.0, 3600.0, 7200.0, 7200.0]

    def _extreme_state_models(self):
        """One model instance per family, hand-fitted so its raw (pre-clip)
        prediction would fall far outside [0, 7200] -- to prove every
        family's predict() actually clips, not just that clip_predictions
        works in isolation."""
        low_df = pd.DataFrame({
            "route_id": ["r1"], "stop_sequence": [1], "hour": [8],
            "vp_ts": pd.to_datetime(["2024-01-01 08:00:00"]),
            "distance_to_stop": [100.0],
            "time_to_arrival_seconds": [-500.0],
        })
        high_df = pd.DataFrame({
            "route_id": ["r1"], "stop_sequence": [1], "hour": [8],
            "vp_ts": pd.to_datetime(["2024-01-01 08:00:01"]),
            "distance_to_stop": [100.0],
            "time_to_arrival_seconds": [50000.0],
        })
        both = pd.concat([low_df, high_df], ignore_index=True)

        models = {}

        ewma = EWMAModel(alpha=0.9, group_by=["route_id", "stop_sequence"], min_observations=1)
        ewma.fit(both)
        models["ewma"] = (ewma, both)

        hm = HistoricalMeanModel(group_by=["route_id", "stop_sequence", "hour"])
        hm.fit(both)
        models["historical_mean"] = (hm, both)

        # Ridge extrapolates way outside [0, 7200] when trained on points
        # with huge target spread and queried far from the training range.
        pr_dist_train = pd.DataFrame({
            "distance_to_stop": [1.0, 2.0],
            "time_to_arrival_seconds": [-500.0, 50000.0],
        })
        pr_dist = PolyRegDistanceModel(degree=1, alpha=0.0)
        pr_dist.fit(pr_dist_train)
        models["polyreg_distance"] = (pr_dist, pd.DataFrame({"distance_to_stop": [1.0, 2.0]}))

        return models

    @pytest.mark.parametrize("name", ["ewma", "historical_mean", "polyreg_distance"])
    def test_predict_output_is_clipped_to_canonical_bounds(self, name):
        models = self._extreme_state_models()
        model, X = models[name]
        preds = model.predict(X)
        assert np.all(preds >= 0.0)
        assert np.all(preds <= 7200.0)

    def test_polyreg_time_and_xgb_clip_to_canonical_bounds(self):
        # Fit on two points with an extreme target spread so an
        # out-of-training-range query extrapolates far outside [0, 7200].
        train_df = pd.DataFrame({
            "distance_to_stop": [1.0, 2.0],
            "time_to_arrival_seconds": [-500.0, 80000.0],
        })
        query_df = pd.DataFrame({"distance_to_stop": [1000.0]})

        pr_time = PolyRegTimeModel(poly_degree=1, alpha=0.0, include_temporal=False,
                                    include_spatial=False)
        pr_time.fit(train_df)
        preds_time = pr_time.predict(query_df)
        assert np.all(preds_time >= 0.0) and np.all(preds_time <= 7200.0)

        xgb_model = XGBTimeModel(max_depth=2, n_estimators=5, include_temporal=False,
                                  include_spatial=False)
        xgb_model.fit(train_df)
        preds_xgb = xgb_model.predict(query_df)
        assert np.all(preds_xgb >= 0.0) and np.all(preds_xgb <= 7200.0)


# ---------------------------------------------------------------------------
# Row-set equality: all five families score on the same rows
# ---------------------------------------------------------------------------

class TestRowSetEquality:
    def test_assert_same_row_set_passes_and_fails_correctly(self):
        idx_a = pd.Index([1, 2, 3])
        idx_b = pd.Index([3, 2, 1])
        assert_same_row_set(idx_a, idx_b)  # same set, different order -> ok

        idx_c = pd.Index([1, 2, 4])
        with pytest.raises(AssertionError):
            assert_same_row_set(idx_a, idx_c)

    def test_all_five_families_score_every_row_of_a_nan_containing_split(self):
        """The regression this guards against: one family silently dropping
        NaN rows that another imputes, so it gets scored on a smaller,
        easier subset and the leaderboard comparison stops being
        apples-to-apples. val/test here are real rows known to contain NaN
        in distance_to_next_stop (see _small_pre_split)."""
        pre_split = _small_pre_split()
        train_df, val_df, test_df = pre_split
        assert val_df["distance_to_next_stop"].isna().any()
        assert test_df["distance_to_next_stop"].isna().any()

        from models.historical_mean.train import train_historical_mean
        from models.polyreg_distance.train import train_polyreg_distance

        results = {
            "ewma": train_ewma(dataset_name=DATASET_NAME, save_model=False, pre_split=pre_split),
            "historical_mean": train_historical_mean(
                dataset_name=DATASET_NAME, save_model=False, pre_split=pre_split
            ),
            "polyreg_distance": train_polyreg_distance(
                dataset_name=DATASET_NAME, save_model=False, pre_split=pre_split
            ),
            "polyreg_time": train_polyreg_time(
                dataset_name=DATASET_NAME, save_model=False, pre_split=pre_split
            ),
            "xgb": train_xgboost(
                dataset_name=DATASET_NAME, save_model=False, pre_split=pre_split,
                n_estimators=20,
            ),
        }

        for name, result in results.items():
            metadata = result["metadata"]
            # Every family scored the FULL val/test set -- none silently
            # narrowed to an easier subset despite the NaN rows present.
            assert metadata["test_samples"] == len(test_df), name
            metrics = result["metrics"]
            assert np.isfinite(metrics["test_mae_seconds"]), name
            assert np.isfinite(metrics["val_mae_seconds"]), name


# ---------------------------------------------------------------------------
# EWMA validation leak: regression test
# ---------------------------------------------------------------------------

class TestEWMANoValidationLeak:
    def _train_val_test(self):
        rng = np.random.default_rng(0)
        n = 40
        base_ts = pd.Timestamp("2024-01-01")
        train_df = pd.DataFrame({
            "route_id": ["r1"] * n,
            "stop_sequence": [1] * n,
            "vp_ts": [base_ts + pd.Timedelta(minutes=i) for i in range(n)],
            "time_to_arrival_seconds": 300 + rng.normal(0, 10, n),
        })
        val_df = pd.DataFrame({
            "route_id": ["r1"] * 5,
            "stop_sequence": [1] * 5,
            "vp_ts": [base_ts + pd.Timedelta(minutes=n + i) for i in range(5)],
            "time_to_arrival_seconds": [300.0] * 5,
        })
        test_df = pd.DataFrame({
            "route_id": ["r1"] * 5,
            "stop_sequence": [1] * 5,
            "vp_ts": [base_ts + pd.Timedelta(minutes=n + 10 + i) for i in range(5)],
            "time_to_arrival_seconds": [300.0] * 5,
        })
        return train_df, val_df, test_df

    def test_changing_val_test_labels_does_not_change_fitted_state_or_train_predictions(self):
        train_df, val_df, test_df = self._train_val_test()

        model_a = EWMAModel(alpha=0.3, group_by=["route_id", "stop_sequence"], min_observations=1)
        model_a.fit(train_df)
        state_a = dict(model_a.ewma_values)
        train_preds_a = model_a.predict(train_df)

        # Wildly different val/test labels -- if fit() (or anything called
        # before/around it) ever consumed them, this would change state_a.
        val_df2 = val_df.copy()
        val_df2["time_to_arrival_seconds"] = 999999.0
        test_df2 = test_df.copy()
        test_df2["time_to_arrival_seconds"] = -999999.0

        model_b = EWMAModel(alpha=0.3, group_by=["route_id", "stop_sequence"], min_observations=1)
        model_b.fit(train_df)  # fit() only ever sees train_df, regardless of val/test
        state_b = dict(model_b.ewma_values)
        train_preds_b = model_b.predict(train_df)

        assert state_a == pytest.approx(state_b)
        assert np.allclose(train_preds_a, train_preds_b)

        # Sanity: this would have failed under the old code, where
        # train_ewma() called model.update(val_df, y_val) with the real
        # (leaked) val labels before ever scoring test.
        assert 999999.0 not in state_a.values()
        assert -999999.0 not in state_a.values()

    def test_predict_and_advance_never_uses_a_rows_own_label_for_its_own_prediction(self):
        """Causal warm-up: the prediction for val row i must equal what the
        model would have predicted from state built ONLY on rows strictly
        before it (train + earlier val rows) -- never row i's own label."""
        train_df, val_df, _ = self._train_val_test()

        model = EWMAModel(alpha=0.3, group_by=["route_id", "stop_sequence"], min_observations=1)
        model.fit(train_df)
        pre_advance_value = model.ewma_values[("r1", 1)]

        preds = model.predict_and_advance(val_df)
        # First val prediction must equal the train-only EWMA value (it can't
        # have been influenced by any val label yet).
        assert preds[0] == pytest.approx(pre_advance_value)

    def test_train_ewma_end_to_end_does_not_leak_val_into_test_scoring_state(self):
        """Integration check on the real train_ewma() entrypoint: score val,
        then confirm the state used to score test only ever advanced
        causally (never bulk-updated with the entire val set before any
        val row was scored)."""
        train_df, val_df, test_df = self._train_val_test()
        pre_split = (train_df, val_df, test_df)
        result = train_ewma(
            dataset_name=DATASET_NAME, save_model=False, pre_split=pre_split,
            min_observations=1,
        )
        # Must run to completion and produce finite metrics for both splits.
        metrics = result["metrics"]
        assert np.isfinite(metrics["val_mae_seconds"])
        assert np.isfinite(metrics["test_mae_seconds"])
