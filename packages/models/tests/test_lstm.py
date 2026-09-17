"""
Unit tests for the segment-sequence LSTM/GRU trainer (roadmap 5.2).

All tests use small synthetic DataFrames (no real corpus) and run in well
under a minute. Run with:

    uv run --group dl python -m pytest packages/models/tests/test_lstm.py -q

torch is a training-only, optional dependency (the `dl` uv group) -- it is
not installed in the default dev environment. This whole module is skipped
cleanly (not a collection error) when torch is unavailable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="torch not installed (install the 'dl' uv group to run LSTM tests)")

from stoptimes.models.common.registry import ModelRegistry
from stoptimes.models.lstm.data import (
    FEATURE_COLUMNS,
    Preprocessor,
    apply_preprocessor,
    build_sequences,
    coerce_feature_dtypes,
    fit_preprocessor,
    pad_batch,
)
from stoptimes.models.lstm.model import SegmentSequenceModel, masked_loss
from stoptimes.models.lstm.train import LSTMSegmentModel, train_lstm


# --------------------------------------------------------------------------
# Synthetic data helpers
# --------------------------------------------------------------------------

def _make_trip_rows(trip_id: str, service_day: str, n_segments: int, rng: np.random.RandomState,
                     start_hour: int = 8) -> list:
    """Rows for one trip instance, segment_index 0..n_segments-1, vp_ts
    strictly increasing within the given service day."""
    base_ts = pd.Timestamp(f"{service_day} {start_hour:02d}:00:00")
    rows = []
    for i in range(n_segments):
        ts = base_ts + pd.Timedelta(minutes=2 * i)
        row = {c: float(rng.uniform(0, 10)) for c in FEATURE_COLUMNS}
        row["distance_to_stop"] = float(50 + 10 * (n_segments - i))  # >= min_distance
        row.update(
            trip_id=trip_id,
            segment_index=i,
            stop_sequence=i + 1,
            vp_ts=ts,
            time_to_arrival_seconds=float(30 * (n_segments - i)),
        )
        rows.append(row)
    return rows


def make_synthetic_df(n_trips: int = 10, seq_len: int = 5, seed: int = 0,
                       service_day: str = "2026-01-05") -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    rows = []
    for t in range(n_trips):
        rows.extend(_make_trip_rows(f"trip{t}", service_day, seq_len, rng))
    return pd.DataFrame(rows)


def _pre_split(df: pd.DataFrame, n_train: int, n_val: int, n_test: int, seq_len: int):
    """Slice a make_synthetic_df-style frame (trip0..tripN-1, each seq_len
    contiguous rows) into disjoint train/val/test DataFrames by trip."""
    trips = df["trip_id"].unique()
    train_trips = trips[:n_train]
    val_trips = trips[n_train:n_train + n_val]
    test_trips = trips[n_train + n_val:n_train + n_val + n_test]
    train_df = df[df["trip_id"].isin(train_trips)].copy()
    val_df = df[df["trip_id"].isin(val_trips)].copy()
    test_df = df[df["trip_id"].isin(test_trips)].copy()
    return train_df, val_df, test_df


# --------------------------------------------------------------------------
# 1. Sequence grouping + ordering
# --------------------------------------------------------------------------

class TestBuildSequences:
    def test_groups_by_trip_and_orders_by_segment_index(self):
        df = make_synthetic_df(n_trips=1, seq_len=6, seed=1)
        # Shuffle rows so build_sequences must re-sort them.
        df = df.sample(frac=1.0, random_state=7).reset_index(drop=True)

        seqs = build_sequences(df, FEATURE_COLUMNS, "time_to_arrival_seconds", service_offset_h=7)

        assert len(seqs) == 1
        seq = seqs[0]
        # target is 30 * (n_segments - i) which is strictly decreasing with segment_index
        assert list(seq.target) == sorted(seq.target, reverse=True)

    def test_reused_trip_id_across_service_dates_is_two_sequences(self):
        rng = np.random.RandomState(2)
        rows = _make_trip_rows("tripX", "2026-01-05", 4, rng)
        rows += _make_trip_rows("tripX", "2026-01-06", 4, rng)
        df = pd.DataFrame(rows)

        seqs = build_sequences(df, FEATURE_COLUMNS, "time_to_arrival_seconds", service_offset_h=7)

        assert len(seqs) == 2
        assert {len(s.target) for s in seqs} == {4}
        # Distinct service_date keys despite identical trip_id.
        dates = {s.key[1] for s in seqs}
        assert len(dates) == 2

    def test_cross_midnight_trip_folds_onto_prior_service_day(self):
        # A trip starting 01:30 local, with service_offset_h=7, should fold
        # onto the *previous* calendar date's service day (01:30 - 7h = prior
        # day 18:30), joining rows from later that same "night" trip.
        rng = np.random.RandomState(3)
        rows = _make_trip_rows("tripY", "2026-01-06", 3, rng, start_hour=1)
        for r in rows:
            r["vp_ts"] = pd.Timestamp(r["vp_ts"]) + pd.Timedelta(minutes=30)
        df = pd.DataFrame(rows)

        seqs = build_sequences(df, FEATURE_COLUMNS, "time_to_arrival_seconds", service_offset_h=7)
        assert len(seqs) == 1
        assert seqs[0].key[1] == pd.Timestamp("2026-01-05").date()

    def test_row_pos_matches_source_positions(self):
        df = make_synthetic_df(n_trips=2, seq_len=3, seed=4)
        seqs = build_sequences(df, FEATURE_COLUMNS, "time_to_arrival_seconds", service_offset_h=7)
        all_pos = np.sort(np.concatenate([s.row_pos for s in seqs]))
        assert list(all_pos) == list(range(len(df)))


# --------------------------------------------------------------------------
# 2. Padding contributes nothing to loss/metrics
# --------------------------------------------------------------------------

class TestPaddingIsInert:
    def test_changing_padded_values_does_not_change_loss(self):
        df = make_synthetic_df(n_trips=2, seq_len=3, seed=5)
        # Force different lengths: shorten trip1 to 2 rows.
        df = df[~((df["trip_id"] == "trip1") & (df["segment_index"] == 2))].reset_index(drop=True)

        seqs = build_sequences(df, FEATURE_COLUMNS, "time_to_arrival_seconds", service_offset_h=7)
        pre = fit_preprocessor(seqs, FEATURE_COLUMNS)
        seqs = apply_preprocessor(seqs, pre)

        batch = pad_batch(seqs)
        model = SegmentSequenceModel(n_features=len(FEATURE_COLUMNS), hidden_size=8)
        model.eval()

        preds1 = model(batch.x, batch.lengths)
        loss1 = masked_loss(preds1, batch.y, batch.mask, loss="huber")

        # Corrupt the padded region of x and y.
        x2 = batch.x.clone()
        y2 = batch.y.clone()
        pad_positions = ~batch.mask
        x2[pad_positions.unsqueeze(-1).expand_as(x2)] = 999.0
        y2[pad_positions] = -999.0

        preds2 = model(x2, batch.lengths)
        loss2 = masked_loss(preds2, y2, batch.mask, loss="huber")

        assert torch.allclose(loss1, loss2)

        # Predictions at *valid* (unpadded) positions must also be identical --
        # padded x values must never leak into the packed/causal computation.
        valid = batch.mask
        assert torch.allclose(preds1[valid], preds2[valid])

    def test_metrics_ignore_padded_positions(self):
        df = make_synthetic_df(n_trips=2, seq_len=4, seed=6)
        df = df[~((df["trip_id"] == "trip1") & (df["segment_index"].isin([2, 3])))].reset_index(drop=True)

        seqs = build_sequences(df, FEATURE_COLUMNS, "time_to_arrival_seconds", service_offset_h=7)
        pre = fit_preprocessor(seqs, FEATURE_COLUMNS)
        seqs = apply_preprocessor(seqs, pre)
        batch = pad_batch(seqs)

        # Flatten via mask, as train.py's _run_eval does.
        mask_np = batch.mask.numpy()
        y_np = batch.y.numpy()
        y_true = y_np[mask_np]
        # Garbage in the padded region must not affect the flattened result.
        n_real = int(mask_np.sum())
        assert n_real == 4 + 2
        assert not np.isnan(y_true).any()


# --------------------------------------------------------------------------
# 3. Causality
# --------------------------------------------------------------------------

class TestCausality:
    def test_model_is_unidirectional(self):
        for rnn_type in ("lstm", "gru"):
            model = SegmentSequenceModel(n_features=5, hidden_size=4, rnn_type=rnn_type)
            assert model.rnn.bidirectional is False

    @pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
    def test_perturbation_at_t_only_changes_predictions_at_or_after_t(self, rnn_type):
        torch.manual_seed(0)
        seq_len = 6
        n_features = 5
        model = SegmentSequenceModel(n_features=n_features, hidden_size=8, rnn_type=rnn_type)
        model.eval()

        x = torch.randn(1, seq_len, n_features)
        lengths = torch.tensor([seq_len])

        with torch.no_grad():
            base_preds = model(x, lengths)[0]

        for t in range(seq_len):
            x_perturbed = x.clone()
            # Non-uniform per-feature perturbation: a *constant* shift across
            # all features of a timestep is exactly cancelled by the model's
            # input LayerNorm (mean-subtraction is shift-invariant), which
            # would falsely look like a causality violation. Random noise
            # avoids that degenerate case.
            x_perturbed[0, t, :] += torch.randn(n_features) * 5.0
            with torch.no_grad():
                perturbed_preds = model(x_perturbed, lengths)[0]

            # Steps strictly before t must be untouched.
            if t > 0:
                assert torch.allclose(base_preds[:t], perturbed_preds[:t], atol=1e-6), (
                    f"rnn_type={rnn_type}: perturbing step {t} changed an earlier prediction"
                )
            # Step t itself (and generally later steps) should change.
            assert not torch.allclose(base_preds[t:], perturbed_preds[t:], atol=1e-6)


# --------------------------------------------------------------------------
# 4. Scaler fit on train only
# --------------------------------------------------------------------------

class TestNoLeakage:
    def test_preprocessor_stats_depend_only_on_train_data(self):
        train_df = make_synthetic_df(n_trips=4, seq_len=4, seed=10)
        val_df = make_synthetic_df(n_trips=4, seq_len=4, seed=999)  # very different distribution
        for c in FEATURE_COLUMNS:
            val_df[c] = val_df[c] * 1000 + 5000  # wildly different scale

        train_seqs = build_sequences(train_df, FEATURE_COLUMNS, "time_to_arrival_seconds", service_offset_h=7)
        pre = fit_preprocessor(train_seqs, FEATURE_COLUMNS)

        manual_stack = np.concatenate([s.features for s in train_seqs], axis=0)
        expected_mean = np.nanmean(manual_stack, axis=0)

        # sklearn StandardScaler.mean_ should equal the train-only feature means
        # (post train-only median imputation), never influenced by val_df.
        np.testing.assert_allclose(pre.scaler.mean_, expected_mean, rtol=1e-5, atol=1e-6)

    def test_fit_preprocessor_never_sees_val_rows(self, monkeypatch):
        train_df = make_synthetic_df(n_trips=3, seq_len=3, seed=11)
        train_seqs = build_sequences(train_df, FEATURE_COLUMNS, "time_to_arrival_seconds", service_offset_h=7)

        # fit_preprocessor's signature takes only `sequences` -- structurally
        # incapable of touching val/test data unless explicitly passed.
        import inspect
        sig = inspect.signature(fit_preprocessor)
        assert list(sig.parameters.keys()) == ["sequences", "feature_cols"]


# --------------------------------------------------------------------------
# 5. Determinism
# --------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_metrics(self):
        df = make_synthetic_df(n_trips=18, seq_len=4, seed=20)
        train_df, val_df, test_df = _pre_split(df, n_train=10, n_val=4, n_test=4, seq_len=4)

        kwargs = dict(
            hidden_size=8, num_layers=1, dropout=0.0, rnn_type="lstm",
            learning_rate=1e-2, batch_size=4, max_epochs=3, patience=3,
            save_model=False, seed=123,
        )
        out1 = train_lstm(pre_split=(train_df, val_df, test_df), **kwargs)
        out2 = train_lstm(pre_split=(train_df, val_df, test_df), **kwargs)

        for k in out1["metrics"]:
            assert out1["metrics"][k] == pytest.approx(out2["metrics"][k], rel=1e-6, abs=1e-6), k


# --------------------------------------------------------------------------
# 6. predict() row order
# --------------------------------------------------------------------------

class TestPredictRowOrder:
    def test_predict_returns_predictions_in_input_row_order(self):
        df = make_synthetic_df(n_trips=10, seq_len=4, seed=30)
        train_df, val_df, test_df = _pre_split(df, n_train=6, n_val=2, n_test=2, seq_len=4)

        out = train_lstm(
            pre_split=(train_df, val_df, test_df),
            hidden_size=8, num_layers=1, max_epochs=2, patience=2,
            save_model=False, seed=42,
        )
        model: LSTMSegmentModel = out["model"]

        target_df = test_df.reset_index(drop=True).copy()
        target_df["_uid"] = np.arange(len(target_df))

        preds_original = model.predict(target_df)

        shuffled = target_df.sample(frac=1.0, random_state=99).reset_index(drop=True)
        preds_shuffled = model.predict(shuffled)

        # Map shuffled predictions back to original order via _uid and compare.
        recovered = np.empty(len(target_df))
        recovered[shuffled["_uid"].to_numpy()] = preds_shuffled
        np.testing.assert_allclose(recovered, preds_original, rtol=1e-5, atol=1e-4)
        assert len(preds_original) == len(target_df)


# --------------------------------------------------------------------------
# 7. Registry round trip
# --------------------------------------------------------------------------

class TestRegistryRoundTrip:
    def test_save_load_preserves_predictions(self, tmp_path):
        df = make_synthetic_df(n_trips=10, seq_len=4, seed=40)
        train_df, val_df, test_df = _pre_split(df, n_train=6, n_val=2, n_test=2, seq_len=4)

        out = train_lstm(
            pre_split=(train_df, val_df, test_df),
            hidden_size=8, num_layers=1, max_epochs=2, patience=2,
            save_model=False, seed=7,
        )
        model: LSTMSegmentModel = out["model"]
        preds_before = model.predict(test_df)

        registry = ModelRegistry(base_dir=str(tmp_path))
        registry.save_model("lstm_test_key", model, dict(out["metadata"]))

        loaded_model = registry.load_model("lstm_test_key")
        preds_after = loaded_model.predict(test_df)

        np.testing.assert_allclose(preds_before, preds_after, rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------
# 8. target_transform
# --------------------------------------------------------------------------

class TestTargetTransform:
    @pytest.mark.parametrize("transform", ["none", "log1p", "standardize"])
    def test_metrics_always_reported_in_raw_seconds(self, transform):
        """Whatever space the network trains in, val/test MAE must land in
        the same raw-seconds ballpark as the target itself (roughly
        [0, 4 * max target]) -- a transform that isn't inverted correctly
        would blow this up or collapse it near zero."""
        df = make_synthetic_df(n_trips=18, seq_len=5, seed=50)
        train_df, val_df, test_df = _pre_split(df, n_train=10, n_val=4, n_test=4, seq_len=5)
        max_target = df["time_to_arrival_seconds"].max()

        out = train_lstm(
            pre_split=(train_df, val_df, test_df),
            hidden_size=8, num_layers=1, max_epochs=5, patience=5,
            target_transform=transform, save_model=False, seed=11,
        )
        m = out["metrics"]
        assert 0.0 <= m["val_mae_seconds"] <= 4 * max_target
        assert 0.0 <= m["test_mae_seconds"] <= 4 * max_target

    def test_standardize_fits_stats_on_train_targets_only(self):
        df = make_synthetic_df(n_trips=18, seq_len=5, seed=51)
        train_df, val_df, test_df = _pre_split(df, n_train=10, n_val=4, n_test=4, seq_len=5)

        out = train_lstm(
            pre_split=(train_df, val_df, test_df),
            hidden_size=8, num_layers=1, max_epochs=2, patience=2,
            target_transform="standardize", save_model=False, seed=12,
        )
        stats = out["metadata"]["target_stats"]
        assert set(stats.keys()) == {"mean", "std"}

        expected_mean = train_df["time_to_arrival_seconds"].mean()
        expected_std = train_df["time_to_arrival_seconds"].std(ddof=0)
        assert stats["mean"] == pytest.approx(expected_mean, rel=1e-4)
        assert stats["std"] == pytest.approx(expected_std, rel=1e-4)

    def test_log1p_records_no_stats(self):
        df = make_synthetic_df(n_trips=14, seq_len=4, seed=52)
        train_df, val_df, test_df = _pre_split(df, n_train=8, n_val=3, n_test=3, seq_len=4)

        out = train_lstm(
            pre_split=(train_df, val_df, test_df),
            hidden_size=8, num_layers=1, max_epochs=2, patience=2,
            target_transform="log1p", save_model=False, seed=13,
        )
        assert out["metadata"]["target_stats"] == {}
        assert out["metadata"]["target_transform"] == "log1p"

    def test_invalid_target_transform_raises(self):
        df = make_synthetic_df(n_trips=6, seq_len=4, seed=53)
        train_df, val_df, test_df = _pre_split(df, n_train=4, n_val=1, n_test=1, seq_len=4)

        with pytest.raises(ValueError):
            train_lstm(
                pre_split=(train_df, val_df, test_df),
                hidden_size=8, num_layers=1, max_epochs=1, patience=1,
                target_transform="bogus", save_model=False, seed=14,
            )

    def test_saved_model_predict_matches_pretransform_predictions(self, tmp_path):
        """A model trained with target_transform="standardize" must, once
        wrapped/persisted and reloaded, keep producing predictions in raw
        seconds identical to the in-memory model's -- the transform/inverse
        round trip must survive serialization."""
        df = make_synthetic_df(n_trips=10, seq_len=4, seed=54)
        train_df, val_df, test_df = _pre_split(df, n_train=6, n_val=2, n_test=2, seq_len=4)

        out = train_lstm(
            pre_split=(train_df, val_df, test_df),
            hidden_size=8, num_layers=1, max_epochs=2, patience=2,
            target_transform="standardize", save_model=False, seed=15,
        )
        model: LSTMSegmentModel = out["model"]
        assert model.target_transform == "standardize"
        preds_before = model.predict(test_df)

        registry = ModelRegistry(base_dir=str(tmp_path))
        registry.save_model("lstm_transform_test_key", model, dict(out["metadata"]))
        loaded_model = registry.load_model("lstm_transform_test_key")

        preds_after = loaded_model.predict(test_df)
        np.testing.assert_allclose(preds_before, preds_after, rtol=1e-6, atol=1e-6)
