"""Tests for stoptimes.models.common.contract_adapter: family/baseline
adapters satisfying stoptimes.core.contracts.model.Model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stoptimes.core.contracts.model import Model
from stoptimes.models.common.contract_adapter import (
    ConstantSpeedBaseline,
    ScheduleBaseline,
    get_model_class,
)


def _toy_frame(n=40):
    rng = np.random.RandomState(0)
    df = pd.DataFrame(
        {
            "route_id": ["r1"] * n,
            "stop_sequence": rng.randint(1, 5, size=n),
            "hour": rng.randint(0, 24, size=n),
            "distance_to_stop": rng.uniform(10, 500, size=n),
            "vp_ts": pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC"),
        }
    )
    y = df["distance_to_stop"].to_numpy() / 8.0 + rng.normal(0, 2, size=n)
    y = np.clip(y, 1, None)
    return df, y


@pytest.mark.parametrize("family", ["historical_mean", "polyreg_distance", "ewma"])
def test_family_adapter_satisfies_model_protocol(family, tmp_path):
    cls = get_model_class(family)
    adapter = cls()
    X, y = _toy_frame()
    fitted = adapter.fit(X, y)
    assert fitted is adapter
    assert isinstance(adapter, Model)

    preds = adapter.predict(X)
    assert preds.shape == (len(X),)

    directory = tmp_path / family
    fmt = adapter.save(directory)
    assert (directory / fmt.filename).exists()

    reloaded = cls.load(directory)
    assert isinstance(reloaded, Model)
    reloaded_preds = reloaded.predict(X)
    assert np.allclose(reloaded_preds, preds)


def test_xgb_adapter_satisfies_model_protocol(tmp_path):
    pytest.importorskip("xgboost")
    cls = get_model_class("xgb")
    adapter = cls(n_estimators=5, max_depth=2)
    X, y = _toy_frame(n=80)
    adapter.fit(X, y)
    assert isinstance(adapter, Model)
    preds = adapter.predict(X)
    assert preds.shape == (len(X),)

    directory = tmp_path / "xgb"
    adapter.save(directory)
    reloaded = cls.load(directory)
    assert np.allclose(reloaded.predict(X), preds)


def test_unknown_family_raises_key_error():
    with pytest.raises(KeyError):
        get_model_class("not-a-family")


def test_schedule_baseline_predicts_zero_and_needs_no_features(tmp_path):
    baseline = ScheduleBaseline()
    assert isinstance(baseline, Model)
    X = pd.DataFrame({"anything": [1, 2, 3]})
    y = np.array([10.0, 20.0, 30.0])
    baseline.fit(X, y)
    preds = baseline.predict(X)
    assert np.array_equal(preds, np.zeros(3))

    directory = tmp_path / "schedule"
    baseline.save(directory)
    reloaded = ScheduleBaseline.load(directory)
    assert np.array_equal(reloaded.predict(X), preds)


def test_constant_speed_baseline_fits_a_speed_from_train_data(tmp_path):
    baseline = ConstantSpeedBaseline()
    assert isinstance(baseline, Model)
    X = pd.DataFrame({"distance_to_stop": [80.0, 160.0, 240.0]})
    y = np.array([10.0, 20.0, 30.0])  # exactly 8 m/s
    baseline.fit(X, y)
    assert baseline.speed_m_s == pytest.approx(8.0)

    preds = baseline.predict(pd.DataFrame({"distance_to_stop": [8.0]}))
    assert preds[0] == pytest.approx(1.0)

    directory = tmp_path / "constant_speed"
    baseline.save(directory)
    reloaded = ConstantSpeedBaseline.load(directory)
    assert reloaded.speed_m_s == pytest.approx(8.0)


def test_constant_speed_baseline_falls_back_to_default_speed_without_valid_data():
    baseline = ConstantSpeedBaseline()
    X = pd.DataFrame({"distance_to_stop": [0.0, 0.0]})
    y = np.array([0.0, 0.0])
    baseline.fit(X, y)
    assert baseline.speed_m_s == ConstantSpeedBaseline.DEFAULT_SPEED_M_S


def test_lstm_adapter_satisfies_model_protocol(tmp_path):
    torch = pytest.importorskip("torch")
    from stoptimes.models.lstm.data import FEATURE_COLUMNS

    cls = get_model_class("lstm")
    adapter = cls(hidden_size=4, num_layers=1, max_epochs=1, patience=1)
    assert adapter.required_features == list(FEATURE_COLUMNS)

    n = 60
    rng = np.random.RandomState(0)
    df = pd.DataFrame(
        {
            "trip_id": [f"t{i // 6}" for i in range(n)],
            "route_id": "r1",
            "vp_ts": pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC"),
            **{col: rng.uniform(0, 100, size=n) for col in FEATURE_COLUMNS},
        }
    )
    y = rng.uniform(1, 100, size=n)

    fitted = adapter.fit(df, y, seed=0)
    assert fitted is adapter
    assert isinstance(adapter, Model)
    preds = adapter.predict(df)
    assert preds.shape == (n,)

    directory = tmp_path / "lstm"
    adapter.save(directory)
    reloaded = cls.load(directory)
    assert np.allclose(reloaded.predict(df), preds)
