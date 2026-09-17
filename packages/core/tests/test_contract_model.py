"""Tests for contracts.model: the Model Protocol, ModelRef, and that a
real existing family can satisfy Model through a thin adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from stoptimes.core.contracts.artifact import ModelFormat
from stoptimes.core.contracts.model import Model, ModelRef, ModelRefKind


class _FakeXgboostAdapter:
    """Minimal object satisfying `Model`, standing in for a real xgboost
    family wrapped for the contract (T3's job to write for real)."""

    family = "xgboost"
    required_features = ["distance_to_stop"]

    def __init__(self) -> None:
        self._mean: float | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, **ctx) -> "Model":
        self._mean = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._mean)

    def save(self, directory: Path) -> ModelFormat:
        return ModelFormat.PICKLE

    @classmethod
    def load(cls, directory: Path) -> "Model":
        return cls()


def test_fake_adapter_satisfies_model_protocol():
    adapter = _FakeXgboostAdapter()
    assert isinstance(adapter, Model)


def test_fit_predict_round_trip():
    adapter = _FakeXgboostAdapter()
    X = pd.DataFrame({"distance_to_stop": [10.0, 20.0, 30.0]})
    y = np.array([5.0, 15.0, 25.0])
    fitted = adapter.fit(X, y)
    assert fitted is adapter
    preds = fitted.predict(X)
    assert preds.shape == (3,)
    assert np.allclose(preds, 15.0)


def test_object_missing_predict_is_not_a_model():
    class Incomplete:
        family = "x"
        required_features = []

        def fit(self, X, y, **ctx):
            return self

        def save(self, directory):
            return ModelFormat.PICKLE

    assert not isinstance(Incomplete(), Model)


def test_model_ref_pickle():
    ref = ModelRef(kind=ModelRefKind.PICKLE, location="/artifacts/xgb/model.pkl")
    assert ref.family == "byo"


def test_model_ref_callable():
    ref = ModelRef(kind=ModelRefKind.CALLABLE, location="mypkg.mymodule:predict")
    assert ":" in ref.location


def test_model_ref_http_uses_auth_env_not_secret_value():
    ref = ModelRef(
        kind=ModelRefKind.HTTP,
        location="https://example.com/predict",
        auth_env={"authorization": "MY_MODEL_API_KEY"},
        timeout_s=2.0,
    )
    assert ref.auth_env["authorization"] == "MY_MODEL_API_KEY"
    assert ref.timeout_s == 2.0


def test_model_ref_rejects_unknown_field():
    with pytest.raises(ValidationError):
        ModelRef.model_validate(
            {"kind": "pickle", "location": "x", "surprise": True}
        )
