"""Thin adapters satisfying `stoptimes.core.contracts.model.Model`.

Every pre-Phase-2 family (`historical_mean`, `polyreg_distance`,
`polyreg_time`, `ewma`, `xgb`) is a plain class with the same shape:
``fit(train_df, target_col=...)`` (mutates in place, returns None) and
``predict(X) -> np.ndarray``. `Model` wants ``fit(X, y, **ctx) -> Model`` and
``predict(X) -> np.ndarray``, plus ``save``/``load``. `FamilyModelAdapter`
bridges the two by gluing `y` back onto `X` as the target column before
calling the family's own `fit` — no family class changes, no rewrite.

`lstm` doesn't fit this shape (it trains on grouped trip sequences, not flat
rows) so it gets its own small adapter, `LSTMModelAdapter`, built on
`lstm.train.train_lstm(pre_split=...)`.

Two baselines that always ship per docs/PLAN.md rule 5 — `schedule` (predict
zero delay) and `constant_speed` (ETA from remaining distance over one
average speed fit at train time) — live here too since they have no existing
family module to wrap.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Type

import numpy as np
import pandas as pd

from stoptimes.core.contracts.artifact import ModelFormat

_MODEL_FILENAME = ModelFormat.PICKLE.filename  # "model.pkl"


class FamilyModelAdapter:
    """Wraps a `models.<family>.train.<Family>Model` instance to satisfy `Model`.

    `required_features` is best-effort: it reflects the inner model's
    `feature_cols`/`group_by` attribute *after* `fit()` (families derive
    their feature list from what's available in the training frame), and is
    empty before fitting or if the family doesn't expose one of those.
    """

    def __init__(self, family: str, inner: Any, *, target_col: str = "time_to_arrival_seconds") -> None:
        self.family = family
        self._inner = inner
        self._target_col = target_col

    @property
    def required_features(self) -> list[str]:
        for attr in ("feature_cols", "group_by"):
            value = getattr(self._inner, attr, None)
            if value:
                return list(value)
        return []

    def fit(self, X: pd.DataFrame, y: np.ndarray, **ctx: Any) -> "FamilyModelAdapter":
        train_df = X.copy()
        train_df[self._target_col] = np.asarray(y)
        # `ModelSpec.params` can't carry a list (contracts.experiment.ParamValue
        # is scalar-or-search-space only), so a family whose grouping key is a
        # list (historical_mean, ewma's `group_by`) can't get it from
        # hyperparameters. Instead it follows whatever columns the experiment
        # already selected via `Experiment.features` -- X's columns are that
        # selection, so this keeps "group by these features" in one place
        # (the experiment's feature list) instead of two.
        if hasattr(self._inner, "group_by") and not ctx.get("keep_group_by"):
            self._inner.group_by = list(X.columns)
        fit_kwargs = {k: v for k, v in ctx.items() if k in _fit_accepted_kwargs(self._inner)}
        self._inner.fit(train_df, target_col=self._target_col, **fit_kwargs)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._inner.predict(X))

    def save(self, directory: Path) -> ModelFormat:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / _MODEL_FILENAME, "wb") as fh:
            pickle.dump(self._inner, fh)
        return ModelFormat.PICKLE

    @classmethod
    def load(cls, directory: Path) -> "FamilyModelAdapter":
        raise NotImplementedError("use the family-specific *Adapter.load(directory)")


def _fit_accepted_kwargs(inner: Any) -> frozenset[str]:
    import inspect

    try:
        return frozenset(inspect.signature(inner.fit).parameters)
    except (TypeError, ValueError):
        return frozenset()


def _make_family_adapter(family: str, inner_cls: Type) -> Type[FamilyModelAdapter]:
    """Build a concrete `<Family>Adapter` class bound to one family's model class."""

    class _Adapter(FamilyModelAdapter):
        def __init__(self, **params: Any) -> None:
            super().__init__(family, inner_cls(**params))

        @classmethod
        def load(cls, directory: Path) -> "_Adapter":
            with open(Path(directory) / _MODEL_FILENAME, "rb") as fh:
                inner = pickle.load(fh)
            adapter = cls.__new__(cls)
            FamilyModelAdapter.__init__(adapter, family, inner)
            return adapter

    _Adapter.__name__ = f"{inner_cls.__name__}Adapter"
    _Adapter.__qualname__ = _Adapter.__name__
    return _Adapter


def _historical_mean_adapter_cls():
    from stoptimes.models.historical_mean.train import HistoricalMeanModel

    return _make_family_adapter("historical_mean", HistoricalMeanModel)


def _ewma_adapter_cls():
    from stoptimes.models.ewma.train import EWMAModel

    return _make_family_adapter("ewma", EWMAModel)


def _polyreg_distance_adapter_cls():
    from stoptimes.models.polyreg_distance.train import PolyRegDistanceModel

    return _make_family_adapter("polyreg_distance", PolyRegDistanceModel)


def _polyreg_time_adapter_cls():
    from stoptimes.models.polyreg_time.train import PolyRegTimeModel

    return _make_family_adapter("polyreg_time", PolyRegTimeModel)


def _xgb_adapter_cls():
    from stoptimes.models.xgb.train import XGBTimeModel

    return _make_family_adapter("xgb", XGBTimeModel)


class LSTMModelAdapter:
    """Adapts `lstm.train.LSTMSegmentModel` (trip-sequence model) to `Model`.

    `fit` cannot use the flat `FamilyModelAdapter` shape: `train_lstm` needs
    a train/validation split of whole trips for early stopping, not one flat
    `(X, y)`. This carves the last `ctx['val_fraction']` (default 10%) of
    rows off `X`/`y`, in the order given, as validation -- callers doing a
    time-ordered split (as `train.experiment` does) get a validation window
    trailing the training window, which is the same shape `train_lstm` uses
    when reading a dataset itself.
    """

    family = "lstm"
    _LSTM_FIT_KWARGS = frozenset(
        {
            "hidden_size",
            "num_layers",
            "dropout",
            "rnn_type",
            "learning_rate",
            "batch_size",
            "max_epochs",
            "patience",
            "weight_decay",
            "grad_clip",
            "loss",
            "target_transform",
            "service_offset_h",
            "seed",
        }
    )

    def __init__(self, **params: Any) -> None:
        self._params = dict(params)
        self._inner = None

    @property
    def required_features(self) -> list[str]:
        from stoptimes.models.lstm.data import FEATURE_COLUMNS

        return list(FEATURE_COLUMNS)

    def fit(self, X: pd.DataFrame, y: np.ndarray, **ctx: Any) -> "LSTMModelAdapter":
        from stoptimes.models.lstm.data import TARGET_COLUMN
        from stoptimes.models.lstm.train import train_lstm

        df = X.copy()
        df[TARGET_COLUMN] = np.asarray(y)

        val_fraction = float(ctx.get("val_fraction", 0.1))
        n_val = max(1, int(len(df) * val_fraction)) if len(df) > 1 else 0
        if n_val == 0 or n_val >= len(df):
            train_df, val_df = df, df
        else:
            train_df, val_df = df.iloc[:-n_val].copy(), df.iloc[-n_val:].copy()

        kwargs = dict(self._params)
        kwargs.update({k: v for k, v in ctx.items() if k in self._LSTM_FIT_KWARGS})

        result = train_lstm(pre_split=(train_df, val_df, val_df), save_model=False, **kwargs)
        self._inner = result["model"]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._inner is None:
            raise ValueError("LSTMModelAdapter not fitted")
        return np.asarray(self._inner.predict(X))

    def save(self, directory: Path) -> ModelFormat:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / _MODEL_FILENAME, "wb") as fh:
            pickle.dump(self._inner, fh)
        return ModelFormat.PICKLE

    @classmethod
    def load(cls, directory: Path) -> "LSTMModelAdapter":
        with open(Path(directory) / _MODEL_FILENAME, "rb") as fh:
            inner = pickle.load(fh)
        adapter = cls()
        adapter._inner = inner
        return adapter


class _BaselineAdapter:
    """Shared save/load for the dependency-free baselines below."""

    def save(self, directory: Path) -> ModelFormat:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / _MODEL_FILENAME, "wb") as fh:
            pickle.dump(self.__dict__, fh)
        return ModelFormat.PICKLE

    @classmethod
    def load(cls, directory: Path) -> "_BaselineAdapter":
        with open(Path(directory) / _MODEL_FILENAME, "rb") as fh:
            state = pickle.load(fh)
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        return obj


class ScheduleBaseline(_BaselineAdapter):
    """Predicts zero delay: the vehicle arrives exactly on schedule.

    No columns needed to compute it, no fitting beyond recording that it was
    fit — the target here (`time_to_arrival_seconds`/`arrival_delay_s`) *is*
    the deviation from schedule, so "predict schedule" is "predict zero".
    Always shipped as a baseline (docs/PLAN.md rule 5).
    """

    family = "schedule"
    required_features: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray, **ctx: Any) -> "ScheduleBaseline":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X), dtype=float)


class ConstantSpeedBaseline(_BaselineAdapter):
    """ETA = remaining distance / one average speed, fit once on train data.

    `required_features` is `["distance_to_stop"]`; `speed_m_s` is the
    training target's implied average speed (median of
    `distance_to_stop / y` over rows with a positive target), so predicting
    needs only geometry already computed by the dataset builder — no
    per-route or per-time conditioning, which is the point of a baseline.
    Always shipped as a baseline (docs/PLAN.md rule 5).
    """

    family = "constant_speed"
    required_features = ["distance_to_stop"]

    DEFAULT_SPEED_M_S = 8.0  # ~29 km/h, a reasonable urban-bus fallback

    def __init__(self) -> None:
        self.speed_m_s = self.DEFAULT_SPEED_M_S

    def fit(self, X: pd.DataFrame, y: np.ndarray, **ctx: Any) -> "ConstantSpeedBaseline":
        distance = X["distance_to_stop"].to_numpy(dtype=float)
        target = np.asarray(y, dtype=float)
        valid = (target > 0) & np.isfinite(distance) & (distance >= 0)
        if valid.any():
            speeds = distance[valid] / target[valid]
            speeds = speeds[np.isfinite(speeds) & (speeds > 0)]
            if len(speeds):
                self.speed_m_s = float(np.median(speeds))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        distance = X["distance_to_stop"].to_numpy(dtype=float)
        return np.maximum(distance, 0.0) / self.speed_m_s


_FAMILY_FACTORIES = {
    "historical_mean": _historical_mean_adapter_cls,
    "ewma": _ewma_adapter_cls,
    "polyreg_distance": _polyreg_distance_adapter_cls,
    "polyreg_time": _polyreg_time_adapter_cls,
    "xgb": _xgb_adapter_cls,
    "xgboost": _xgb_adapter_cls,
}

_BASELINE_FACTORIES = {
    "schedule": ScheduleBaseline,
    "constant_speed": ConstantSpeedBaseline,
}


def get_model_class(family: str) -> Type:
    """Resolve a family name (from `ModelSpec.family` / `Experiment.baselines`)
    to a class satisfying `Model`, importing that family's (possibly
    optional, e.g. xgboost/torch) dependencies lazily.

    Raises `KeyError` for an unknown family name.
    """
    if family == "lstm":
        return LSTMModelAdapter
    if family in _BASELINE_FACTORIES:
        return _BASELINE_FACTORIES[family]
    if family in _FAMILY_FACTORIES:
        return _FAMILY_FACTORIES[family]()
    raise KeyError(
        f"unknown model family {family!r}; available: "
        f"{sorted({*_FAMILY_FACTORIES, *_BASELINE_FACTORIES, 'lstm'})}"
    )
