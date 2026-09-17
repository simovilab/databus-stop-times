"""Tests for contracts.experiment: schema, round-trip, examples/experiments/*.yaml."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from stoptimes.core.contracts.experiment import (
    DEFAULT_BASELINES,
    DataSelection,
    Experiment,
    ModelSpec,
    ParamChoices,
    ParamRange,
    SplitStrategy,
    dump_experiment,
    load_experiment,
)

EXAMPLES_DIR = Path(__file__).parents[3] / "examples" / "experiments"


def _experiment(**overrides) -> Experiment:
    base = dict(
        name="test-experiment",
        data=DataSelection(sources=["mbta"]),
        model=ModelSpec(family="xgboost", params={"max_depth": 5}),
        features=["distance_to_stop"],
        split=SplitStrategy(kind="time", train_fraction=0.8, test_fraction=0.2),
    )
    base.update(overrides)
    return Experiment(**base)


def test_round_trip_yaml(tmp_path):
    experiment = _experiment()
    path = dump_experiment(experiment, tmp_path / "e.yaml")
    assert load_experiment(path) == experiment


def test_baselines_default_and_overridable():
    assert _experiment().baselines == list(DEFAULT_BASELINES)
    assert _experiment(baselines=[]).baselines == []


def test_data_selection_requires_sources_or_datasets():
    with pytest.raises(ValidationError):
        DataSelection()
    DataSelection(datasets=["mbta-2026-summer"])  # ok


def test_split_time_requires_fractions():
    with pytest.raises(ValidationError):
        SplitStrategy(kind="time")
    SplitStrategy(kind="time", train_fraction=0.7, test_fraction=0.3)


def test_split_kfold_requires_n_folds():
    with pytest.raises(ValidationError):
        SplitStrategy(kind="kfold")
    SplitStrategy(kind="kfold", n_folds=5)


def test_split_fractions_cannot_exceed_one():
    with pytest.raises(ValidationError):
        SplitStrategy(kind="time", train_fraction=0.8, test_fraction=0.5)


def test_param_value_accepts_scalars_and_search_spaces():
    experiment = _experiment(
        model=ModelSpec(
            family="xgboost",
            params={
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": ParamRange(low=0.5, high=1.0),
                "booster": ParamChoices(choices=["gbtree", "dart"]),
            },
        )
    )
    assert experiment.model.params["max_depth"] == 6
    assert isinstance(experiment.model.params["subsample"], ParamRange)
    assert isinstance(experiment.model.params["booster"], ParamChoices)


def test_unknown_top_level_field_rejected():
    with pytest.raises(ValidationError):
        Experiment.model_validate({**_experiment().model_dump(), "surprise": 1})


@pytest.mark.parametrize(
    "filename", ["xgb-baseline.yaml", "historical-mean-kfold.yaml"]
)
def test_shipped_examples_validate(filename):
    path = EXAMPLES_DIR / filename
    assert path.exists(), f"missing example {path}"
    experiment = load_experiment(path)
    assert experiment.baselines  # baselines always ship
    assert experiment.model.family
    assert experiment.features
