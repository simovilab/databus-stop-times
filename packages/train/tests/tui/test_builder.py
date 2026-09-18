"""builder.py: ExperimentDraft assembly, validation, and the write ->
load_experiment round trip. No Textual import anywhere in this file --
these tests must pass whether or not the optional 'tui' extra is installed.
"""

from __future__ import annotations

from datetime import date

import pytest

from stoptimes.core.contracts.experiment import ParamChoices, ParamRange, load_experiment
from stoptimes.train.tui.builder import (
    ExperimentDraft,
    ExperimentDraftError,
    make_param_choices,
    make_param_range,
    open_draft,
    validate_draft,
    write_draft,
)


def _valid_draft(**overrides) -> ExperimentDraft:
    base = ExperimentDraft(
        name="demo-experiment",
        sources=["mbta"],
        family="xgboost",
        params={"max_depth": 6, "learning_rate": 0.05},
        features=["distance_to_stop", "hour_sin"],
        split_kind="time",
        train_fraction=0.8,
        test_fraction=0.2,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_valid_draft_has_no_problems() -> None:
    assert validate_draft(_valid_draft()) == []


def test_missing_name_and_family_and_data_are_all_reported_together() -> None:
    draft = ExperimentDraft()
    problems = validate_draft(draft)
    assert any("name" in p for p in problems)
    assert any("family" in p for p in problems)
    assert any("source" in p or "dataset" in p for p in problems)


def test_unknown_feature_name_reported_clearly() -> None:
    draft = _valid_draft(features=["not_a_real_feature"])
    problems = validate_draft(draft)
    assert any("not_a_real_feature" in p for p in problems)


def test_kfold_split_without_n_folds_reported_by_contract() -> None:
    draft = _valid_draft(split_kind="kfold", train_fraction=None, test_fraction=None, n_folds=None)
    problems = validate_draft(draft)
    assert any("n_folds" in p for p in problems)


def test_to_experiment_round_trips_through_from_experiment() -> None:
    draft = _valid_draft(
        start_date=date(2026, 6, 1),
        end_date=date(2026, 8, 31),
        routes=["1", "2"],
        tags=["smoke"],
    )
    experiment = draft.to_experiment()
    rebuilt = ExperimentDraft.from_experiment(experiment)
    assert rebuilt == draft


def test_write_draft_writes_loadable_yaml_that_equals_the_built_experiment(tmp_path) -> None:
    draft = _valid_draft()
    out_path = tmp_path / "experiment.yaml"

    written = write_draft(draft, out_path)

    assert out_path.exists()
    reloaded = load_experiment(out_path)
    assert reloaded == written
    assert reloaded == draft.to_experiment()


def test_write_draft_refuses_to_write_an_invalid_draft(tmp_path) -> None:
    draft = ExperimentDraft()  # missing everything
    out_path = tmp_path / "experiment.yaml"

    with pytest.raises(ExperimentDraftError) as exc_info:
        write_draft(draft, out_path)

    assert not out_path.exists()
    assert exc_info.value.problems


def test_open_draft_round_trips_a_shipped_example() -> None:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[4]
    example = repo_root / "examples" / "experiments" / "xgb-baseline.yaml"
    assert example.exists(), example

    draft = open_draft(example)
    experiment = load_experiment(example)
    assert draft.to_experiment() == experiment


def test_edit_then_resave_preserves_untouched_fields(tmp_path) -> None:
    draft = _valid_draft()
    out_path = tmp_path / "experiment.yaml"
    write_draft(draft, out_path)

    reopened = open_draft(out_path)
    reopened.name = "demo-experiment-v2"
    reopened.params["max_depth"] = 8

    rewritten = write_draft(reopened, out_path)

    assert rewritten.name == "demo-experiment-v2"
    assert rewritten.model.params["max_depth"] == 8
    assert rewritten.model.params["learning_rate"] == 0.05  # untouched field preserved
    assert rewritten.data.sources == ["mbta"]  # untouched field preserved


def test_param_range_and_choices_survive_the_round_trip(tmp_path) -> None:
    draft = _valid_draft(
        params={
            "max_depth": make_param_range(3, 10, step=1),
            "n_estimators": make_param_choices([100, 200, 300]),
        }
    )
    out_path = tmp_path / "experiment.yaml"

    write_draft(draft, out_path)
    reloaded = load_experiment(out_path)

    assert isinstance(reloaded.model.params["max_depth"], ParamRange)
    assert reloaded.model.params["max_depth"].low == 3
    assert reloaded.model.params["max_depth"].high == 10
    assert isinstance(reloaded.model.params["n_estimators"], ParamChoices)
    assert reloaded.model.params["n_estimators"].choices == [100, 200, 300]
