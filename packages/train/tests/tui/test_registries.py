"""registries.py sources TUI choices from real registries, not hardcoded
lists -- these tests check that the family/baseline set matches
`contract_adapter.get_model_class`'s own accepted names, that every family
has a description, and that features/targets are non-empty and documented.
"""

from __future__ import annotations

import pytest

from stoptimes.train.tui.registries import (
    list_baselines,
    list_features,
    list_model_families,
    list_targets,
)


def test_model_families_match_get_model_class() -> None:
    from stoptimes.models.common.contract_adapter import get_model_class

    families = list_model_families()
    assert families, "expected at least one model family"
    for info in families:
        assert get_model_class(info.name) is not None
        assert info.description, f"{info.name} has no description"


def test_model_families_include_known_names() -> None:
    names = {f.name for f in list_model_families()}
    assert {"xgb", "xgboost", "historical_mean", "ewma", "polyreg_distance", "polyreg_time", "lstm"} <= names


def test_unknown_family_rejected_by_get_model_class() -> None:
    from stoptimes.models.common.contract_adapter import get_model_class

    with pytest.raises(KeyError):
        get_model_class("not-a-real-family")


def test_xgb_params_introspected_from_real_signature() -> None:
    families = {f.name: f for f in list_model_families()}
    xgb = families["xgb"]
    param_names = {p.name for p in xgb.params}
    assert {"max_depth", "n_estimators", "learning_rate"} <= param_names
    max_depth = next(p for p in xgb.params if p.name == "max_depth")
    assert max_depth.default == 5
    assert max_depth.tunable is True


def test_baselines_are_not_in_model_family_choices_by_default() -> None:
    names = {f.name for f in list_model_families()}
    assert "schedule" not in names
    assert "constant_speed" not in names


def test_baselines_available_separately() -> None:
    baselines = list_baselines()
    names = {b.name for b in baselines}
    assert names == {"schedule", "constant_speed"}
    assert all(b.description for b in baselines)


def test_include_baselines_flag_merges_both_lists() -> None:
    with_baselines = list_model_families(include_baselines=True)
    names = {f.name for f in with_baselines}
    assert "schedule" in names
    assert "constant_speed" in names


def test_features_come_from_core_registry_and_are_documented() -> None:
    from stoptimes.core.features.registry import list_features as core_list_features

    features = list_features()
    assert [f.name for f in features] == [f.name for f in core_list_features()]
    assert all(f.description for f in features)


def test_targets_include_the_experiment_default_and_legacy_alias() -> None:
    names = {t.name for t in list_targets()}
    assert "arrival_delay_s" in names
    assert "time_to_arrival_seconds" in names
    assert all(t.description for t in list_targets())
