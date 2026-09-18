from datetime import datetime

import pytest
from pydantic import ValidationError

from stoptimes.core.contracts.artifact import ArtifactMetadata, ModelScope, Provenance, ScopeLevel
from stoptimes.core.contracts.selection import SelectionPolicy, select_artifact


def artifact(artifact_id, mae, scope=None, family="xgboost", metric="test_mae_seconds"):
    return ArtifactMetadata(
        artifact_id=artifact_id,
        family=family,
        scope=scope or ModelScope(source_id="mbta"),
        metrics={metric: mae},
        provenance=Provenance(created_at=datetime(2026, 9, 1)),
    )


GLOBAL = artifact("global_xgb", 60.0)
ROUTE_22 = artifact("route22_xgb", 50.0, ModelScope(level=ScopeLevel.ROUTE, source_id="mbta", route_ids=["22"]))
ROUTE_22_BAD = artifact("route22_hm", 80.0, ModelScope(level=ScopeLevel.ROUTE, source_id="mbta", route_ids=["22"]))
TRIP_T1 = artifact("trip1_xgb", 55.0, ModelScope(level=ScopeLevel.TRIP, source_id="mbta", trip_ids=["t1"]))
ALL = [GLOBAL, ROUTE_22, ROUTE_22_BAD, TRIP_T1]


def test_most_specific_scope_wins_even_with_worse_metric():
    # trip model's MAE (55) is worse than the route model's (50): specificity first.
    chosen = select_artifact(ALL, SelectionPolicy(), source_id="mbta", route_id="22", trip_id="t1")
    assert chosen.artifact_id == "trip1_xgb"


def test_best_metric_within_the_chosen_scope():
    chosen = select_artifact(ALL, SelectionPolicy(), source_id="mbta", route_id="22", trip_id="other")
    assert chosen.artifact_id == "route22_xgb"


def test_falls_back_to_global_for_an_unknown_route():
    chosen = select_artifact(ALL, SelectionPolicy(), source_id="mbta", route_id="999")
    assert chosen.artifact_id == "global_xgb"


def test_other_sources_are_never_selected():
    bucr = artifact("bucr_global", 1.0, ModelScope(source_id="bucr"))
    assert select_artifact([bucr], SelectionPolicy(), source_id="mbta") is None


def test_pins_win_over_ranking():
    policy = SelectionPolicy(pins={"route:22": "route22_hm"})
    chosen = select_artifact(ALL, policy, source_id="mbta", route_id="22")
    assert chosen.artifact_id == "route22_hm"


def test_pin_to_a_missing_artifact_falls_through_to_ranking():
    policy = SelectionPolicy(pins={"global": "deleted_model"})
    chosen = select_artifact(ALL, policy, source_id="mbta", route_id="999")
    assert chosen.artifact_id == "global_xgb"


def test_scope_order_can_be_narrowed_to_global_only():
    policy = SelectionPolicy(scope_order=[ScopeLevel.GLOBAL])
    chosen = select_artifact(ALL, policy, source_id="mbta", route_id="22", trip_id="t1")
    assert chosen.artifact_id == "global_xgb"


def test_artifacts_without_the_metric_are_skipped():
    no_metric = artifact("no_metric", 1.0, metric="r2")
    assert select_artifact([no_metric], SelectionPolicy(), source_id="mbta") is None


def test_higher_is_better_metric():
    policy = SelectionPolicy(metric="r2", lower_is_better=False)
    worse = artifact("r2_low", 0.4, metric="r2")
    better = artifact("r2_high", 0.8, metric="r2")
    assert select_artifact([worse, better], policy, source_id="mbta").artifact_id == "r2_high"


def test_family_filter():
    policy = SelectionPolicy(families=["historical_mean"])
    hm = artifact("hm_global", 90.0, family="historical_mean")
    assert select_artifact([*ALL, hm], policy, source_id="mbta").artifact_id == "hm_global"


def test_empty_candidates_return_none():
    assert select_artifact([], SelectionPolicy(), source_id="mbta") is None


def test_direction_scope_matches_route_and_direction():
    scope = ModelScope(level=ScopeLevel.DIRECTION, source_id="mbta", route_ids=["22"], direction_id=0)
    assert scope.matches(source_id="mbta", route_id="22", direction_id=0)
    assert not scope.matches(source_id="mbta", route_id="22", direction_id=1)


def test_scope_requires_the_keys_of_its_level():
    with pytest.raises(ValidationError):
        ModelScope(level=ScopeLevel.ROUTE, source_id="mbta")
    with pytest.raises(ValidationError):
        ModelScope(level=ScopeLevel.TRIP, source_id="mbta")
    with pytest.raises(ValidationError):
        ModelScope(level=ScopeLevel.DIRECTION, source_id="mbta", route_ids=["22"])


def test_a_source_agnostic_artifact_matches_any_source():
    shared = artifact("shared", 10.0, ModelScope())
    assert select_artifact([shared], SelectionPolicy(), source_id="anything").artifact_id == "shared"
