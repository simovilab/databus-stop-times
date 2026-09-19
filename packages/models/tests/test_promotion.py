"""Tests for stoptimes.models.common.promotion: the promote/rollback history
sidecar (U7). The pointer itself (`serving.selection.pins`) is CLI-owned
(`commands/models.py`) and covered there; this file covers the history
bookkeeping in isolation.
"""

from __future__ import annotations

import json

import pytest
from stoptimes.models.common.promotion import (
    RollbackError,
    current_promotion,
    load_history,
    promotions_path,
    record_promotion,
    rollback_target,
)


def test_promotions_path_derives_sidecar_name(tmp_path):
    profile = tmp_path / "mbta.yaml"
    assert promotions_path(profile) == tmp_path / "mbta.promotions.json"


def test_load_history_missing_file_is_empty(tmp_path):
    assert load_history(tmp_path / "nope.promotions.json") == {}


def test_record_promotion_appends_and_persists(tmp_path):
    path = tmp_path / "mbta.promotions.json"
    record_promotion(path, "global", "a1")
    history = record_promotion(path, "global", "a2")
    assert [e["artifact_id"] for e in history["global"]] == ["a1", "a2"]

    reloaded = load_history(path)
    assert [e["artifact_id"] for e in reloaded["global"]] == ["a1", "a2"]
    for entry in reloaded["global"]:
        assert "at" in entry


def test_record_promotion_keeps_pins_independent(tmp_path):
    path = tmp_path / "mbta.promotions.json"
    record_promotion(path, "global", "a1")
    history = record_promotion(path, "route:22", "r1")
    assert [e["artifact_id"] for e in history["global"]] == ["a1"]
    assert [e["artifact_id"] for e in history["route:22"]] == ["r1"]


def test_current_promotion_returns_latest_or_none(tmp_path):
    path = tmp_path / "mbta.promotions.json"
    assert current_promotion(load_history(path), "global") is None
    record_promotion(path, "global", "a1")
    history = record_promotion(path, "global", "a2")
    assert current_promotion(history, "global")["artifact_id"] == "a2"


def test_rollback_target_restores_previous_and_pops_current(tmp_path):
    path = tmp_path / "mbta.promotions.json"
    record_promotion(path, "global", "a1")
    record_promotion(path, "global", "a2")
    record_promotion(path, "global", "a3")

    restored, history = rollback_target(path, "global")
    assert restored == "a2"
    assert [e["artifact_id"] for e in history["global"]] == ["a1", "a2"]

    # A second rollback goes one further back.
    restored2, history2 = rollback_target(path, "global")
    assert restored2 == "a1"
    assert [e["artifact_id"] for e in history2["global"]] == ["a1"]


def test_rollback_target_raises_with_no_history(tmp_path):
    path = tmp_path / "mbta.promotions.json"
    with pytest.raises(RollbackError):
        rollback_target(path, "global")


def test_rollback_target_raises_with_only_one_promotion(tmp_path):
    path = tmp_path / "mbta.promotions.json"
    record_promotion(path, "global", "a1")
    with pytest.raises(RollbackError):
        rollback_target(path, "global")


def test_history_file_is_readable_json(tmp_path):
    path = tmp_path / "mbta.promotions.json"
    record_promotion(path, "global", "a1")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["pins"]["global"][0]["artifact_id"] == "a1"
