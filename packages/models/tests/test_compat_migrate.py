"""Tests for stoptimes.models.compat.migrate: legacy gtfs-eta registry ->
artifact contract layout.

Synthetic legacy registries are built in `tmp_path`: a pickle of a tiny
`PolyRegDistanceModel` re-homed under a spoofed pre-migration module name
(mirroring `test_legacy_pickle_compat._pickle_bytes_under_legacy_module`),
plus a hand-written `{key}_meta.json` matching real gtfs-eta artifacts'
shape (top-level hyperparameters, `features`, `*_samples`, `metrics`,
optionally a `provenance` block).
"""

from __future__ import annotations

import json
import pickle
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from stoptimes.core.contracts.artifact import ModelFormat, read_metadata
from stoptimes.models.compat.migrate import migrate, plan_migration
from stoptimes.models.polyreg_distance.train import PolyRegDistanceModel


def _pickle_bytes_under_legacy_module(obj: Any, legacy_module: str) -> bytes:
    """See test_legacy_pickle_compat.py for the full rationale: spoof the
    class's `__module__` (and register fake parent packages in
    `sys.modules`, since pickle's `save_global` imports them) so the bytes
    on disk look exactly like a real pre-migration artifact."""
    cls = type(obj)
    original_module = cls.__module__

    leaf_parts = legacy_module.split(".")
    module_names = [".".join(leaf_parts[: i + 1]) for i in range(len(leaf_parts))]
    inserted = [name for name in module_names if name not in sys.modules]
    for name in inserted:
        sys.modules[name] = types.ModuleType(name)
    setattr(sys.modules[legacy_module], cls.__name__, cls)

    cls.__module__ = legacy_module
    try:
        return pickle.dumps(obj)
    finally:
        cls.__module__ = original_module
        for name in inserted:
            del sys.modules[name]


def _fit_polyreg_distance(**kwargs) -> PolyRegDistanceModel:
    model = PolyRegDistanceModel(degree=kwargs.get("degree", 1), alpha=kwargs.get("alpha", 1.0),
                                  route_specific=False)
    distances = np.linspace(0, 3000, 50)
    train_df = pd.DataFrame(
        {"distance_to_stop": distances, "time_to_arrival_seconds": distances / 4.5}
    )
    model.fit(train_df)
    return model


def _write_legacy_artifact(
    legacy_dir: Path,
    key: str,
    *,
    legacy_module: str = "polyreg_distance.train",
    meta: dict[str, Any] | None = None,
    model=None,
) -> PolyRegDistanceModel:
    """Write `{key}.pkl` + `{key}_meta.json` into `legacy_dir`, mirroring a
    real gtfs-eta `ModelRegistry.save_model` output."""
    legacy_dir.mkdir(parents=True, exist_ok=True)
    model = model if model is not None else _fit_polyreg_distance()
    payload = _pickle_bytes_under_legacy_module(model, legacy_module)
    (legacy_dir / f"{key}.pkl").write_bytes(payload)

    full_meta = {
        "model_type": "polyreg_distance",
        "dataset": "unit_test_dataset",
        "route_id": None,
        "degree": 1,
        "alpha": 1.0,
        "route_specific": False,
        "features": ["distance_to_stop"],
        "n_samples": 50,
        "train_samples": 40,
        "test_samples": 10,
        "metrics": {"val_mae_seconds": 12.3, "test_mae_seconds": 13.4, "note": "n/a"},
        "model_key": key,
        "saved_at": "2026-08-18T11:57:00.478753",
        "model_path": f"{key}.pkl",
    }
    if meta:
        full_meta.update(meta)
    (legacy_dir / f"{key}_meta.json").write_text(json.dumps(full_meta), encoding="utf-8")
    return model


PROVENANCE_BLOCK = {
    "git_sha": "abc1234",
    "lib_versions": {"scikit-learn": "1.9.1", "numpy": "2.1.0"},
    "python_version": "3.12.4",
    "dataset_name": "unit_test_dataset",
    "dataset_sha256": "deadbeef" * 8,
    "seed": 42,
    "created_at": "2026-08-18T11:57:00.478753",
}


# ── plan_migration ────────────────────────────────────────────────────────────


def test_plan_pairs_pkl_and_meta(tmp_path):
    _write_legacy_artifact(tmp_path, "model_a")
    _write_legacy_artifact(tmp_path, "model_b")

    plan = plan_migration(tmp_path)

    assert {i.key for i in plan} == {"model_a", "model_b"}
    assert all(i.status == "planned" for i in plan)
    assert all(i.artifact_id == i.key for i in plan)


def test_plan_reports_orphans(tmp_path):
    _write_legacy_artifact(tmp_path, "paired")
    (tmp_path / "orphan_pkl.pkl").write_bytes(pickle.dumps({"x": 1}))
    (tmp_path / "orphan_meta_meta.json").write_text("{}", encoding="utf-8")

    plan = {i.key: i for i in plan_migration(tmp_path)}

    assert plan["paired"].status == "planned"
    assert plan["orphan_pkl"].status == "orphan"
    assert "metadata" in plan["orphan_pkl"].reason
    assert plan["orphan_meta"].status == "orphan"
    assert "model" in plan["orphan_meta"].reason


# ── migrate: happy path ───────────────────────────────────────────────────────


def test_migrate_writes_readable_artifact_with_legacy_preserved(tmp_path):
    legacy_dir = tmp_path / "legacy"
    registry_dir = tmp_path / "registry"
    legacy_meta_extra = {"provenance": PROVENANCE_BLOCK}
    _write_legacy_artifact(legacy_dir, "xgboost_test_key", meta=legacy_meta_extra)

    report = migrate(legacy_dir, registry_dir, dry_run=False)

    assert [i.key for i in report.migrated] == ["xgboost_test_key"]
    assert report.ok

    meta = read_metadata(registry_dir / "xgboost_test_key")
    assert meta.artifact_id == "xgboost_test_key"
    assert meta.family == "polyreg_distance"
    assert meta.features == ["distance_to_stop"]
    assert meta.params == {"degree": 1, "alpha": 1.0, "route_specific": False}
    assert meta.data.dataset == "unit_test_dataset"
    assert meta.data.n_samples == 50
    assert meta.data.n_train == 40
    assert meta.data.n_test == 10
    assert meta.data.dataset_sha256 == PROVENANCE_BLOCK["dataset_sha256"]
    assert meta.provenance.git_sha == "abc1234"
    assert meta.provenance.lib_versions == PROVENANCE_BLOCK["lib_versions"]
    assert meta.provenance.seed == 42
    assert meta.metrics == {"val_mae_seconds": 12.3, "test_mae_seconds": 13.4}
    assert meta.tags == ["migrated"]
    assert meta.format == ModelFormat.PICKLE
    assert meta.legacy["model_key"] == "xgboost_test_key"
    assert meta.legacy["degree"] == 1


def test_migrated_model_loads_with_plain_pickle_and_predicts_identically(tmp_path):
    legacy_dir = tmp_path / "legacy"
    registry_dir = tmp_path / "registry"
    original = _write_legacy_artifact(legacy_dir, "poly_key")
    probe = pd.DataFrame({"distance_to_stop": [100.0, 500.0, 1000.0]})
    expected = original.predict(probe)

    migrate(legacy_dir, registry_dir)

    model_path = registry_dir / "poly_key" / "model.pkl"
    with open(model_path, "rb") as fh:
        reloaded = pickle.load(fh)  # plain pickle -- no LegacyModuleUnpickler

    assert type(reloaded).__module__ == "stoptimes.models.polyreg_distance.train"
    np.testing.assert_allclose(reloaded.predict(probe), expected)


def test_missing_provenance_warns_and_falls_back_to_saved_at(tmp_path):
    legacy_dir = tmp_path / "legacy"
    registry_dir = tmp_path / "registry"
    _write_legacy_artifact(legacy_dir, "no_provenance_key")

    report = migrate(legacy_dir, registry_dir)

    item = report.migrated[0]
    assert any("lib_versions" in w for w in item.warnings)

    meta = read_metadata(registry_dir / "no_provenance_key")
    assert meta.provenance.lib_versions == {}
    assert meta.provenance.created_at == datetime.fromisoformat("2026-08-18T11:57:00.478753")


# ── idempotence / overwrite ───────────────────────────────────────────────────


def test_rerun_without_overwrite_skips_existing(tmp_path):
    legacy_dir = tmp_path / "legacy"
    registry_dir = tmp_path / "registry"
    _write_legacy_artifact(legacy_dir, "idem_key")

    first = migrate(legacy_dir, registry_dir)
    second = migrate(legacy_dir, registry_dir)

    assert [i.key for i in first.migrated] == ["idem_key"]
    assert second.migrated == []
    assert [i.key for i in second.skipped] == ["idem_key"]


def test_overwrite_true_remigrates(tmp_path):
    legacy_dir = tmp_path / "legacy"
    registry_dir = tmp_path / "registry"
    _write_legacy_artifact(legacy_dir, "idem_key", meta={"degree": 1})
    migrate(legacy_dir, registry_dir)

    # Change the legacy metadata and re-migrate with overwrite.
    _write_legacy_artifact(legacy_dir, "idem_key", meta={"degree": 3})
    report = migrate(legacy_dir, registry_dir, overwrite=True)

    assert [i.key for i in report.migrated] == ["idem_key"]
    meta = read_metadata(registry_dir / "idem_key")
    assert meta.params["degree"] == 3


# ── dry run ───────────────────────────────────────────────────────────────────


def test_dry_run_writes_nothing(tmp_path):
    legacy_dir = tmp_path / "legacy"
    registry_dir = tmp_path / "registry"
    _write_legacy_artifact(legacy_dir, "dry_key")

    report = migrate(legacy_dir, registry_dir, dry_run=True)

    assert [i.key for i in report.planned] == ["dry_key"]
    assert not registry_dir.exists() or not any(registry_dir.iterdir())


# ── orphans are never migrated ────────────────────────────────────────────────


def test_orphans_are_not_migrated(tmp_path):
    legacy_dir = tmp_path / "legacy"
    registry_dir = tmp_path / "registry"
    (legacy_dir).mkdir(parents=True)
    (legacy_dir / "lonely.pkl").write_bytes(pickle.dumps({"x": 1}))

    report = migrate(legacy_dir, registry_dir)

    assert report.orphans and report.orphans[0].key == "lonely"
    assert report.migrated == []
    assert not registry_dir.exists() or not any(registry_dir.iterdir())


# ── artifact id sanitization ──────────────────────────────────────────────────


def test_artifact_id_is_sanitized_and_mapping_recorded(tmp_path):
    legacy_dir = tmp_path / "legacy"
    registry_dir = tmp_path / "registry"
    bad_key = "bad key:name"
    _write_legacy_artifact(legacy_dir, bad_key)

    report = migrate(legacy_dir, registry_dir)

    item = report.migrated[0]
    assert item.key == bad_key
    assert item.artifact_id != bad_key
    assert item.artifact_id[0].isalnum()
    assert any(bad_key in w for w in item.warnings)
    assert (registry_dir / item.artifact_id / "artifact.json").exists()


# ── unpicklable model (missing optional lib) is skipped, not crashed ─────────


def test_unpicklable_model_is_skipped_not_crashed(tmp_path):
    legacy_dir = tmp_path / "legacy"
    registry_dir = tmp_path / "registry"
    legacy_dir.mkdir(parents=True)

    # A pickle bytestream referencing a module that genuinely doesn't exist
    # (stands in for an artifact needing an uninstalled optional dependency
    # like xgboost/torch): pickled under a fake module that's removed from
    # sys.modules again immediately, same trick as _pickle_bytes_under_legacy_module.
    broken_payload = _pickle_bytes_under_legacy_module(_fit_polyreg_distance(), "totally_missing_module")
    (legacy_dir / "broken_key.pkl").write_bytes(broken_payload)
    (legacy_dir / "broken_key_meta.json").write_text(
        json.dumps({"model_type": "xgboost", "model_key": "broken_key", "saved_at": "2026-01-01T00:00:00"}),
        encoding="utf-8",
    )
    # And a healthy sibling to confirm the run continues past the failure.
    _write_legacy_artifact(legacy_dir, "healthy_key")

    report = migrate(legacy_dir, registry_dir)

    assert report.ok  # a skip is not a failure
    broken = [i for i in report.skipped if i.key == "broken_key"]
    assert len(broken) == 1
    assert "unpickle" in broken[0].reason
    assert [i.key for i in report.migrated] == ["healthy_key"]
