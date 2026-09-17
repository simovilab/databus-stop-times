"""
Tests for legacy pickle compatibility (roadmap: packages/ migration).

Artifacts pickled before the migration to `packages/*` record their class's
module path as it was in the pre-migration gtfs-eta repo layout, e.g.
`models.polyreg_distance.train`, `core.config`, `eta_service.estimator`,
`feature_engineering.temporal`. After the migration those modules live
under `stoptimes.*` (see docs/MIGRATION_MAP.md). These tests confirm
`stoptimes.models.common.legacy_unpickler.remap_legacy_module` maps every
documented legacy prefix correctly, and that `ModelRegistry.load_model`
(which uses `LegacyModuleUnpickler` internally, not a global import hook)
can load an artifact pickled under a spoofed legacy module name.

Run with: pytest packages/models/tests/test_legacy_pickle_compat.py -v
"""

from __future__ import annotations

import pickle
import sys
import types
from typing import Any

import numpy as np
import pandas as pd
import pytest

from stoptimes.models.common.legacy_unpickler import (
    LegacyModuleUnpickler,
    remap_legacy_module,
)
from stoptimes.models.common.registry import ModelRegistry
from stoptimes.models.polyreg_distance.train import PolyRegDistanceModel


class TestRemapLegacyModule:
    """Every legacy prefix named in the P0-A task / docs/MIGRATION_MAP.md
    resolves to its documented stoptimes.* home."""

    @pytest.mark.parametrize(
        "legacy, expected",
        [
            ("models.xgb.train", "stoptimes.models.xgb.train"),
            ("models.polyreg_distance.train", "stoptimes.models.polyreg_distance.train"),
            ("models.common.registry", "stoptimes.models.common.registry"),
            ("models.evaluation.leaderboard", "stoptimes.train.evaluation.leaderboard"),
            ("models.train_all_models", "stoptimes.train.train_all_models"),
            ("core.config", "stoptimes.core.config"),
            ("core.logging", "stoptimes.core.logging"),
            ("eta_service.estimator", "stoptimes.serve.estimator"),
            ("eta_service", "stoptimes.serve"),
            ("feature_engineering.spatial", "stoptimes.core.spatial.geo"),
            ("feature_engineering.temporal", "stoptimes.core.features.temporal"),
            ("feature_engineering.segment_core", "stoptimes.core.segments"),
            ("feature_engineering.navsat_reader", "stoptimes.train.sources.navsat_reader"),
            ("feature_engineering.bucr_holidays", "stoptimes.train.sources.bucr_holidays"),
            ("feature_engineering.weather", "stoptimes.train.datasets.weather"),
            # Bare (unprefixed) forms: gtfs-eta's own trainers ran with
            # `sys.path.append(<models dir>)`, so real artifacts on disk
            # record these, not the "models.*"-prefixed form -- confirmed
            # against the real registry at gtfs-eta/models/trained/ (see
            # docs/notes/p0-a.md).
            ("xgb.train", "stoptimes.models.xgb.train"),
            ("historical_mean.train", "stoptimes.models.historical_mean.train"),
            ("polyreg_distance.train", "stoptimes.models.polyreg_distance.train"),
            ("polyreg_time.train", "stoptimes.models.polyreg_time.train"),
            ("ewma.train", "stoptimes.models.ewma.train"),
            ("lstm.train", "stoptimes.models.lstm.train"),
            ("common.registry", "stoptimes.models.common.registry"),
            ("evaluation.leaderboard", "stoptimes.train.evaluation.leaderboard"),
            ("train_all_models", "stoptimes.train.train_all_models"),
        ],
    )
    def test_known_legacy_prefixes_remap(self, legacy, expected):
        assert remap_legacy_module(legacy) == expected

    def test_already_current_module_is_unchanged(self):
        assert remap_legacy_module("stoptimes.models.xgb.train") == "stoptimes.models.xgb.train"

    def test_unrelated_module_is_unchanged(self):
        # A pickled object's state can reference arbitrary third-party
        # modules (numpy, sklearn, ...) that must never be touched.
        assert remap_legacy_module("numpy.core.multiarray") == "numpy.core.multiarray"


def _pickle_bytes_under_legacy_module(obj: Any, legacy_module: str) -> bytes:
    """Serialize `obj` with its class's `__module__` temporarily spoofed to
    `legacy_module`, reproducing exactly what an artifact saved by the
    pre-migration code (where that module really was the class's home)
    looks like on disk today.

    Spoofing `__module__` alone isn't enough: pickle's `save_global` does a
    real `__import__(module)` to verify the name resolves, which walks every
    parent package too (importing "models.polyreg_distance.train" imports
    "models" and "models.polyreg_distance" along the way) -- so a fake
    module object is registered in `sys.modules` under the leaf name *and*
    every parent, for the duration of the dump, mirroring the class having
    actually lived there without needing a real legacy package on disk."""
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


def _fit_polyreg_distance() -> PolyRegDistanceModel:
    model = PolyRegDistanceModel(degree=1, alpha=1.0, route_specific=False)
    distances = np.linspace(0, 3000, 50)
    train_df = pd.DataFrame(
        {
            "distance_to_stop": distances,
            "time_to_arrival_seconds": distances / 4.5,
        }
    )
    model.fit(train_df)
    return model


class TestLegacyModuleUnpicklerDirect:
    @pytest.mark.parametrize(
        "legacy_module",
        [
            "models.polyreg_distance.train",  # the "models.*"-prefixed form
            "polyreg_distance.train",  # the bare form real artifacts actually use
        ],
    )
    def test_loads_object_pickled_under_spoofed_legacy_module(self, legacy_module):
        model = _fit_polyreg_distance()
        payload = _pickle_bytes_under_legacy_module(model, legacy_module)

        import io

        loaded = LegacyModuleUnpickler(io.BytesIO(payload)).load()

        assert type(loaded).__module__ == "stoptimes.models.polyreg_distance.train"
        assert type(loaded).__name__ == "PolyRegDistanceModel"


class TestRegistryLoadsLegacyPickledArtifact:
    """The scenario that matters end to end: a model saved by the
    pre-migration code (class module = a legacy name) sitting in a registry
    directory today must still load through ModelRegistry.load_model."""

    def test_registry_load_model_handles_legacy_module_pickle(self, tmp_path):
        model = _fit_polyreg_distance()
        preds_before = model.predict(pd.DataFrame({"distance_to_stop": [100.0, 500.0, 1000.0]}))

        registry = ModelRegistry(base_dir=str(tmp_path))
        saved_path = registry.save_model(
            "legacy_pin_test",
            model,
            {"model_type": "polyreg_distance", "dataset": "unit_test", "route_id": None},
        )

        # Overwrite the artifact on disk with a version pickled under the
        # pre-migration module name -- everything else about the registry
        # entry (registry.json, meta.json) is exactly what save_model wrote.
        legacy_payload = _pickle_bytes_under_legacy_module(model, "models.polyreg_distance.train")
        saved_path.write_bytes(legacy_payload)
        registry.clear_cache()

        loaded = registry.load_model("legacy_pin_test", use_cache=False)

        assert type(loaded).__module__ == "stoptimes.models.polyreg_distance.train"
        preds_after = loaded.predict(pd.DataFrame({"distance_to_stop": [100.0, 500.0, 1000.0]}))
        np.testing.assert_allclose(preds_before, preds_after)
