"""In-process model caching in ModelRegistry.load_model.

Without it, load_model unpickled from disk on every call, so a serving path
calling it per prediction paid model-load cost per prediction -- measured at
23-109 ms p50 through eta_service.estimator, dominated by loading rather than
inference (models/evaluation/cost.py).

Caching a mutable object changes behaviour: callers now share one instance
instead of each getting a private copy. These tests pin both the speed-up and
the invalidation rule that makes it safe.
"""
from __future__ import annotations

import pickle
import time

import numpy as np
import pytest

from models.common.registry import ModelRegistry


class _Dummy:
    def __init__(self, value: int = 1) -> None:
        self.value = value


@pytest.fixture()
def registry(tmp_path):
    reg = ModelRegistry(base_dir=str(tmp_path))
    reg.save_model("k", _Dummy(1), {"model_type": "dummy", "metrics": {}})
    return reg


def test_second_load_returns_the_same_object(registry):
    assert registry.load_model("k") is registry.load_model("k")


def test_use_cache_false_returns_a_fresh_object(registry):
    first = registry.load_model("k")
    fresh = registry.load_model("k", use_cache=False)
    assert fresh is not first
    assert fresh.value == first.value


def test_clear_cache_forces_a_reload(registry):
    first = registry.load_model("k")
    registry.clear_cache()
    assert registry.load_model("k") is not first


def test_replacing_the_artifact_invalidates_the_cache(registry, tmp_path):
    first = registry.load_model("k")
    assert first.value == 1

    # Rewrite the .pkl in place, as a retrain-and-resave would.
    path = registry._resolve_artifact_path(registry.registry["k"]["model_path"])
    time.sleep(0.01)  # ensure a distinct mtime_ns
    with open(path, "wb") as f:
        pickle.dump(_Dummy(2), f)

    reloaded = registry.load_model("k")
    assert reloaded.value == 2, "a replaced artifact must not be served from cache"


def test_cache_is_faster_than_reloading(registry):
    registry.load_model("k")  # warm

    t0 = time.perf_counter()
    for _ in range(20):
        registry.load_model("k")
    cached = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(20):
        registry.load_model("k", use_cache=False)
    uncached = time.perf_counter() - t0

    assert cached < uncached, (
        f"cached path ({cached:.4f}s) should beat reloading ({uncached:.4f}s)"
    )


def test_missing_key_still_raises(registry):
    with pytest.raises(KeyError):
        registry.load_model("nope")
