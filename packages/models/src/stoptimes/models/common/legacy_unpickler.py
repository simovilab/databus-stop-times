"""
Legacy pickle compatibility: remap pre-migration module paths.

Model artifacts pickled before the `packages/` migration record their
class's module path as it was in the pre-migration `gtfs-eta` repo layout
-- e.g. `models.xgb.train`, `models.polyreg_distance.train`, `core.config`,
`eta_service.estimator`, `feature_engineering.temporal`. After the
migration those modules live under `stoptimes.*` (see
docs/MIGRATION_MAP.md), so a stock `pickle.load()` on an old artifact
raises `ModuleNotFoundError` -- the module name baked into the pickle
bytestream no longer exists.

`LegacyModuleUnpickler` is a `pickle.Unpickler` subclass that remaps a
legacy module name to its `stoptimes.*` home in `find_class`, so an old
artifact still unpickles correctly, and it's a no-op for artifacts already
pickled under `stoptimes.*` module names. It's wired into
`ModelRegistry.load_model` only -- deliberately NOT installed as a global
`sys.meta_path`/`sys.modules` hook, which would silently rewrite imports
for the whole process, well beyond model loading.
"""

from __future__ import annotations

import pickle
from typing import Any, List

# gtfs-eta's own trainers ran with `sys.path.append(<models dir>)` (see e.g.
# the pre-migration models/xgb/train.py), so real artifacts saved that way
# record *bare* module names like "xgb.train" / "historical_mean.train" --
# not "models.xgb.train" -- confirmed 2026-09 against real artifacts at
# gtfs-eta/models/trained/ (see docs/notes/p0-a.md for the manual check).
# Both forms are remapped to the same stoptimes.models.<family> home.
_BARE_MODEL_FAMILY_MODULES = {
    "common",
    "historical_mean",
    "polyreg_distance",
    "polyreg_time",
    "ewma",
    "xgb",
    "lstm",
}


def remap_legacy_module(module: str) -> str:
    """
    Return `module`'s `stoptimes.*` home if it matches a known
    pre-migration (gtfs-eta repo) module, else `module` unchanged.

    Matching is by leading dotted *segment*, not raw string prefix --
    `feature_engineering.navsat_reader` must match the `navsat_*` rule even
    though "navsat_reader" isn't "navsat" followed by a dot. Modules
    already under `stoptimes.*` (or any other module a pickled object's
    state happens to reference, e.g. numpy internals) don't match any
    legacy module and pass through untouched.

    Mirrors the gtfs-eta table in docs/MIGRATION_MAP.md:
        core/*.py                          -> stoptimes.core.*
        feature_engineering/spatial.py      -> stoptimes.core.spatial.geo
        feature_engineering/temporal.py     -> stoptimes.core.features.temporal
        feature_engineering/segment_core.py -> stoptimes.core.segments
        feature_engineering/navsat_*.py     -> stoptimes.train.sources.navsat_*
        feature_engineering/bucr_*.py       -> stoptimes.train.sources.bucr_*
        feature_engineering/*.py (rest)     -> stoptimes.train.datasets.*
        models/common/, models/<family>/    -> stoptimes.models.*
        models/evaluation/                  -> stoptimes.train.evaluation.*
        models/train_all_models.py          -> stoptimes.train.train_all_models
        eta_service/                        -> stoptimes.serve.*
    """
    parts = module.split(".")
    head, rest = parts[0], parts[1:]

    if head == "feature_engineering":
        if not rest:
            return module
        first, tail = rest[0], rest[1:]
        if first == "spatial":
            new_parts: List[str] = ["stoptimes", "core", "spatial", "geo"]
        elif first == "temporal":
            new_parts = ["stoptimes", "core", "features", "temporal"]
        elif first == "segment_core":
            new_parts = ["stoptimes", "core", "segments"]
        elif first.startswith("navsat") or first.startswith("bucr"):
            new_parts = ["stoptimes", "train", "sources", first]
        else:
            new_parts = ["stoptimes", "train", "datasets", first]
        return ".".join(new_parts + tail)

    if head == "models":
        if rest[:1] == ["evaluation"]:
            return ".".join(["stoptimes", "train", "evaluation"] + rest[1:])
        if rest[:1] == ["train_all_models"]:
            return ".".join(["stoptimes", "train", "train_all_models"] + rest[1:])
        return ".".join(["stoptimes", "models"] + rest)

    if head == "eta_service":
        return ".".join(["stoptimes", "serve"] + rest)

    if head == "core":
        return ".".join(["stoptimes", "core"] + rest)

    # Bare (unprefixed) legacy forms -- see _BARE_MODEL_FAMILY_MODULES.
    if head == "evaluation":
        return ".".join(["stoptimes", "train", "evaluation"] + rest)
    if head == "train_all_models":
        return ".".join(["stoptimes", "train", "train_all_models"] + rest)
    if head in _BARE_MODEL_FAMILY_MODULES:
        return ".".join(["stoptimes", "models", head] + rest)

    return module


class LegacyModuleUnpickler(pickle.Unpickler):
    """
    `pickle.Unpickler` that transparently remaps pre-migration module
    paths (see `remap_legacy_module`) when resolving a pickled class.

    Falls back to the original, unmapped module name if the remapped
    import doesn't actually resolve (module missing, or the class isn't
    present under it) -- so the error a caller sees is about the real
    (legacy) name, not a remapped one that never existed.
    """

    def find_class(self, module: str, name: str) -> Any:
        remapped = remap_legacy_module(module)
        if remapped != module:
            try:
                return super().find_class(remapped, name)
            except (ImportError, AttributeError):
                pass
        return super().find_class(module, name)
