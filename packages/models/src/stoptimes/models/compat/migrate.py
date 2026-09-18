"""Migrate a legacy gtfs-eta model registry into the artifact contract layout.

The pre-migration registry (`stoptimes.models.common.registry.ModelRegistry`,
née gtfs-eta) is a flat directory::

    {model_key}.pkl        # pickled model, class module path pre-packages/ move
    {model_key}_meta.json  # model_type, dataset, hyperparameters, features,
                            # n_samples/*_samples, metrics, provenance (newer runs only)
    registry.json           # index (not needed here -- we walk the directory itself)

`migrate()` reads each `{key}.pkl` + `{key}_meta.json` pair, unpickles the
model with `LegacyModuleUnpickler` (remapping pre-`packages/` module paths),
re-pickles it under its now-current module path so a later `pickle.load`
needs no shim, and writes `artifact.json` + `model.pkl` under
`<registry>/<artifact_id>/` per `stoptimes.core.contracts.artifact`.

Deliberately dependency-light: only `pickle` + the stdlib + the artifact
contract (pydantic) are imported at module load time, so
`stoptimes models migrate --help` doesn't pull in pandas/sklearn/xgboost --
those only get imported transitively when a legacy model's class is
unpickled.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from stoptimes.core.contracts.artifact import (
    ModelScope,
    ScopeLevel,
    ArtifactMetadata,
    DataWindow,
    ModelFormat,
    Provenance,
    artifact_dir,
    write_metadata,
)

from stoptimes.models.common.legacy_unpickler import LegacyModuleUnpickler

_ARTIFACT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._=-]*$")
_META_SUFFIX = "_meta.json"

# Top-level `{key}_meta.json` keys that are bookkeeping/derived facts already
# extracted elsewhere by `_build_metadata` (family, features, data window,
# metrics, provenance) -- everything ELSE at the top level is treated as a
# hyperparameter. This is a closed list of "definitely not a hyperparam"
# keys, not an allowlist of known hyperparameter names, so a new model
# family's hyperparameters (whatever they're called) are picked up without
# this module needing to know about them.
_NON_HYPERPARAM_KEYS = frozenset(
    {
        "model_type",
        "dataset",
        "route_id",
        "features",
        "n_samples",
        "train_samples",
        "test_samples",
        "val_samples",
        "n_trips",
        "metrics",
        "model_key",
        "saved_at",
        "model_path",
        "provenance",
    }
)


@dataclass(frozen=True)
class MigrationItem:
    """Outcome (planned or actual) for one legacy `{key}` pair."""

    key: str
    status: str  # "planned", "dry_run", "migrated", "skipped", "failed", "orphan"
    artifact_id: str | None = None
    reason: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class MigrationReport:
    items: tuple[MigrationItem, ...]

    @property
    def migrated(self) -> list[MigrationItem]:
        return [i for i in self.items if i.status == "migrated"]

    @property
    def planned(self) -> list[MigrationItem]:
        return [i for i in self.items if i.status in ("planned", "dry_run")]

    @property
    def skipped(self) -> list[MigrationItem]:
        return [i for i in self.items if i.status == "skipped"]

    @property
    def failed(self) -> list[MigrationItem]:
        return [i for i in self.items if i.status == "failed"]

    @property
    def orphans(self) -> list[MigrationItem]:
        return [i for i in self.items if i.status == "orphan"]

    @property
    def ok(self) -> bool:
        """No item failed outright (skips and orphans are not failures)."""
        return not self.failed


def plan_migration(legacy_dir: str | Path) -> list[MigrationItem]:
    """Pair up `{key}.pkl` / `{key}_meta.json` files under `legacy_dir`.

    Does not touch the filesystem beyond listing it. A `.pkl` without a
    matching `_meta.json` (or vice versa) is reported as an "orphan" item,
    never migrated.
    """
    legacy_dir = Path(legacy_dir)
    meta_keys = {p.name[: -len(_META_SUFFIX)] for p in legacy_dir.glob(f"*{_META_SUFFIX}")}
    pkl_keys = {p.stem for p in legacy_dir.glob("*.pkl")}

    items: list[MigrationItem] = []
    for key in sorted(meta_keys | pkl_keys):
        has_meta = key in meta_keys
        has_pkl = key in pkl_keys
        if not (has_meta and has_pkl):
            missing = "metadata (*_meta.json)" if not has_meta else "model (*.pkl)"
            items.append(MigrationItem(key=key, status="orphan", reason=f"missing {missing} file"))
            continue

        artifact_id, warnings = _sanitize_artifact_id(key)
        items.append(
            MigrationItem(key=key, status="planned", artifact_id=artifact_id, warnings=tuple(warnings))
        )
    return items


def migrate(
    legacy_dir: str | Path,
    registry_dir: str | Path,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
) -> MigrationReport:
    """Migrate every legacy artifact under `legacy_dir` into `registry_dir`.

    Idempotent: an artifact whose destination directory already exists is
    reported "skipped" unless `overwrite=True`. `dry_run=True` reports what
    would happen (status "dry_run") without writing anything.
    """
    legacy_dir = Path(legacy_dir)
    registry_dir = Path(registry_dir)

    results: list[MigrationItem] = []
    for item in plan_migration(legacy_dir):
        if item.status == "orphan":
            results.append(item)
            continue
        results.append(
            _migrate_one(item, legacy_dir, registry_dir, dry_run=dry_run, overwrite=overwrite)
        )
    return MigrationReport(items=tuple(results))


def _migrate_one(
    item: MigrationItem,
    legacy_dir: Path,
    registry_dir: Path,
    *,
    dry_run: bool,
    overwrite: bool,
) -> MigrationItem:
    key = item.key
    artifact_id = item.artifact_id
    assert artifact_id is not None
    warnings = list(item.warnings)

    target_dir = artifact_dir(registry_dir, artifact_id)
    if target_dir.exists() and not overwrite:
        return MigrationItem(
            key=key,
            status="skipped",
            artifact_id=artifact_id,
            reason="artifact already exists at destination (use --overwrite)",
            warnings=tuple(warnings),
        )

    meta_path = legacy_dir / f"{key}{_META_SUFFIX}"
    pkl_path = legacy_dir / f"{key}.pkl"

    try:
        legacy_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - report, never crash the run
        return MigrationItem(
            key=key,
            status="failed",
            artifact_id=artifact_id,
            reason=f"could not read metadata: {exc!r}",
            warnings=tuple(warnings),
        )

    try:
        with open(pkl_path, "rb") as fh:
            model = LegacyModuleUnpickler(fh).load()
    except Exception as exc:  # noqa: BLE001 - e.g. missing optional lib (xgboost/torch)
        return MigrationItem(
            key=key,
            status="skipped",
            artifact_id=artifact_id,
            reason=f"could not unpickle model: {exc!r}",
            warnings=tuple(warnings),
        )

    try:
        meta = _build_metadata(artifact_id, legacy_meta, warnings)
    except Exception as exc:  # noqa: BLE001
        return MigrationItem(
            key=key,
            status="failed",
            artifact_id=artifact_id,
            reason=f"could not build artifact metadata: {exc!r}",
            warnings=tuple(warnings),
        )

    if dry_run:
        return MigrationItem(key=key, status="dry_run", artifact_id=artifact_id, warnings=tuple(warnings))

    try:
        _write_artifact(model, meta, registry_dir, target_dir)
    except Exception as exc:  # noqa: BLE001
        return MigrationItem(
            key=key,
            status="failed",
            artifact_id=artifact_id,
            reason=f"could not write artifact: {exc!r}",
            warnings=tuple(warnings),
        )

    return MigrationItem(key=key, status="migrated", artifact_id=artifact_id, warnings=tuple(warnings))


def _write_artifact(model: Any, meta: ArtifactMetadata, registry_dir: Path, target_dir: Path) -> None:
    """Write `model.pkl` + `artifact.json` atomically: build in a temp
    sibling directory, verify it, then rename into place."""
    registry_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(dir=registry_dir, prefix=f".migrate-{meta.artifact_id}-"))
    try:
        model_path = tmp_dir / meta.model_filename
        with open(model_path, "wb") as fh:
            pickle.dump(model, fh)

        # The whole point of re-pickling is that a later, plain `pickle.load`
        # (no `LegacyModuleUnpickler` shim) can read it -- verify that now,
        # while we can still fail the migration instead of shipping a bad
        # artifact.
        with open(model_path, "rb") as fh:
            pickle.load(fh)

        write_metadata(meta, tmp_dir)

        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        os.rename(tmp_dir, target_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def _build_metadata(artifact_id: str, legacy_meta: dict[str, Any], warnings: list[str]) -> ArtifactMetadata:
    family = str(legacy_meta.get("model_type") or "unknown")
    features = list(legacy_meta.get("features") or [])
    params = {k: v for k, v in legacy_meta.items() if k not in _NON_HYPERPARAM_KEYS}

    legacy_provenance = legacy_meta.get("provenance") or {}
    if not isinstance(legacy_provenance, dict):
        legacy_provenance = {}

    created_at_raw = legacy_provenance.get("created_at") or legacy_meta.get("saved_at")
    if not created_at_raw:
        raise ValueError(f"no provenance.created_at or saved_at in metadata for {artifact_id!r}")
    created_at = _parse_datetime(created_at_raw)

    lib_versions = legacy_provenance.get("lib_versions") or {}
    if not lib_versions:
        warnings.append("no lib_versions in legacy provenance -- left empty, not invented")

    provenance = Provenance(
        created_at=created_at,
        git_sha=legacy_provenance.get("git_sha"),
        python_version=legacy_provenance.get("python_version"),
        lib_versions=lib_versions,
        seed=legacy_provenance.get("seed"),
    )

    metrics_raw = legacy_meta.get("metrics") or {}
    metrics = {
        k: v
        for k, v in metrics_raw.items()
        if v is None or (isinstance(v, (int, float)) and not isinstance(v, bool))
    }

    data = DataWindow(
        dataset=legacy_meta.get("dataset"),
        dataset_sha256=legacy_provenance.get("dataset_sha256"),
        n_samples=legacy_meta.get("n_samples"),
        n_train=legacy_meta.get("train_samples"),
        n_test=legacy_meta.get("test_samples"),
    )

    # Legacy metadata carried the route as a bare `route_id` (None/"global" =
    # a model for every route); the contract expresses that as a ModelScope.
    legacy_route = legacy_meta.get("route_id")
    scope = (
        ModelScope(level=ScopeLevel.ROUTE, route_ids=[str(legacy_route)])
        if legacy_route not in (None, "", "global")
        else ModelScope()
    )

    return ArtifactMetadata(
        artifact_id=artifact_id,
        family=family,
        scope=scope,
        features=features,
        params=params,
        data=data,
        provenance=provenance,
        metrics=metrics,
        tags=["migrated"],
        format=ModelFormat.PICKLE,
        legacy=legacy_meta,
    )


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _sanitize_artifact_id(key: str) -> tuple[str, list[str]]:
    """Coerce a legacy `model_key` into the contract's `artifact_id` pattern
    (`^[A-Za-z0-9][A-Za-z0-9._=-]*$`), recording the mapping if it changed.

    Every real pre-migration `model_key` observed in gtfs-eta's
    `models/trained/` already matches this pattern (it's built from
    `[A-Za-z0-9_=-]` segments joined by `_`), so this is a safety net for
    keys from other sources, not the common case.
    """
    if _ARTIFACT_ID_PATTERN.match(key):
        return key, []

    sanitized = re.sub(r"[^A-Za-z0-9._=-]", "-", key)
    sanitized = sanitized.lstrip("._=-")
    if not sanitized:
        sanitized = "artifact"
    if not sanitized[0].isalnum():
        sanitized = f"a-{sanitized}"

    return sanitized, [f"sanitized artifact id: {key!r} -> {sanitized!r}"]
