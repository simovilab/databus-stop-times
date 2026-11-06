"""`mlflow:<model>@<alias>` and `mlflow:<model>/<version>` predictors.

See docs/MLFLOW_INTEGRATION.md §0 D9.

The ref is resolved to one model version **once, when the run starts**: a run that began on
`stoptimes-mbta@staging` keeps scoring that version even if the alias moves mid-run, so every
prediction in a run comes from one model. The version is then scored exactly like a
`stoptimes:<artifact_id>` predictor (same loading, drift check, live session).

The MLflow server is `$STOPTIMES_MLFLOW` (e.g. `https://ml.simovi.org`), else
`$STOPTIMES_REGISTRY` when that is an `mlflow+` URI. Credentials come from MLflow's own env
vars or `~/.mlflow/credentials`.
"""

from __future__ import annotations

import logging
from typing import Any

from stoptimes.core.registry import MLFLOW_ENV, RegistryError, mlflow_registry
from stoptimes.val.predictors.artifact import StoptimesArtifactPredictor

log = logging.getLogger(__name__)

PREFIX = "mlflow:"


def mlflow_predictor_ids(versions: list[Any]) -> list[str]:
    """Predictor ids for registry versions: `<name>@<alias>` per alias, then `<name>/<version>`.

    Aliases come first (they are what one normally picks); versions follow, newest first per model.
    """
    by_alias: list[str] = []
    by_version: list[str] = []
    for v in sorted(versions, key=lambda v: (v.name, -v.version)):
        by_alias.extend(f"{PREFIX}{v.name}@{alias}" for alias in v.aliases)
        by_version.append(f"{PREFIX}{v.name}/{v.version}")
    return sorted(by_alias) + by_version


def build_mlflow_predictor(ref: str) -> StoptimesArtifactPredictor:
    """The predictor for `mlflow:<ref>`.

    Raises `RegistryError` if no server is configured or *ref* is unknown.
    """
    registry = mlflow_registry()
    if registry is None:
        raise RegistryError(
            f"predictor 'mlflow:{ref}' needs an MLflow server: set {MLFLOW_ENV}=https://<host>"
        )
    version = registry.resolve(ref)  # type: ignore[attr-defined]
    log.info(
        "mlflow:%s -> %s v%s (artifact %s)", ref, version.name, version.version, version.artifact_id
    )
    predictor = StoptimesArtifactPredictor(
        version.artifact_id,
        registry,
        artifact=version.meta,
        predictor_id=f"{PREFIX}{ref}",
    )
    predictor.resolved_version = f"{version.name}/{version.version}"
    return predictor
