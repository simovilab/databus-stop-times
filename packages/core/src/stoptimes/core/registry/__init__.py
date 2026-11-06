"""Model registry backends behind one interface (`RegistryBackend`); `open_registry` picks one.

    from stoptimes.core.registry import open_registry
    reg = open_registry()                      # STOPTIMES_REGISTRY, else the configured directory
    reg = open_registry("artifacts/models")    # a directory
    for meta in reg.list(): ...
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import TypeAlias

from stoptimes.core.registry.base import ArtifactExistsError, RegistryBackend, RegistryError, SkipHandler
from stoptimes.core.registry.filesystem import FilesystemRegistry

__all__ = [
    "ArtifactExistsError",
    "FilesystemRegistry",
    "REGISTRY_ENV",
    "RegistryBackend",
    "RegistryError",
    "RegistryLike",
    "SkipHandler",
    "open_registry",
]

#: Selects the registry when no explicit one is given: a directory, or `mlflow+<tracking uri>`.
REGISTRY_ENV = "STOPTIMES_REGISTRY"

MLFLOW_SCHEME = "mlflow+"

RegistryLike: TypeAlias = RegistryBackend | Path | str


def open_registry(spec: RegistryLike | None = None) -> RegistryBackend:
    """The registry for `spec`.

    - a `RegistryBackend` is returned as is;
    - `None`: `$STOPTIMES_REGISTRY` if set, else `core.config`'s `model_registry_dir`
      (which honors `STOPTIMES_MODEL_REGISTRY_DIR`);
    - `"mlflow+<tracking uri>"`: the MLflow backend (`stoptimes-core[mlflow]`), one shared instance
      per URI;
    - anything else is a directory path.
    """
    if isinstance(spec, RegistryBackend) and not isinstance(spec, (str, Path)):
        return spec
    if spec is None:
        env = os.environ.get(REGISTRY_ENV, "").strip()
        if env:
            spec = env
        else:
            from stoptimes.core.config import get_config

            return FilesystemRegistry(get_config().model_registry_dir)
    if isinstance(spec, str) and spec.startswith(MLFLOW_SCHEME):
        return _mlflow_registry(spec[len(MLFLOW_SCHEME) :])
    return FilesystemRegistry(spec)


_MLFLOW_REGISTRIES: dict[str, RegistryBackend] = {}
_MLFLOW_LOCK = threading.Lock()


def _mlflow_registry(tracking_uri: str) -> RegistryBackend:
    """One `MlflowRegistry` per tracking URI per process, so its listing cache and downloads are shared."""
    if not tracking_uri:
        raise RegistryError("registry 'mlflow+': missing tracking URI, e.g. mlflow+https://ml.example.org")
    with _MLFLOW_LOCK:
        registry = _MLFLOW_REGISTRIES.get(tracking_uri)
        if registry is None:
            from stoptimes.core.registry.mlflow_backend import MlflowRegistry

            registry = _MLFLOW_REGISTRIES[tracking_uri] = MlflowRegistry(tracking_uri)
        return registry
