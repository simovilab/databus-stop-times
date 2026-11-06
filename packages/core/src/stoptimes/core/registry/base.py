"""The model-registry seam: one interface every reader and writer of artifacts goes through.

A registry stores `core.contracts.artifact` artifacts (`<artifact_id>/{artifact.json, model.<ext>,
card.md, assets...}`). Callers never touch the storage layout themselves: they list metadata, ask for
a local directory to load from, and write through `staged`. That is what lets a second backend
(MLflow, docs/MLFLOW_INTEGRATION.md §0) slot in without changing serve, val or train.

Not to be confused with `stoptimes.core.features.registry` (the feature registry) or the legacy
pickle `stoptimes.models.common.registry.ModelRegistry`.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol, runtime_checkable

from stoptimes.core.contracts.artifact import ArtifactMetadata

#: Called with (entry name, exception) for each artifact `list(strict=False)` skips.
SkipHandler = Callable[[str, Exception], None]


class RegistryError(RuntimeError):
    """A registry could not be opened or an operation on it failed."""


class ArtifactExistsError(RegistryError, FileExistsError):
    """`staged` was asked to write an artifact id that already exists, without `overwrite`."""


@runtime_checkable
class RegistryBackend(Protocol):
    """Storage for artifacts. Implementations: `FilesystemRegistry` (default); MLflow (planned)."""

    @property
    def uri(self) -> str:
        """Human-readable location, used in messages and as the `STOPTIMES_REGISTRY` value."""
        ...

    def list(self, *, strict: bool = False, on_skip: SkipHandler | None = None) -> list[ArtifactMetadata]:
        """Metadata of every artifact, sorted by artifact id, without loading any model.

        A malformed entry raises when `strict`, otherwise it is skipped (reported to `on_skip`).
        """
        ...

    def get(self, artifact_id: str) -> ArtifactMetadata | None:
        """One artifact's metadata, or None when it doesn't exist."""
        ...

    def exists(self, artifact_id: str) -> bool:
        """Whether an artifact with this id exists."""
        ...

    def describe(self, artifact_id: str) -> str:
        """Where this artifact lives (or would live), for messages."""
        ...

    def local_root(self, meta: ArtifactMetadata) -> Path:
        """A local directory that contains `<meta.artifact_id>/`, ready to load from.

        A remote backend downloads (and caches) the artifact first; the filesystem backend
        returns its own root.
        """
        ...

    def card(self, artifact_id: str) -> str | None:
        """The artifact's `card.md` text, or None when it has none."""
        ...

    def staged(self, artifact_id: str, *, overwrite: bool = False) -> AbstractContextManager[Path]:
        """Write a new artifact: yields an empty directory; on a clean exit its contents become
        `<artifact_id>` in the registry, atomically replacing an existing one when `overwrite`.
        On an exception nothing is published.
        """
        ...
