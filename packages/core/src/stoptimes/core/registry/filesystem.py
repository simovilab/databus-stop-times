"""`FilesystemRegistry`: artifacts as subdirectories of one root directory (the default backend)."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from stoptimes.core.contracts.artifact import (
    METADATA_FILENAME,
    ArtifactMetadata,
    artifact_dir,
    read_metadata,
)
from stoptimes.core.registry.base import ArtifactExistsError, SkipHandler

CARD_FILENAME = "card.md"


class FilesystemRegistry:
    """`<root>/<artifact_id>/{artifact.json, model.<ext>, card.md, assets...}`.

    The root need not exist: listing an absent root returns nothing, and the first write creates it.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser()

    def __repr__(self) -> str:
        return f"FilesystemRegistry({str(self.root)!r})"

    @property
    def uri(self) -> str:
        return str(self.root)

    def list(self, *, strict: bool = False, on_skip: SkipHandler | None = None) -> list[ArtifactMetadata]:
        if not self.root.is_dir():
            return []
        metadatas: list[ArtifactMetadata] = []
        for child in sorted(self.root.iterdir()):
            if not (child / METADATA_FILENAME).is_file():
                continue
            try:
                metadatas.append(read_metadata(child))
            except Exception as exc:  # noqa: BLE001 - malformed artifact.json: caller decides via strict
                if strict:
                    raise
                if on_skip is not None:
                    on_skip(child.name, exc)
        return metadatas

    def get(self, artifact_id: str) -> ArtifactMetadata | None:
        directory = artifact_dir(self.root, artifact_id)
        if not (directory / METADATA_FILENAME).is_file():
            return None
        return read_metadata(directory)

    def exists(self, artifact_id: str) -> bool:
        return artifact_dir(self.root, artifact_id).exists()

    def describe(self, artifact_id: str) -> str:
        return str(artifact_dir(self.root, artifact_id))

    def local_root(self, meta: ArtifactMetadata) -> Path:
        return self.root

    def card(self, artifact_id: str) -> str | None:
        path = artifact_dir(self.root, artifact_id) / CARD_FILENAME
        return path.read_text(encoding="utf-8") if path.is_file() else None

    @contextmanager
    def staged(self, artifact_id: str, *, overwrite: bool = False) -> Iterator[Path]:
        target = artifact_dir(self.root, artifact_id)
        if target.exists() and not overwrite:
            raise ArtifactExistsError(f"artifact {artifact_id!r} already exists at {target}")
        self.root.mkdir(parents=True, exist_ok=True)
        # Staged inside the root so the final rename stays on one filesystem (atomic).
        tmp_dir = Path(tempfile.mkdtemp(dir=self.root, prefix=f".train-{artifact_id}-"))
        try:
            yield tmp_dir
            if target.exists():
                shutil.rmtree(target)
            os.rename(tmp_dir, target)
        except BaseException:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
