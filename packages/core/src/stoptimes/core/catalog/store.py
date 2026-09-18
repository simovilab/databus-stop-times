"""The local data catalog: a JSON file, no database.

Location resolution mirrors `core.config.ProjectConfig.model_registry_dir`
(the same fallback shape, kept local here rather than added to
`core/config.py`, which T1 doesn't own): `$STOPTIMES_HOME/catalog.json` if
`STOPTIMES_HOME` is set, else `./artifacts/catalog.json`.

Concurrency: "safe enough for a CLI", matching
`contracts.artifact.write_metadata` — every *write* is atomic (temp file +
`os.replace`), so a reader never observes a half-written file. The
read-modify-write sequence in `add`/`remove` is not itself locked (two
concurrent `data add` calls can race and one's change can be lost) — the
same tradeoff the artifact registry makes, acceptable for a single operator's
CLI, not for concurrent writers.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from stoptimes.core.config import get_config

from .models import DatasetEntry

CATALOG_FILENAME = "catalog.json"


class DatasetNotFoundError(KeyError):
    """`resolve`/`get` (when raising) found no dataset with that name."""


class DatasetExistsError(ValueError):
    """`add` without `overwrite=True` collided with an existing name."""


def catalog_path(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the catalog JSON file's path.

    1. `explicit`, if given.
    2. `$STOPTIMES_HOME/catalog.json`.
    3. `./artifacts/catalog.json`, relative to the current working directory.
    """
    if explicit is not None:
        return Path(explicit)
    config = get_config()
    if config.stoptimes_home is not None:
        return config.stoptimes_home / CATALOG_FILENAME
    return Path.cwd() / "artifacts" / CATALOG_FILENAME


class Catalog:
    """CRUD over the local dataset registry: `add`, `get`, `list`, `remove`, `resolve`."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self._path = catalog_path(path)

    @property
    def path(self) -> Path:
        return self._path

    def _read_all(self) -> dict[str, DatasetEntry]:
        if not self._path.exists():
            return {}
        text = self._path.read_text(encoding="utf-8").strip()
        if not text:
            return {}
        raw = json.loads(text)
        return {name: DatasetEntry.model_validate(data) for name, data in raw.items()}

    def _write_all(self, entries: dict[str, DatasetEntry]) -> None:
        directory = self._path.parent
        directory.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {name: entry.model_dump(mode="json") for name, entry in entries.items()},
            indent=2,
            sort_keys=True,
        )
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".catalog.", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload + "\n")
            os.replace(tmp, self._path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def add(self, entry: DatasetEntry, *, overwrite: bool = False) -> DatasetEntry:
        """Register `entry`. Raises `DatasetExistsError` on a name collision
        unless `overwrite=True`."""
        entries = self._read_all()
        if entry.name in entries and not overwrite:
            raise DatasetExistsError(f"dataset {entry.name!r} already exists in the catalog")
        entries[entry.name] = entry
        self._write_all(entries)
        return entry

    def get(self, name: str) -> DatasetEntry | None:
        return self._read_all().get(name)

    def list(self) -> list[DatasetEntry]:
        return sorted(self._read_all().values(), key=lambda e: e.name)

    def remove(self, name: str) -> bool:
        """Returns whether `name` was present (and removed)."""
        entries = self._read_all()
        if name not in entries:
            return False
        del entries[name]
        self._write_all(entries)
        return True

    def resolve(self, name: str) -> str:
        """The registered `location` for `name`. Raises `DatasetNotFoundError`."""
        entry = self.get(name)
        if entry is None:
            raise DatasetNotFoundError(name)
        return entry.location
