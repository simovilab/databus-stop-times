"""YAML -> SourceConfig loader + source discovery. Phase P0."""

from __future__ import annotations

from pathlib import Path

import yaml

from etaval.domain.models import SourceConfig

# Default shipped sources directory — resolved relative to this package file,
# not relative to cwd, so the CLI works from any working directory.
_SOURCES_DIR = Path(__file__).parent.parent / "config" / "sources"


def load_source_config(path: str | Path) -> SourceConfig:
    """Parse a YAML file at *path* and return a SourceConfig.

    Raises FileNotFoundError if the file does not exist.
    Raises pydantic.ValidationError if required fields are missing or wrong type.
    """
    path = Path(path)
    with path.open() as fh:
        data = yaml.safe_load(fh)
    return SourceConfig(**data)


def list_sources(config_dir: Path | None = None) -> list[str]:
    """Return the stems of all *.yaml source configs in *config_dir*.

    Defaults to the shipped ``etaval/config/sources/`` directory.
    Results are sorted alphabetically.
    """
    directory = Path(config_dir) if config_dir is not None else _SOURCES_DIR
    return sorted(p.stem for p in directory.glob("*.yaml"))


def load_named_source(name: str, config_dir: Path | None = None) -> SourceConfig:
    """Load a SourceConfig by *name* from *config_dir*.

    Defaults to the shipped ``etaval/config/sources/`` directory.
    Raises FileNotFoundError if ``{name}.yaml`` does not exist in the directory.
    """
    directory = Path(config_dir) if config_dir is not None else _SOURCES_DIR
    return load_source_config(directory / f"{name}.yaml")
