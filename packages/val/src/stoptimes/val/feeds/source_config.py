"""YAML -> SourceConfig loader + source discovery. Phase P0.

Source configs live alongside agency profiles in ``examples/agencies/`` (one
YAML per agency, holding both val's feed/detector fields and the optional
`stoptimes.core.config` profile keys -- see `examples/agencies/README.md`).
Discovery uses the shared `stoptimes.core.config.sources_search_path`
resolver so `stoptimes val` and `stoptimes serve` look in the same places:

1. An explicit directory (e.g. the CLI's ``--sources-dir``).
2. `STOPTIMES_SOURCES`, `os.pathsep`-separated.
3. `./examples/agencies`, relative to the current working directory.

`SourceConfig` (a pydantic model) ignores unknown top-level keys by default,
so agency-profile-only fields (`timezone`, `holiday_region`, `serving`) in a
shared file don't need special handling here.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from stoptimes.core.config import sources_search_path
from stoptimes.core.domain.models import SourceConfig


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
    """Return the stems of all *.yaml source configs found on the search path.

    *config_dir*, if given, is searched exclusively (e.g. ``--sources-dir``).
    Otherwise every directory from `sources_search_path()` is searched and
    the results merged. Results are sorted alphabetically.
    """
    names: set[str] = set()
    for directory in sources_search_path(explicit=config_dir):
        directory = Path(directory)
        if directory.is_dir():
            names.update(p.stem for p in directory.glob("*.yaml"))
    return sorted(names)


def load_named_source(name: str, config_dir: Path | None = None) -> SourceConfig:
    """Load a SourceConfig by *name*, searching `sources_search_path()`.

    *config_dir*, if given, is searched exclusively (e.g. ``--sources-dir``).
    Raises FileNotFoundError, listing every directory searched, if
    ``{name}.yaml`` isn't found in any of them.
    """
    dirs = sources_search_path(explicit=config_dir)
    for directory in dirs:
        candidate = Path(directory) / f"{name}.yaml"
        if candidate.is_file():
            return load_source_config(candidate)
    searched = ", ".join(str(d) for d in dirs)
    raise FileNotFoundError(
        f"source {name!r} not found. Searched: {searched}. "
        "Set STOPTIMES_SOURCES (or pass --sources-dir) to point at your "
        "agency profile directory -- see examples/agencies/README.md."
    )
