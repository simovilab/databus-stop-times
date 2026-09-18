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

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from stoptimes.core.config import sources_search_path
from stoptimes.core.domain.models import SourceConfig

FeedFormat = Literal["json", "pb", "auto"]


@dataclass(frozen=True)
class FeedFormats:
    """Per-feed wire format, read from the same source YAML as ``SourceConfig``.

    ``SourceConfig`` (pydantic) ignores unknown top-level keys, so an agency
    YAML can carry ``vehicle_positions_format`` / ``trip_updates_format``
    (``"json"`` | ``"pb"`` | ``"auto"``, default ``"auto"``) without a
    ``core`` contract change -- ``load_feed_formats`` reads the same file a
    second time for just these two keys. ``"auto"`` sniffs the format per
    response from Content-Type / body (see ``stoptimes.val.feeds.pb.sniff_format``),
    which is also what an agency YAML with neither key set gets by default --
    so existing JSON-only configs behave exactly as before.
    """

    vehicle_positions: FeedFormat = "auto"
    trip_updates: FeedFormat = "auto"


def load_source_config(path: str | Path) -> SourceConfig:
    """Parse a YAML file at *path* and return a SourceConfig.

    Raises FileNotFoundError if the file does not exist.
    Raises pydantic.ValidationError if required fields are missing or wrong type.
    """
    path = Path(path)
    with path.open() as fh:
        data = yaml.safe_load(fh)
    return SourceConfig(**data)


def load_feed_formats(path: str | Path) -> FeedFormats:
    """Read the optional ``{vehicle_positions,trip_updates}_format`` keys from *path*.

    Raises FileNotFoundError if the file does not exist.
    """
    path = Path(path)
    with path.open() as fh:
        data = yaml.safe_load(fh) or {}
    return FeedFormats(
        vehicle_positions=data.get("vehicle_positions_format", "auto"),
        trip_updates=data.get("trip_updates_format", "auto"),
    )


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


def load_named_feed_formats(name: str, config_dir: Path | None = None) -> FeedFormats:
    """Load ``FeedFormats`` for a named source, searching the same path as ``load_named_source``.

    Returns the all-``"auto"`` default (never raises) if the named source
    isn't found -- callers that need the missing-source error should call
    ``load_named_source``/``list_sources`` themselves; this helper is meant
    to be layered on top of an already-resolved source.
    """
    dirs = sources_search_path(explicit=config_dir)
    for directory in dirs:
        candidate = Path(directory) / f"{name}.yaml"
        if candidate.is_file():
            return load_feed_formats(candidate)
    return FeedFormats()
