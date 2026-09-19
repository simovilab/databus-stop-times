"""Load an agency's `Source` config the same way every CLI command does."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from stoptimes.core.config import sources_search_path
from stoptimes.core.contracts.adapters import source_from_val_yaml
from stoptimes.core.contracts.source import Source

__all__ = ["resolve_source"]


def resolve_source(source_id: str, *, sources_dir: str | Path | None = None) -> Source:
    """Load the `Source` config for `source_id`.

    Searches `sources_dir` if given, else `$STOPTIMES_SOURCES`, else
    `./examples/agencies` (`stoptimes.core.config.sources_search_path`) for a
    `<source_id>.yaml` file, exactly like `stoptimes data arrivals --source`
    and `stoptimes serve --source` do.

    Example:
        >>> from stoptimes.core.api import resolve_source
        >>> src = resolve_source("demo", sources_dir="examples/agencies")  # doctest: +SKIP
        >>> src.id  # doctest: +SKIP
        'demo'

    Raises:
        FileNotFoundError: no `<source_id>.yaml` was found in any candidate
            directory.
    """
    candidates = (
        [Path(sources_dir)] if sources_dir else sources_search_path(os.environ.get("STOPTIMES_SOURCES"))
    )
    for directory in candidates:
        path = Path(directory) / f"{source_id}.yaml"
        if path.is_file():
            return source_from_val_yaml(yaml.safe_load(path.read_text(encoding="utf-8")))
    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"no agency file for source {source_id!r} (searched: {searched})")
