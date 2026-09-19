"""`stoptimes data arrivals` as a function: observations -> actual stop arrivals.

Thin orchestration over `stoptimes.core.arrivals.derive_arrivals` (the pure
map-matching function, re-exported unchanged below) that also handles the
I/O `stoptimes data arrivals` does: reading observations/static, writing the
result, and optionally registering it in the catalog.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from stoptimes.core.api._catalog import load_observations
from stoptimes.core.api._sources import resolve_source
from stoptimes.core.arrivals import ArrivalParams, derive_arrivals
from stoptimes.core.catalog import Catalog, DatasetEntry, DatasetKind
from stoptimes.core.contracts.source import Source

__all__ = ["derive_arrivals", "ArrivalParams", "build_arrivals_dataset"]


def build_arrivals_dataset(
    *,
    observations: str,
    static: str,
    source: str,
    out: str | Path,
    sources_dir: str | Path | None = None,
    register: str | None = None,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Derive an Arrival table from observations and write it to `out`.

    Equivalent to `stoptimes data arrivals`: resolves `source`'s agency
    config, loads `observations` (a catalog dataset name or a path/URI) and
    `static` (a GTFS zip path or URL), derives arrivals via
    `stoptimes.core.arrivals.derive_arrivals`, writes a parquet file, and
    (if `register` is given) adds it to the local catalog.

    Args:
        observations: a `stoptimes.core.api.add_dataset` name, or a
            path/URI `stoptimes.core.readers.read_observations` understands.
        static: a GTFS static `.zip` path or URL.
        source: the agency id (`examples/agencies/<source>.yaml`).
        out: where to write the Arrival parquet.
        sources_dir: agency config directory; defaults to
            `$STOPTIMES_SOURCES` / `./examples/agencies`.
        register: also register the output under this catalog name.
        overwrite: replace an existing `out` file / catalog entry.

    Returns:
        `(arrivals_frame, summary)` where `summary` matches `stoptimes data
        arrivals --json`'s ``{"source", "observations", "arrivals",
        "by_method", "static_tables", "out", "registered"}``.

    Example:
        >>> from stoptimes.core.api import build_arrivals_dataset  # doctest: +SKIP
        >>> arrivals, summary = build_arrivals_dataset(
        ...     observations="positions", static="gtfs.zip", source="demo",
        ...     out="arrivals.parquet",
        ... )  # doctest: +SKIP
        >>> summary["arrivals"] > 0  # doctest: +SKIP
        True

    Raises:
        FileExistsError: `out` already exists and `overwrite` is False.
    """
    from stoptimes.core import readers as readers_module

    src: Source = resolve_source(source, sources_dir=sources_dir)
    frame = load_observations(observations, source_id=source)
    loader, static_feed = readers_module.read_static(static, source)
    if loader is None:
        raise ValueError(f"{static!r} is missing required static table(s): {static_feed.missing_required()}")

    out_path = Path(out)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out} already exists (pass overwrite=True to replace it)")

    arrivals = derive_arrivals(frame, loader, source=src)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrivals.to_parquet(out_path, index=False)

    date_start = date_end = None
    if not arrivals.empty:
        date_start = str(arrivals["arrival_time"].min())
        date_end = str(arrivals["arrival_time"].max())

    if register:
        entry = DatasetEntry(
            name=register,
            source_id=source,
            kind=DatasetKind.ARRIVALS,
            location=str(out_path),
            date_start=date_start,
            date_end=date_end,
            row_count=int(len(arrivals)),
            added_at=pd.Timestamp.now(tz="UTC").to_pydatetime(),
            extra={"derived_from": observations, "static": static},
        )
        Catalog().add(entry, overwrite=overwrite)

    by_method = (
        {str(k): int(v) for k, v in arrivals["method"].value_counts().items()} if not arrivals.empty else {}
    )
    summary = {
        "source": source,
        "observations": int(len(frame)),
        "arrivals": int(len(arrivals)),
        "by_method": by_method,
        "static_tables": static_feed.tables,
        "out": str(out_path),
        "registered": register,
    }
    return arrivals, summary
