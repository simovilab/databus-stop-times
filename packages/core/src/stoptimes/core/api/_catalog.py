"""`stoptimes data add/list/remove` as functions: the local dataset catalog.

Mirrors `stoptimes.cli.commands.data` exactly (same auto-format-guessing,
same row-count/date-range bookkeeping) so the CLI can be rewritten as a
thin wrapper over these without behavior changes. See docs/notes/p3-u4.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from stoptimes.core import readers as readers_module
from stoptimes.core.catalog import Catalog, DatasetEntry, DatasetExistsError, DatasetKind
from stoptimes.core.contracts.data import ObservationKind

__all__ = [
    "add_dataset",
    "list_datasets",
    "get_dataset",
    "remove_dataset",
    "load_observations",
    "load_arrivals_frame",
    "guess_format",
    "DatasetEntry",
    "DatasetExistsError",
    "DatasetKind",
]

_FORMAT_CHOICES = ("gtfsrt_json", "gtfsrt_pb", "archive_parquet", "avl_csv")


def guess_format(location: str, *, mapping: str | Path | None = None) -> str:
    """Guess a reader format from a location's extension.

    Example:
        >>> from stoptimes.core.api import guess_format
        >>> guess_format("archive/2026-09-17.parquet")
        'archive_parquet'
        >>> guess_format("avl.csv")
        'avl_csv'
    """
    if mapping is not None:
        return "avl_csv"
    lower = location.lower()
    if lower.endswith(".csv"):
        return "avl_csv"
    if lower.endswith(".pb") or lower.endswith(".bin"):
        return "gtfsrt_pb"
    if lower.endswith(".json"):
        return "gtfsrt_json"
    return "archive_parquet"


def add_dataset(
    name: str,
    location: str,
    *,
    source: str,
    kind: str = "observations",
    format: str | None = None,
    observation_kind: str = "vehicle_position",
    mapping: str | Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    overwrite: bool = False,
) -> DatasetEntry:
    """Register a GTFS static / realtime / AVL dataset in the local catalog.

    Reads just enough of `location` to record its shape (row count, date
    range for observations; table presence for a static feed) and writes a
    `DatasetEntry`. Equivalent to `stoptimes data add`.

    Example:
        >>> from stoptimes.core.api import add_dataset  # doctest: +SKIP
        >>> entry = add_dataset(
        ...     "positions", "archive/", source="demo", kind="observations",
        ... )  # doctest: +SKIP
        >>> entry.kind.value  # doctest: +SKIP
        'observations'

    Raises:
        DatasetExistsError: `name` is already registered and `overwrite` is False.
        ValueError: an unknown `kind`/`format`/`observation_kind`, or
            `format="avl_csv"` with no `mapping`.
    """
    dataset_kind = DatasetKind(kind)

    row_count: int | None = None
    date_start = start_date
    date_end = end_date
    extra: dict = {}

    if dataset_kind is DatasetKind.STATIC:
        _, static_feed = readers_module.read_static(location, source)
        extra = {"tables": static_feed.tables, "missing_required": static_feed.missing_required()}
    elif dataset_kind is DatasetKind.ARRIVALS:
        extra = {}
    else:
        reader_format = format or guess_format(location, mapping=mapping)
        if reader_format not in _FORMAT_CHOICES:
            raise ValueError(f"unknown format {reader_format!r}; expected one of {_FORMAT_CHOICES}")
        obs_kind = ObservationKind(observation_kind)

        mapping_obj = None
        mapping_dump = None
        if reader_format == "avl_csv":
            if mapping is None:
                raise ValueError("format='avl_csv' requires a mapping= file")
            mapping_obj = readers_module.ColumnMapping.from_file(mapping)
            mapping_dump = mapping_obj.model_dump()

        df = readers_module.read_observations(
            location,
            format=reader_format,
            kind=obs_kind,
            source_id=source,
            mapping=mapping_obj,
            start_date=start_date,
            end_date=end_date,
        )
        row_count = int(len(df))
        if row_count and "ts" in df.columns:
            usable = df["ts"].dropna()
            if not usable.empty:
                date_start = date_start or str(usable.min())
                date_end = date_end or str(usable.max())
        extra = {"format": reader_format, "observation_kind": obs_kind.value}
        if mapping_dump is not None:
            extra["mapping"] = mapping_dump

    entry = DatasetEntry(
        name=name,
        source_id=source,
        kind=dataset_kind,
        location=location,
        date_start=date_start,
        date_end=date_end,
        row_count=row_count,
        added_at=pd.Timestamp.now(tz="UTC").to_pydatetime(),
        extra=extra,
    )
    return Catalog().add(entry, overwrite=overwrite)


def list_datasets() -> list[DatasetEntry]:
    """List every dataset registered in the local catalog.

    Example:
        >>> from stoptimes.core.api import list_datasets
        >>> list_datasets()  # doctest: +SKIP
        [DatasetEntry(name='positions', ...)]
    """
    return Catalog().list()


def get_dataset(name: str) -> DatasetEntry | None:
    """The catalog entry named `name`, or `None` if it isn't registered."""
    return Catalog().get(name)


def remove_dataset(name: str) -> bool:
    """Remove `name` from the catalog (never deletes the underlying file).

    Returns whether an entry was actually present and removed.
    """
    return Catalog().remove(name)


def load_observations(reference: str, *, source_id: str) -> pd.DataFrame:
    """Load an Observation frame from a catalog dataset name, path, or URI.

    `reference` is looked up in the catalog first; if it isn't a registered
    name, it's treated as a path/URI directly, with the format guessed from
    its extension (same fallback `stoptimes data frame`/`data arrivals` use).

    Example:
        >>> from stoptimes.core.api import load_observations  # doctest: +SKIP
        >>> df = load_observations("positions", source_id="demo")  # doctest: +SKIP
    """
    entry = Catalog().get(reference)
    location = entry.location if entry is not None else reference
    extra = entry.extra if entry is not None else {}
    reader_format = extra.get("format") or guess_format(location)
    kind = ObservationKind(extra.get("observation_kind", ObservationKind.VEHICLE_POSITION.value))
    mapping = None
    if reader_format == "avl_csv" and extra.get("mapping"):
        mapping = readers_module.ColumnMapping(**extra["mapping"])

    return readers_module.read_observations(
        location, format=reader_format, kind=kind, source_id=source_id, mapping=mapping
    )


def load_arrivals_frame(reference: str) -> pd.DataFrame:
    """Load an Arrival table from a catalog dataset name or a parquet path/URI.

    Example:
        >>> from stoptimes.core.api import load_arrivals_frame  # doctest: +SKIP
        >>> arrivals = load_arrivals_frame("arrivals")  # doctest: +SKIP
    """
    entry = Catalog().get(reference)
    location = entry.location if entry is not None else reference
    return pd.read_parquet(location)
