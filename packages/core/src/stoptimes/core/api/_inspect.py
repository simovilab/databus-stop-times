"""`stoptimes data inspect` as a function: coverage of a registered dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stoptimes.core.api._catalog import guess_format
from stoptimes.core.catalog import Catalog
from stoptimes.core.catalog.inspect import build_coverage_report
from stoptimes.core.contracts.data import ObservationKind

__all__ = ["inspect_dataset"]


def inspect_dataset(
    name: str,
    *,
    static: str | Path | None = None,
    gap_threshold_s: float = 300.0,
) -> dict[str, Any]:
    """Summarize coverage of a catalog dataset (rows, date range, gaps, and
    trip/route coverage against a static feed if given).

    Equivalent to `stoptimes data inspect`.

    Args:
        name: a dataset name registered with `stoptimes.core.api.add_dataset`.
        static: an optional GTFS static `.zip` to check trip coverage against.
        gap_threshold_s: a gap this wide (seconds) between consecutive
            timestamps counts toward `gap_count`.

    Returns:
        A flat `{"kind": str, ...}` dict; the remaining keys depend on
        `kind` (`static`/`arrivals`/an observation kind) -- see
        `stoptimes.core.catalog.inspect.build_coverage_report`.

    Example:
        >>> from stoptimes.core.api import inspect_dataset  # doctest: +SKIP
        >>> report = inspect_dataset("positions")  # doctest: +SKIP
        >>> report["rows"] >= 0  # doctest: +SKIP
        True

    Raises:
        KeyError: `name` is not a registered dataset.
    """
    from stoptimes.core import readers as readers_module

    entry = Catalog().get(name)
    if entry is None:
        raise KeyError(f"no such dataset: {name!r}")

    if entry.kind.value == "static":
        _, feed = readers_module.read_static(entry.location, entry.source_id or "unknown")
        return {"kind": "static", "tables": feed.tables, "missing_required": feed.missing_required()}

    if entry.kind.value == "arrivals":
        return {"kind": "arrivals", "note": "coverage for arrivals is build_arrivals_dataset's summary."}

    obs_kind = ObservationKind(entry.extra.get("observation_kind", "vehicle_position"))
    reader_format = entry.extra.get("format", "archive_parquet")
    mapping_obj = None
    if entry.extra.get("mapping"):
        mapping_obj = readers_module.ColumnMapping.model_validate(entry.extra["mapping"])
    df = readers_module.read_observations(
        entry.location,
        format=reader_format,
        kind=obs_kind,
        source_id=entry.source_id or "unknown",
        mapping=mapping_obj,
    )

    static_trip_ids = None
    if static is not None:
        loader, static_feed = readers_module.read_static(static, entry.source_id or "unknown")
        if loader is not None:
            static_trip_ids = set(loader.trips["trip_id"].astype(str))

    report = build_coverage_report(
        df, obs_kind.value, static_trip_ids=static_trip_ids, gap_threshold_s=gap_threshold_s
    )
    return {"kind": obs_kind.value, **report}
