"""`stoptimes diagnose` as a function: feed quality report before training."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from stoptimes.core.api._catalog import guess_format
from stoptimes.core.api._sources import resolve_source
from stoptimes.core.catalog import Catalog
from stoptimes.core.contracts.data import ObservationKind
from stoptimes.core.contracts.source import FeedKind, Source
from stoptimes.core.diagnose import DiagnoseParams, DiagnosticReport, diagnose_feed

__all__ = ["DiagnoseParams", "DiagnosticReport", "diagnose"]


def _load_observation_frame(
    *,
    location: str | None,
    format_override: str | None,
    kind: ObservationKind,
    feed_kind: FeedKind,
    src: Source | None,
    source_id: str,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame | None:
    from stoptimes.core import readers as readers_module

    if location is None and src is None:
        return None
    if location is not None:
        entry = Catalog().get(location)
        loc = entry.location if entry is not None else location
        extra = entry.extra if entry is not None else {}
        reader_format = format_override or extra.get("format") or guess_format(loc)
        mapping_obj = None
        if reader_format == "avl_csv" and extra.get("mapping"):
            mapping_obj = readers_module.ColumnMapping(**extra["mapping"])
        return readers_module.read_observations(
            loc,
            format=reader_format,
            kind=kind,
            source_id=source_id,
            mapping=mapping_obj,
            start_date=start_date,
            end_date=end_date,
        )
    return readers_module.read_observations_for_source(
        src, feed_kind, start_date=start_date, end_date=end_date
    )


def diagnose(
    *,
    static: str | None = None,
    vehicle_positions: str | None = None,
    vehicle_positions_format: str | None = None,
    trip_updates: str | None = None,
    trip_updates_format: str | None = None,
    from_: str | None = None,
    to: str | None = None,
    source: str | None = None,
    sources_dir: str | Path | None = None,
    gap_threshold_s: float = 300.0,
    stale_threshold_s: float = 120.0,
    majority_threshold: float = 0.5,
    max_samples: int = 10,
) -> DiagnosticReport:
    """Report GTFS / GTFS-RT feed quality problems before training on them.

    Equivalent to `stoptimes diagnose`. `vehicle_positions`/`trip_updates`
    each accept a catalog dataset name, a path/URI, or a live feed URL; if
    omitted and `source` resolves to a configured agency, that agency's
    matching feed is used instead. A feed with quality problems is a normal,
    successful return -- see `report.findings` -- not an exception.

    Example:
        >>> from stoptimes.core.api import diagnose
        >>> report = diagnose()  # no static, no realtime data at all
        >>> report.static_present
        False
        >>> any(f.id == "static_missing" for f in report.findings)
        True

    Raises:
        ValueError / OSError: a given `static`/`vehicle_positions`/
            `trip_updates` location could not be read at all.
    """
    from stoptimes.core import readers as readers_module

    source_id = source or "diagnose"
    src = resolve_source(source, sources_dir=sources_dir) if source else None

    static_location = static or (src.gtfs_static_url if src is not None else None)
    static_loader = None
    static_feed = None
    if static_location is not None:
        static_loader, static_feed = readers_module.read_static(static_location, source_id)

    vp_df = _load_observation_frame(
        location=vehicle_positions,
        format_override=vehicle_positions_format,
        kind=ObservationKind.VEHICLE_POSITION,
        feed_kind=FeedKind.VEHICLE_POSITIONS,
        src=src,
        source_id=source_id,
        start_date=from_,
        end_date=to,
    )
    tu_df = _load_observation_frame(
        location=trip_updates,
        format_override=trip_updates_format,
        kind=ObservationKind.TRIP_UPDATE,
        feed_kind=FeedKind.TRIP_UPDATES,
        src=src,
        source_id=source_id,
        start_date=from_,
        end_date=to,
    )

    params = DiagnoseParams(
        gap_threshold_s=gap_threshold_s,
        stale_threshold_s=stale_threshold_s,
        majority_threshold=majority_threshold,
        max_samples=max_samples,
    )
    return diagnose_feed(
        vp_df=vp_df, tu_df=tu_df, static_loader=static_loader, static_feed=static_feed, params=params
    )
