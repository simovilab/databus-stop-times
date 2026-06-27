"""Shared live-run dependency assembly (Phase A).

Centralises all live-run wiring so both the CLI and the future FastAPI layer
can share it without code duplication.

Public API
----------
parse_duration(value)              -> float
PREDICTOR_REGISTRY                  dict[str, type]
DEFAULT_PREDICTOR_IDS               list[str]
download_gtfs(cfg)                 -> Path   (sync; caller must os.unlink)
build_predictors(predictor_ids)    -> list
build_detectors(cfg)               -> list
build_live_run(cfg, ...)           -> LiveRunDeps
LiveRunDeps                         dataclass
_RouteFilterLiveSource              class (also re-exported for CLI use)
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from etaval.domain.models import SourceConfig
from etaval.engine.runtime import FeedSnapshot, LiveFeedSource
from etaval.feeds.gtfsrt_json import (
    TripUpdatesPoller,
    VehiclePositionRecord,
    VehiclePositionsPoller,
)
from etaval.feeds.static import GtfsStaticLoader
from etaval.groundtruth.position_radius import PositionRadiusDetector
from etaval.groundtruth.shape_distance import ShapeDistanceDetector
from etaval.groundtruth.stop_status import StopStatusDetector
from etaval.predictors.constant_speed import ConstantSpeedPredictor
from etaval.predictors.gtfsrt_tripupdates import GtfsRtTripUpdatesPredictor

# ── Predictor registry ────────────────────────────────────────────────────────

PREDICTOR_REGISTRY: dict[str, type] = {
    "gtfsrt": GtfsRtTripUpdatesPredictor,
    "baseline:constant_speed": ConstantSpeedPredictor,
}
DEFAULT_PREDICTOR_IDS: list[str] = ["gtfsrt", "baseline:constant_speed"]


# ── Duration parser ───────────────────────────────────────────────────────────


def parse_duration(value: str) -> float:
    """Parse a human-friendly duration string into seconds.

    Supported formats:

    - ``<N>h``  — hours (e.g. ``"1h"`` → 3600.0)
    - ``<N>m``  — minutes (e.g. ``"30m"`` → 1800.0)
    - ``<N>s``  — seconds (e.g. ``"90s"`` → 90.0)
    - ``<N>``   — bare number treated as seconds (e.g. ``"45"`` → 45.0)

    Raises:
        ValueError: If *value* is empty, has an unrecognised suffix, or its
            numeric part cannot be converted to float.

    Examples::

        >>> parse_duration("30m")
        1800.0
        >>> parse_duration("1h")
        3600.0
        >>> parse_duration("90s")
        90.0
        >>> parse_duration("45")
        45.0
    """
    s = value.strip()
    if not s:
        raise ValueError(f"Empty duration string: {value!r}")
    try:
        if s.endswith("h"):
            return float(s[:-1]) * 3600.0
        if s.endswith("m"):
            return float(s[:-1]) * 60.0
        if s.endswith("s"):
            return float(s[:-1])
        return float(s)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid duration {value!r}: {exc}") from exc


# ── Route filter (preserves LiveFeedSource semantics) ─────────────────────────


class _RouteFilterLiveSource(LiveFeedSource):
    """LiveFeedSource wrapper that drops VP records for non-matching route IDs.

    Inherits ``LiveFeedSource`` so ``isinstance(feed_source, LiveFeedSource)``
    remains ``True`` inside ``run_engine`` — this preserves the live-mode
    sleep (``asyncio.sleep(poll_interval_s)``) and the wall-clock duration check.

    ``LiveFeedSource.__init__`` is intentionally bypassed via
    ``object.__init__`` to avoid constructing duplicate poller references.
    This means ``self._vp`` and ``self._tu`` are never set on this subclass;
    they are never accessed because ``__anext__`` is fully overridden.
    """

    def __init__(self, inner: LiveFeedSource, route_ids: set[str]) -> None:
        object.__init__(self)  # bypass LiveFeedSource.__init__
        self._inner = inner
        self._route_ids = route_ids

    def __aiter__(self) -> _RouteFilterLiveSource:
        return self

    async def __anext__(self) -> FeedSnapshot:
        snap: FeedSnapshot = await self._inner.__anext__()
        filtered: list[VehiclePositionRecord] = [
            r for r in snap.vp_records if r.route_id in self._route_ids
        ]
        return FeedSnapshot(
            header_timestamp=snap.header_timestamp,
            vp_records=filtered,
            tu_header_timestamp=snap.tu_header_timestamp,
            tu_records=snap.tu_records,
        )


# ── GTFS download ─────────────────────────────────────────────────────────────


def download_gtfs(cfg: SourceConfig) -> Path:
    """Download the GTFS static zip synchronously to a NamedTemporaryFile.

    The caller is responsible for deleting the returned path after use::

        path = download_gtfs(cfg)
        try:
            loader = GtfsStaticLoader(path)
        finally:
            os.unlink(path)

    Kept synchronous so the future API can call it via ``asyncio.to_thread``.

    Args:
        cfg: SourceConfig containing ``gtfs_static_url`` and ``headers``.

    Returns:
        Path to the temporary .zip file.

    Raises:
        httpx.HTTPError: on network or HTTP-level failures.
    """
    with httpx.Client(timeout=120.0) as client:
        resp = client.get(cfg.gtfs_static_url, headers=dict(cfg.headers))
        resp.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp.write(resp.content)
        return Path(tmp.name)


# ── Predictor / detector builders ─────────────────────────────────────────────


def build_predictors(predictor_ids: list[str] | None = None) -> list[Any]:
    """Instantiate predictor objects from IDs.

    Args:
        predictor_ids: IDs to instantiate.  ``None`` uses DEFAULT_PREDICTOR_IDS.

    Returns:
        List of predictor instances.

    Raises:
        KeyError: if any ID is not in PREDICTOR_REGISTRY.
    """
    ids = predictor_ids if predictor_ids is not None else DEFAULT_PREDICTOR_IDS
    return [PREDICTOR_REGISTRY[pid]() for pid in ids]


def build_detectors(cfg: SourceConfig) -> list[Any]:
    """Instantiate the standard detector suite from a SourceConfig.

    Returns:
        ``[ShapeDistanceDetector, PositionRadiusDetector, StopStatusDetector]``
    """
    return [
        ShapeDistanceDetector(shape_threshold_m=cfg.shape_threshold_m),
        PositionRadiusDetector(arrival_radius_m=cfg.arrival_radius_m, dwell_s=cfg.dwell_s),
        StopStatusDetector(),
    ]


# ── LiveRunDeps ───────────────────────────────────────────────────────────────


@dataclass
class LiveRunDeps:
    """All dependencies needed to call run_engine in live mode.

    The caller MUST delete ``gtfs_path`` after the run::

        deps = build_live_run(cfg)
        try:
            ...run_engine(deps.feed_source, deps.static_loader, ...)...
        finally:
            try:
                os.unlink(deps.gtfs_path)
            except OSError:
                pass
    """

    feed_source: LiveFeedSource
    static_loader: GtfsStaticLoader
    predictors: list[Any]
    detectors: list[Any]
    gtfs_path: Path


def build_live_run(
    cfg: SourceConfig,
    predictor_ids: list[str] | None = None,
    routes: list[str] | None = None,
) -> LiveRunDeps:
    """Download GTFS and assemble all live-run dependencies.

    Downloads the GTFS static zip, loads it into a GtfsStaticLoader, builds
    VehiclePositionsPoller + TripUpdatesPoller + LiveFeedSource (optionally
    wrapped with a route filter), instantiates predictors and detectors.

    The caller MUST delete ``deps.gtfs_path`` after the run (see LiveRunDeps).

    Args:
        cfg:           SourceConfig for the target agency.
        predictor_ids: Predictor IDs to instantiate; ``None`` uses
                       DEFAULT_PREDICTOR_IDS.
        routes:        Optional list of route IDs to filter to.  ``None`` or
                       empty means all routes.

    Returns:
        LiveRunDeps instance.  The temporary GTFS zip at ``gtfs_path`` is
        still present — the caller must clean it up.

    Raises:
        httpx.HTTPError: if GTFS download fails.
        Exception:       if GtfsStaticLoader cannot parse the zip.
    """
    gtfs_path = download_gtfs(cfg)
    static_loader = GtfsStaticLoader(gtfs_path)

    headers: dict[str, str] = dict(cfg.headers)
    vp_poller = VehiclePositionsPoller(cfg.vehicle_positions_url, headers=headers)
    tu_poller = TripUpdatesPoller(cfg.trip_updates_url, headers=headers)
    live_source = LiveFeedSource(vp_poller, tu_poller)

    feed_source: LiveFeedSource
    if routes:
        feed_source = _RouteFilterLiveSource(live_source, set(routes))
    else:
        feed_source = live_source

    return LiveRunDeps(
        feed_source=feed_source,
        static_loader=static_loader,
        predictors=build_predictors(predictor_ids),
        detectors=build_detectors(cfg),
        gtfs_path=gtfs_path,
    )
