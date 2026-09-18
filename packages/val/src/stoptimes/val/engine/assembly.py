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
build_replay_run(cfg, ...)         -> ReplayRunDeps
LiveRunDeps                         dataclass
ReplayRunDeps                       dataclass
_RouteFilterLiveSource              class (also re-exported for CLI use)
"""

from __future__ import annotations

import datetime as dt
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from stoptimes.core.domain.models import SourceConfig
from stoptimes.val.engine.runtime import FeedSnapshot, LiveFeedSource
from stoptimes.core.realtime.gtfsrt_json import VehiclePositionRecord
from stoptimes.val.feeds.pb import AutoTripUpdatesPoller, AutoVehiclePositionsPoller
from stoptimes.val.feeds.source_config import FeedFormats
from stoptimes.core.gtfs.static import GtfsStaticLoader
from stoptimes.val.groundtruth.position_radius import PositionRadiusDetector
from stoptimes.val.groundtruth.shape_distance import ShapeDistanceDetector
from stoptimes.val.groundtruth.stop_status import StopStatusDetector
from stoptimes.val.predictors.artifact import StoptimesArtifactPredictor
from stoptimes.val.predictors.constant_speed import ConstantSpeedPredictor
from stoptimes.val.predictors.gtfs_eta_models import GtfsEtaModelPredictor
from stoptimes.val.predictors.gtfsrt_tripupdates import GtfsRtTripUpdatesPredictor
from stoptimes.val.predictors.pipeline_adapter import PipelineTripUpdatesPredictor
from stoptimes.val.predictors.schedule_baseline import ScheduleBaselinePredictor

# ── Predictor registry ────────────────────────────────────────────────────────
#
# Values are zero-arg factories rather than bare types so entries can close
# over which trained model_type (or file/artifact ref) to request.
#
# "stoptimes:{historical_mean,polyreg_distance,polyreg_time,xgboost}" are the
# canonical names for the four trained-model families gtfs-eta shipped
# (`GtfsEtaModelPredictor` unchanged underneath -- only the advertised
# predictor_id differs, via `_stoptimes_alias`). "gtfs_eta:*" is kept
# working as a *deprecated* alias of the matching "stoptimes:*" entry (same
# predictor, same predictor_id as before -- existing run data and scripts
# that filter/group by `gtfs_eta:xgboost` etc. keep working), emitting a
# `DeprecationWarning` on construction. "gtfs_eta:ewma" was retired upstream
# and was never re-added here.
#
# Two more predictor families resolve *dynamically* rather than through this
# fixed dict -- see `resolve_predictor_factory` below:
#   stoptimes:<artifact_id-or-scope>   -> StoptimesArtifactPredictor
#   pipeline:<file.py>                 -> PipelineTripUpdatesPredictor


def _stoptimes_alias(model_type: str) -> GtfsEtaModelPredictor:
    """A GtfsEtaModelPredictor advertised under its new canonical `stoptimes:*` id."""
    predictor = GtfsEtaModelPredictor(model_type)
    predictor.predictor_id = f"stoptimes:{model_type}"
    return predictor


def _gtfs_eta_deprecated_alias(model_type: str) -> GtfsEtaModelPredictor:
    """A GtfsEtaModelPredictor built via its deprecated `gtfs_eta:*` id.

    `predictor_id` is left as `f"gtfs_eta:{model_type}"` (unchanged) so runs
    started this way stay grouped/filterable exactly as before; only a
    DeprecationWarning is new.
    """
    warnings.warn(
        f"Predictor id 'gtfs_eta:{model_type}' is deprecated; use "
        f"'stoptimes:{model_type}' instead. 'gtfs_eta:*' is kept only as an "
        "alias for backward compatibility with existing run data/scripts.",
        DeprecationWarning,
        stacklevel=2,
    )
    return GtfsEtaModelPredictor(model_type)


_TRAINED_MODEL_TYPES: tuple[str, ...] = (
    "historical_mean",
    "polyreg_distance",
    "polyreg_time",
    "xgboost",
)

PREDICTOR_REGISTRY: dict[str, Any] = {
    "gtfsrt": GtfsRtTripUpdatesPredictor,
    "baseline:constant_speed": ConstantSpeedPredictor,
    "baseline:schedule": ScheduleBaselinePredictor,
    **{f"stoptimes:{mt}": (lambda mt=mt: _stoptimes_alias(mt)) for mt in _TRAINED_MODEL_TYPES},
    **{f"gtfs_eta:{mt}": (lambda mt=mt: _gtfs_eta_deprecated_alias(mt)) for mt in _TRAINED_MODEL_TYPES},
}
DEFAULT_PREDICTOR_IDS: list[str] = ["gtfsrt", "baseline:constant_speed"]

# Prefixes resolved dynamically (arbitrary suffix), rather than as fixed
# PREDICTOR_REGISTRY entries -- see `resolve_predictor_factory`.
_DYNAMIC_PREDICTOR_PREFIXES: tuple[str, ...] = ("stoptimes:", "pipeline:")


def is_known_predictor_id(pid: str) -> bool:
    """True if *pid* resolves via `PREDICTOR_REGISTRY` or a dynamic prefix.

    Use this (not ``pid in PREDICTOR_REGISTRY``) to validate predictor IDs
    from a CLI flag or API request body, so ``stoptimes:<any-artifact-id>``
    and ``pipeline:<any-file.py>`` are accepted without needing to be
    enumerated in the fixed dict.
    """
    if pid in PREDICTOR_REGISTRY:
        return True
    return any(pid.startswith(prefix) and len(pid) > len(prefix) for prefix in _DYNAMIC_PREDICTOR_PREFIXES)


def resolve_predictor_factory(pid: str) -> Any:
    """Return a zero-arg factory for *pid* -- a fixed `PREDICTOR_REGISTRY`
    entry, or one built dynamically for a `stoptimes:`/`pipeline:` prefix
    whose suffix isn't one of the four known trained-model families.

    Raises:
        KeyError: *pid* doesn't match a fixed entry or a known dynamic prefix.
    """
    if pid in PREDICTOR_REGISTRY:
        return PREDICTOR_REGISTRY[pid]
    if pid.startswith("stoptimes:") and len(pid) > len("stoptimes:"):
        ref = pid[len("stoptimes:") :]
        return lambda: StoptimesArtifactPredictor(ref)
    if pid.startswith("pipeline:") and len(pid) > len("pipeline:"):
        path = pid[len("pipeline:") :]
        return lambda: PipelineTripUpdatesPredictor(path)
    raise KeyError(pid)


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
        KeyError: if any ID resolves through neither PREDICTOR_REGISTRY nor
            a dynamic `stoptimes:`/`pipeline:` prefix (see `resolve_predictor_factory`).
    """
    ids = predictor_ids if predictor_ids is not None else DEFAULT_PREDICTOR_IDS
    return [resolve_predictor_factory(pid)() for pid in ids]


def _attach_static_loader(predictors: list[Any], static_loader: GtfsStaticLoader) -> None:
    """Give every predictor that exposes ``attach_static_loader`` the run's loader.

    Duck-typed opt-in, mirroring the ``accepts_position_history`` pattern:
    any predictor instance with a callable ``attach_static_loader`` attribute
    is called with *static_loader*; predictors without the method are left
    untouched. Currently only :class:`~stoptimes.val.predictors.schedule_baseline.
    ScheduleBaselinePredictor` uses this hook.
    """
    for predictor in predictors:
        attach = getattr(predictor, "attach_static_loader", None)
        if callable(attach):
            attach(static_loader)


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
    formats: FeedFormats | None = None,
) -> LiveRunDeps:
    """Download GTFS and assemble all live-run dependencies.

    Downloads the GTFS static zip, loads it into a GtfsStaticLoader, builds
    the VP/TU pollers + LiveFeedSource (optionally wrapped with a route
    filter), instantiates predictors and detectors.

    The caller MUST delete ``deps.gtfs_path`` after the run (see LiveRunDeps).

    Args:
        cfg:           SourceConfig for the target agency.
        predictor_ids: Predictor IDs to instantiate; ``None`` uses
                       DEFAULT_PREDICTOR_IDS.
        routes:        Optional list of route IDs to filter to.  ``None`` or
                       empty means all routes.
        formats:       Per-feed wire format (``FeedFormats``, see
                       ``stoptimes.val.feeds.source_config``). ``None`` uses
                       ``FeedFormats()`` — both feeds default to ``"auto"``,
                       which sniffs JSON vs protobuf per response
                       (``stoptimes.val.feeds.pb.sniff_format``) and is a
                       no-op for existing JSON-only sources.

    Returns:
        LiveRunDeps instance.  The temporary GTFS zip at ``gtfs_path`` is
        still present — the caller must clean it up.

    Raises:
        httpx.HTTPError: if GTFS download fails.
        Exception:       if GtfsStaticLoader cannot parse the zip.
    """
    gtfs_path = download_gtfs(cfg)
    static_loader = GtfsStaticLoader(gtfs_path)

    formats = formats or FeedFormats()
    headers: dict[str, str] = dict(cfg.headers)
    vp_poller = AutoVehiclePositionsPoller(
        cfg.vehicle_positions_url, headers=headers, format=formats.vehicle_positions
    )
    tu_poller = AutoTripUpdatesPoller(
        cfg.trip_updates_url, headers=headers, format=formats.trip_updates
    )
    live_source = LiveFeedSource(vp_poller, tu_poller)

    feed_source: LiveFeedSource
    if routes:
        feed_source = _RouteFilterLiveSource(live_source, set(routes))
    else:
        feed_source = live_source

    predictors = build_predictors(predictor_ids)
    _attach_static_loader(predictors, static_loader)

    return LiveRunDeps(
        feed_source=feed_source,
        static_loader=static_loader,
        predictors=predictors,
        detectors=build_detectors(cfg),
        gtfs_path=gtfs_path,
    )


# ── ReplayRunDeps (A1: archived-Parquet replay) ────────────────────────────────


@dataclass
class ReplayRunDeps:
    """All dependencies needed to call run_engine in archived-replay mode.

    Unlike :class:`LiveRunDeps`, ``gtfs_path`` is only set (and only needs
    cleanup) when the static GTFS zip was fetched from S3 to a temp file; a
    local ``--replay-gtfs`` / ``--replay-gtfs-dir`` match is a path the caller
    does not own and must NOT delete.
    """

    feed_source: Any  # ArchiveReplayFeedSource (stoptimes.val.feeds.replay_archive)
    static_loader: GtfsStaticLoader
    predictors: list[Any]
    detectors: list[Any]
    gtfs_path: Path | None


def build_replay_run(
    cfg: SourceConfig,
    *,
    vp_path: str | Path,
    tu_path: str | Path | None = None,
    start: Any = None,
    end: Any = None,
    routes: list[str] | None = None,
    predictor_ids: list[str] | None = None,
    static_gtfs_zip: str | Path | None = None,
    static_gtfs_dir: str | Path | None = None,
) -> ReplayRunDeps:
    """Assemble all dependencies for an archived-Parquet replay run.

    Mirrors :func:`build_live_run`'s wiring of predictors/detectors, but
    sources snapshots from :class:`~stoptimes.val.feeds.replay_archive.ArchiveReplayFeedSource`
    instead of live HTTP polling, and resolves the static GTFS zip via
    :func:`~stoptimes.val.feeds.replay_archive.resolve_static_gtfs` (a local
    ``static_gtfs_zip`` / ``static_gtfs_dir`` snapshot matched to the first
    archived tick's date, rather than downloading ``cfg.gtfs_static_url``).

    Args:
        cfg:            SourceConfig; only its detector params are used here
                        (arrival_radius_m, dwell_s, shape_threshold_m) — the
                        live feed URLs are ignored in replay mode.
        vp_path:        Archived VP Parquet path (dir, file, or s3://).
        tu_path:        Archived TU Parquet path; ``None`` for VP-only replay.
        start, end:     Optional replay window bounds (epoch seconds or ISO
                        8601 strings).
        routes:         Optional route ID filter; ``None``/empty means all.
        predictor_ids:  Predictor IDs to instantiate; ``None`` uses
                        DEFAULT_PREDICTOR_IDS.
        static_gtfs_zip: Explicit static GTFS zip path — takes priority.
        static_gtfs_dir: Directory of dated ``YYYY-MM-DD.zip`` snapshots; the
                        newest one <= the first archived tick's date is used.

    Returns:
        ReplayRunDeps instance.

    Raises:
        ValueError: no VP rows matched the path/window/route filter, or
            neither static_gtfs_zip nor static_gtfs_dir was given.
        FileNotFoundError: the resolved static GTFS zip does not exist, or no
            dated snapshot <= the target date was found in static_gtfs_dir.
    """
    from stoptimes.val.feeds.replay_archive import ArchiveReplayFeedSource, resolve_static_gtfs

    if static_gtfs_zip is None and static_gtfs_dir is None:
        raise ValueError(
            "build_replay_run requires one of static_gtfs_zip or static_gtfs_dir"
        )

    route_ids = set(routes) if routes else None

    feed_source = ArchiveReplayFeedSource(
        vp_path, tu_path, start=start, end=end, route_ids=route_ids
    )
    snapshots = feed_source.snapshots
    if not snapshots:
        raise ValueError(
            "No archived VehiclePosition rows matched the given path/window/route filter"
        )

    first_tick = snapshots[0].header_timestamp
    first_date = dt.datetime.fromtimestamp(first_tick, tz=dt.UTC).date()

    if static_gtfs_zip is not None:
        static_zip_path = resolve_static_gtfs(local_zip=static_gtfs_zip)
    else:
        static_zip_path = resolve_static_gtfs(local_dir=static_gtfs_dir, target_date=first_date)

    static_loader = GtfsStaticLoader(static_zip_path)

    predictors = build_predictors(predictor_ids)
    _attach_static_loader(predictors, static_loader)

    return ReplayRunDeps(
        feed_source=feed_source,
        static_loader=static_loader,
        predictors=predictors,
        detectors=build_detectors(cfg),
        gtfs_path=None,
    )
