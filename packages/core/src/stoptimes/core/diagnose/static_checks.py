"""Checks that need the GTFS static feed — trip/route/stop coverage against
what was actually observed, and whether the static feed itself carries what
map-matching needs (`shapes.txt`, `shape_dist_traveled`).

Every function takes a `GtfsStaticLoader | None` (see `stoptimes.core.gtfs.static`)
and the `StaticFeed` describing which tables it has (see
`stoptimes.core.contracts.data`), plus whichever Observation frame(s) it
needs — never re-parses the zip itself (`stoptimes.core.readers.read_static`
already did that). `None`/empty inputs degrade to an explanatory finding,
never an exception, matching `realtime_checks`.
"""

from __future__ import annotations

import pandas as pd
from stoptimes.core.contracts.data import StaticFeed
from stoptimes.core.diagnose._util import bounded_samples, severity_by_share
from stoptimes.core.diagnose.models import DiagnoseParams, Finding, Severity
from stoptimes.core.gtfs.static import GtfsStaticLoader


def check_static_presence(
    loader: GtfsStaticLoader | None, static_feed: StaticFeed | None, params: DiagnoseParams
) -> Finding | None:
    if static_feed is not None:
        return None
    return Finding(
        id="static_missing",
        severity=Severity.BLOCKER,
        title="No GTFS static feed provided",
        explanation="No GTFS static .zip was supplied.",
        blocks=(
            "Nothing here can determine whether observed trips match the published schedule, "
            "whether shapes exist to support map-matching, or which stops/routes are covered — "
            "and `stoptimes.core.arrivals.derive_arrivals` requires a static feed unconditionally."
        ),
        recommendation="Provide the agency's GTFS static .zip (`diagnose --static PATH_OR_URL`).",
    )


def check_static_required_tables(static_feed: StaticFeed | None, params: DiagnoseParams) -> Finding | None:
    if static_feed is None:
        return None
    missing = static_feed.missing_required()
    if not missing:
        return None
    return Finding(
        id="static_required_tables_missing",
        severity=Severity.BLOCKER,
        title="GTFS static feed is missing required table(s)",
        explanation=f"Missing required table(s): {missing}.",
        blocks="Without stops/trips/stop_times, no static-dependent check below can run, and the feed cannot be loaded for training at all.",
        recommendation=f"Add {missing} to the GTFS static .zip.",
        samples=list(missing),
    )


def check_shapes(
    loader: GtfsStaticLoader | None, static_feed: StaticFeed | None, params: DiagnoseParams
) -> Finding | None:
    if static_feed is None or loader is None:
        return None
    if "shapes" not in static_feed.tables or loader.shapes.empty:
        return Finding(
            id="shapes_missing",
            severity=Severity.DEGRADED,
            title="No shapes.txt in the GTFS static feed",
            explanation="shapes.txt is absent or empty.",
            blocks=(
                "`stoptimes.core.arrivals._resolve_trip_geometry` gets no polyline for any trip, "
                "so map-matched (progress-based) arrival derivation is skipped entirely in favor of "
                "a cruder stop-proximity method — every trip, not just some."
            ),
            recommendation="Add shapes.txt to the GTFS static feed with each trip's shape_id populated in trips.txt.",
        )
    shapes = loader.shapes
    if "shape_dist_traveled" not in shapes.columns:
        missing_share = 1.0
        n_missing = int(len(shapes))
    else:
        missing_mask = shapes["shape_dist_traveled"].isna()
        n_missing = int(missing_mask.sum())
        missing_share = n_missing / len(shapes) if len(shapes) else 0.0
    if n_missing == 0:
        return None
    affected_shapes: list[str] = []
    if "shape_id" in shapes.columns:
        if "shape_dist_traveled" in shapes.columns:
            rows = shapes.loc[shapes["shape_dist_traveled"].isna(), "shape_id"]
        else:
            rows = shapes["shape_id"]
        affected_shapes = sorted(rows.dropna().unique().tolist())
    return Finding(
        id="shape_dist_traveled_missing",
        severity=Severity.DEGRADED,
        title="shapes.txt points missing shape_dist_traveled",
        explanation=(
            f"{n_missing} of {len(shapes)} shape points ({missing_share:.0%}) have no "
            "shape_dist_traveled value."
        ),
        blocks=(
            "`stoptimes.core.spatial.polyline.build_polyline` falls back to cumulative haversine "
            "distance between shape points for any shape with even one point missing this value — "
            "accurate when points are dense, but understates distance across sparse or curved arcs "
            "(the vendor-authored value tracks road/rail curvature that straight segments between "
            "sparse points miss)."
        ),
        recommendation="Populate shape_dist_traveled for every point in every shape, or accept haversine-approximated distances for the affected shapes.",
        count=n_missing,
        total=int(len(shapes)),
        share=missing_share,
        samples=bounded_samples(affected_shapes, params),
    )


def _observed_ids(vp_df: pd.DataFrame | None, tu_df: pd.DataFrame | None, column: str) -> set[str]:
    ids: set[str] = set()
    for df in (vp_df, tu_df):
        if df is not None and column in df.columns:
            ids |= set(df[column].dropna().astype(str))
            ids.discard("")
    return ids


def check_trip_id_match(
    loader: GtfsStaticLoader | None,
    vp_df: pd.DataFrame | None,
    tu_df: pd.DataFrame | None,
    params: DiagnoseParams,
) -> list[Finding]:
    if loader is None:
        return []
    observed = _observed_ids(vp_df, tu_df, "trip_id")
    if not observed:
        return []
    static_trip_ids = set(loader.trips["trip_id"].astype(str)) if "trip_id" in loader.trips.columns else set()

    findings: list[Finding] = []

    unmatched = observed - static_trip_ids
    if unmatched:
        share = len(unmatched) / len(observed)
        findings.append(
            Finding(
                id="trip_ids_unmatched_static",
                severity=severity_by_share(share, params),
                title="Observed trip_ids not found in the static feed",
                explanation=f"{len(unmatched)} of {len(observed)} distinct observed trip_ids ({share:.0%}) do not appear in trips.txt.",
                blocks="These trips cannot be joined to a schedule, a shape, or a stop sequence — arrivals cannot be derived and any TripUpdate for them cannot be validated.",
                recommendation="Check for a static/realtime version mismatch (a schedule change not yet reflected in the realtime trip_id space), or an agency-specific trip_id transform that needs to be undone before joining.",
                count=len(unmatched),
                total=len(observed),
                share=share,
                samples=bounded_samples(sorted(unmatched), params),
            )
        )

    if static_trip_ids:
        never_observed = static_trip_ids - observed
        if never_observed:
            share = len(never_observed) / len(static_trip_ids)
            # Below the majority threshold this is informational, not degraded: a
            # single-day observation window naturally misses trips that simply
            # hadn't run yet, and that alone shouldn't read as a data problem.
            severity = severity_by_share(share, params) if share >= params.majority_threshold else Severity.INFO
            findings.append(
                Finding(
                    id="static_trips_never_observed",
                    severity=severity,
                    title="Scheduled trips never observed in realtime data",
                    explanation=f"{len(never_observed)} of {len(static_trip_ids)} scheduled trip_ids ({share:.0%}) never appear in the observations.",
                    blocks="These trips get zero training examples; if the observation window doesn't cover every service day, this may simply reflect trips that hadn't run yet rather than a real gap.",
                    recommendation="Confirm the observation window's date coverage before treating this as a recording problem; if it does cover the full schedule, check whether these trips actually operate.",
                    count=len(never_observed),
                    total=len(static_trip_ids),
                    share=share,
                    samples=bounded_samples(sorted(never_observed), params),
                )
            )

    return findings


def check_route_match(
    loader: GtfsStaticLoader | None,
    vp_df: pd.DataFrame | None,
    tu_df: pd.DataFrame | None,
    static_feed: StaticFeed | None,
    params: DiagnoseParams,
) -> list[Finding]:
    if loader is None or static_feed is None or "routes" not in static_feed.tables or loader.routes.empty:
        return []
    observed = _observed_ids(vp_df, tu_df, "route_id")
    if not observed:
        return []
    static_route_ids = set(loader.routes["route_id"].astype(str)) if "route_id" in loader.routes.columns else set()
    if not static_route_ids:
        return []

    findings: list[Finding] = []
    unmatched = observed - static_route_ids
    if unmatched:
        share = len(unmatched) / len(observed)
        findings.append(
            Finding(
                id="route_ids_unmatched_static",
                severity=severity_by_share(share, params),
                title="Observed route_ids not found in the static feed",
                explanation=f"{len(unmatched)} of {len(observed)} distinct observed route_ids ({share:.0%}) do not appear in routes.txt.",
                blocks="Per-route reporting (coverage, model selection by route) is meaningless for these routes.",
                recommendation="Check for a static/realtime version mismatch or a route_id renaming between feeds.",
                count=len(unmatched),
                total=len(observed),
                share=share,
                samples=bounded_samples(sorted(unmatched), params),
            )
        )

    never_observed = static_route_ids - observed
    if never_observed:
        share = len(never_observed) / len(static_route_ids)
        findings.append(
            Finding(
                id="static_routes_never_observed",
                severity=Severity.INFO,
                title="Scheduled routes never observed in realtime data",
                explanation=f"{len(never_observed)} of {len(static_route_ids)} scheduled route_ids ({share:.0%}) never appear in the observations.",
                blocks="No training data exists for these routes; per-route model selection has nothing to select for them.",
                recommendation="Confirm coverage before assuming these routes are broken — they may simply run outside the observation window.",
                count=len(never_observed),
                total=len(static_route_ids),
                share=share,
                samples=bounded_samples(sorted(never_observed), params),
            )
        )
    return findings


def check_stop_coverage(
    loader: GtfsStaticLoader | None,
    vp_df: pd.DataFrame | None,
    tu_df: pd.DataFrame | None,
    params: DiagnoseParams,
) -> Finding | None:
    if loader is None or loader.stops.empty or "stop_id" not in loader.stops.columns:
        return None
    observed = _observed_ids(vp_df, tu_df, "stop_id")
    if not observed:
        return None
    static_stop_ids = set(loader.stops["stop_id"].astype(str))
    if not static_stop_ids:
        return None
    never_observed = static_stop_ids - observed
    if not never_observed:
        return None
    share = len(never_observed) / len(static_stop_ids)
    severity = severity_by_share(share, params) if share >= params.majority_threshold else Severity.INFO
    return Finding(
        id="static_stops_never_observed",
        severity=severity,
        title="Scheduled stops never seen in realtime data",
        explanation=f"{len(never_observed)} of {len(static_stop_ids)} scheduled stop_ids ({share:.0%}) never appear as a `stop_id` in the observations.",
        blocks="These stops get no arrival labels; if used as an evaluation target, error metrics there are undefined, not zero.",
        recommendation="Check whether vehicle-position/trip-update stop_id reporting is enabled for the full network, not just a subset.",
        count=len(never_observed),
        total=len(static_stop_ids),
        share=share,
        samples=bounded_samples(sorted(never_observed), params),
    )
