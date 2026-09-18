"""Feed quality diagnostics: "can this GTFS static + realtime data support
ETA modelling at all?" (Phase 3, track U3, docs/PLAN.md §7.1's Newcomer
story).

Public entry point: `diagnose_feed(...)`, a pure function over
already-`stoptimes.core.readers`-read frames — no I/O here, so it works
identically over a live feed, a recorded archive, or a hand-built fixture,
and is unit-testable without a real feed. It returns a `DiagnosticReport`
(data, not printing); the CLI (`stoptimes diagnose`) and `stoptimes data
inspect` both render it.

Every check tolerates a missing piece (no static feed, no shapes, no
vehicle positions, no trip updates) by reporting that fact as a `Finding`
rather than raising — a newcomer's feed is exactly the broken case this
exists to describe. See `vp_checks.py`/`tu_checks.py` for realtime-only
checks and `static_checks.py` for checks that need the GTFS static feed.
"""

from __future__ import annotations

import pandas as pd
from stoptimes.core.contracts.data import StaticFeed
from stoptimes.core.diagnose import static_checks, tu_checks, vp_checks
from stoptimes.core.diagnose.models import (
    SEVERITY_ORDER,
    DiagnoseParams,
    DiagnosticReport,
    Finding,
    Severity,
)
from stoptimes.core.gtfs.static import GtfsStaticLoader

__all__ = [
    "DiagnoseParams",
    "DiagnosticReport",
    "Finding",
    "Severity",
    "diagnose_feed",
]


def diagnose_feed(
    *,
    vp_df: pd.DataFrame | None = None,
    tu_df: pd.DataFrame | None = None,
    static_loader: GtfsStaticLoader | None = None,
    static_feed: StaticFeed | None = None,
    params: DiagnoseParams | None = None,
) -> DiagnosticReport:
    """Build a `DiagnosticReport` from whatever data is available.

    Args:
        vp_df: an `OBSERVATION_VP_SCHEMA`-shaped frame (see
            `stoptimes.core.readers.read_observations`), or `None`/empty if
            no vehicle positions were recorded.
        tu_df: an `OBSERVATION_TU_SCHEMA`-shaped frame, or `None`/empty.
        static_loader: a loaded `GtfsStaticLoader` (see
            `stoptimes.core.readers.read_static`), or `None` if no static
            feed was supplied or it was missing a required table.
        static_feed: the `StaticFeed` describing which tables `static_loader`
            has — required even when `static_loader` is `None`, so a
            "missing required table" finding can still be produced (pass
            `None` for both only when no static location was given at all).
        params: named thresholds; `DiagnoseParams()` defaults are used and
            echoed back on the report when omitted.

    Never raises on missing input — every check degrades to an explanatory
    `Finding` instead. The only way to get an empty `findings` list is a
    feed with no detectable problems at all.
    """
    p = params or DiagnoseParams()
    findings: list[Finding] = []

    def add(finding: Finding | None) -> None:
        if finding is not None:
            findings.append(finding)

    def add_all(items: list[Finding]) -> None:
        findings.extend(items)

    # --- realtime-only checks ---
    add(vp_checks.check_vp_presence(vp_df, p))
    add(vp_checks.check_vp_position_validity(vp_df, p))
    add(vp_checks.check_vp_gaps(vp_df, p))
    add(vp_checks.check_vp_duplicates(vp_df, p))
    add(vp_checks.check_vp_out_of_order(vp_df, p))
    add(vp_checks.check_vp_staleness(vp_df, p))

    add(tu_checks.check_tu_presence(tu_df, p))
    add(tu_checks.check_tu_prediction_validity(tu_df, p))
    add(tu_checks.check_tu_duplicates(tu_df, p))
    add(tu_checks.check_tu_out_of_order(tu_df, p))
    add(tu_checks.check_tu_staleness(tu_df, p))

    # --- static-dependent checks ---
    add(static_checks.check_static_presence(static_loader, static_feed, p))
    add(static_checks.check_static_required_tables(static_feed, p))
    add(static_checks.check_shapes(static_loader, static_feed, p))
    add_all(static_checks.check_trip_id_match(static_loader, vp_df, tu_df, p))
    add_all(static_checks.check_route_match(static_loader, vp_df, tu_df, static_feed, p))
    add(static_checks.check_stop_coverage(static_loader, vp_df, tu_df, p))

    verdict_severity, verdict = _summarize(findings)

    return DiagnosticReport(
        static_present=static_feed is not None,
        vp_present=vp_df is not None and len(vp_df) > 0,
        tu_present=tu_df is not None and len(tu_df) > 0,
        static_tables=list(static_feed.tables) if static_feed is not None else [],
        vp_rows=int(len(vp_df)) if vp_df is not None else 0,
        tu_rows=int(len(tu_df)) if tu_df is not None else 0,
        findings=findings,
        verdict=verdict,
        verdict_severity=verdict_severity,
        params=p,
    )


def _summarize(findings: list[Finding]) -> tuple[Severity, str]:
    """Worst-severity-wins verdict, with a plain-English action for a newcomer."""
    if not findings:
        return Severity.INFO, "OK — no problems found. This feed looks workable for ETA modelling."

    worst = max((f.severity for f in findings), key=lambda s: SEVERITY_ORDER[s])
    blockers = [f for f in findings if f.severity is Severity.BLOCKER]
    degraded = [f for f in findings if f.severity is Severity.DEGRADED]

    if worst is Severity.BLOCKER:
        names = ", ".join(f.title for f in blockers)
        return (
            Severity.BLOCKER,
            f"BLOCKER — {len(blockers)} issue(s) will stop ETA modelling outright: {names}. "
            "Fix these before investing further time.",
        )
    if worst is Severity.DEGRADED:
        names = ", ".join(f.title for f in degraded)
        return (
            Severity.DEGRADED,
            f"DEGRADED — modelling is possible, but {len(degraded)} issue(s) will cost accuracy "
            f"or coverage: {names}.",
        )
    return Severity.INFO, "OK — only informational findings; no blocking or accuracy-degrading issues found."
