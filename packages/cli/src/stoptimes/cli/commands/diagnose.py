"""`stoptimes diagnose` — report GTFS / GTFS-RT feed quality problems."""

from __future__ import annotations

from stoptimes.cli._util import planned


def diagnose() -> None:
    """Report GTFS / GTFS-RT feed quality problems before training on them."""
    planned("diagnose", "Phase 3 · track U3 (feed diagnostics)")
