"""`stoptimes serve` — run inference and publish GTFS-RT TripUpdates."""

from __future__ import annotations

from stoptimes.cli._util import planned


def serve() -> None:
    """Run inference and publish GTFS-RT TripUpdates (HTTP, file, MQTT, ...)."""
    planned("serve", "Phase 2 · track T4 (serving pipeline)")
