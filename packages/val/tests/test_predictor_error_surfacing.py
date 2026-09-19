"""test_predictor_error_surfacing — a failing predictor must say why (T6/F1 defect fix).

A third-party tester found that `val replay`'s ``Predictor errors: {pipeline:...: 103}``
line was the *only* trace of 103 identical predictor failures -- the actual
exception was nowhere, not on screen, not in ``summary.json``. That is a
defect in its own right (the same class of blindness that let the
``pipeline:`` adapter's one-line bug ship unnoticed): a predictor can fail on
every single tick and still read as a footnote under an otherwise cheerful
summary.

Covers, independent of any specific predictor implementation:

1. ``run_engine`` (``stoptimes.val.engine.runtime``) captures the first
   exception per predictor_id (type + message), plus a total-attempts count,
   in ``RunSummary``.
2. ``RunSummary.to_dict()``/``from_dict()`` round-trips both fields (so a
   non-interactive run keeps the cause in ``summary.json``).
3. The CLI's ``_report_run`` (``stoptimes.val.cli.main``) prints the real
   exception, and reads distinctly louder when a predictor failed on every
   tick it ran versus only sometimes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from stoptimes.core.domain.models import Stop, TripInstance, VehicleState
from stoptimes.val.cli.main import _report_run
from stoptimes.val.engine.runtime import (
    FeedSnapshot,
    ReplayFeedSource,
    RunSummary,
    run_engine,
)
from stoptimes.core.realtime.gtfsrt_json import parse_vp_feed
from stoptimes.core.gtfs.static import GtfsStaticLoader
from stoptimes.val.storage.sqlite import SqliteStorage

_FIXTURES = Path(__file__).parent / "fixtures"
_MINI_GTFS = _FIXTURES / "mini_gtfs.zip"


class _AlwaysFailsPredictor:
    """A predictor whose predict() raises on every call."""

    predictor_id = "always-fails"

    def predict(
        self,
        trip: TripInstance,
        upcoming_stops: list[Stop],
        vehicle_state: VehicleState,
        trip_updates: Any = None,
    ) -> list:
        raise RuntimeError("deliberately broken: missing feature column 'foo'")


class _FailsOnFirstTickOnlyPredictor:
    """A predictor that raises once, then works -- must not read as total failure."""

    predictor_id = "fails-once"

    def __init__(self) -> None:
        self._calls = 0

    def predict(
        self,
        trip: TripInstance,
        upcoming_stops: list[Stop],
        vehicle_state: VehicleState,
        trip_updates: Any = None,
    ) -> list:
        self._calls += 1
        if self._calls == 1:
            raise ValueError("transient: cold cache on first tick")
        return []


def _make_vp_snapshot(t: float, lat: float, lon: float, seq: int) -> FeedSnapshot:
    vp_dict = {
        "header": {"timestamp": t},
        "entity": [
            {
                "id": "e1",
                "vehicle": {
                    "trip": {
                        "trip_id": "trip1",
                        "route_id": "r1",
                        "direction_id": 0,
                        "start_date": "20240101",
                    },
                    "vehicle": {"id": "v1"},
                    "position": {"latitude": lat, "longitude": lon, "speed": 3.0},
                    "timestamp": t,
                    "current_status": "IN_TRANSIT_TO",
                    "current_stop_sequence": seq,
                },
            }
        ],
    }
    vp_ts, vp_records = parse_vp_feed(vp_dict)
    return FeedSnapshot(header_timestamp=vp_ts, vp_records=vp_records, tu_header_timestamp=None, tu_records=[])


def _three_tick_snapshots() -> list[FeedSnapshot]:
    return [
        _make_vp_snapshot(1000.0, 42.3601, -71.0589, 1),
        _make_vp_snapshot(1100.0, 42.3610, -71.0580, 2),
        _make_vp_snapshot(1200.0, 42.3620, -71.0570, 3),
    ]


@pytest.mark.asyncio
async def test_run_engine_captures_first_error_and_attempts_when_predictor_always_fails() -> None:
    static_loader = GtfsStaticLoader(_MINI_GTFS)
    storage = SqliteStorage(":memory:")
    predictor = _AlwaysFailsPredictor()

    summary = await run_engine(
        feed_source=ReplayFeedSource(_three_tick_snapshots()),
        static_loader=static_loader,
        predictors=[predictor],
        detectors=[],
        storage=storage,
        source_name="test",
    )

    assert summary.predictor_errors["always-fails"] == summary.ticks == 3
    assert summary.predictor_attempts["always-fails"] == 3
    assert summary.predictor_errors["always-fails"] == summary.predictor_attempts["always-fails"]
    assert "RuntimeError" in summary.predictor_first_error["always-fails"]
    assert "missing feature column 'foo'" in summary.predictor_first_error["always-fails"]


@pytest.mark.asyncio
async def test_run_engine_partial_failure_is_distinguishable_from_total_failure() -> None:
    static_loader = GtfsStaticLoader(_MINI_GTFS)
    storage = SqliteStorage(":memory:")
    predictor = _FailsOnFirstTickOnlyPredictor()

    summary = await run_engine(
        feed_source=ReplayFeedSource(_three_tick_snapshots()),
        static_loader=static_loader,
        predictors=[predictor],
        detectors=[],
        storage=storage,
        source_name="test",
    )

    assert summary.predictor_errors["fails-once"] == 1
    assert summary.predictor_attempts["fails-once"] == 3
    assert summary.predictor_errors["fails-once"] != summary.predictor_attempts["fails-once"]
    assert "ValueError" in summary.predictor_first_error["fails-once"]
    assert "transient: cold cache on first tick" in summary.predictor_first_error["fails-once"]


def test_run_summary_round_trips_first_error_and_attempts() -> None:
    summary = RunSummary(
        ticks=3,
        predictor_errors={"p1": 3},
        predictor_first_error={"p1": "RuntimeError: boom"},
        predictor_attempts={"p1": 3},
    )
    restored = RunSummary.from_dict(summary.to_dict())

    assert restored.predictor_errors == summary.predictor_errors
    assert restored.predictor_first_error == summary.predictor_first_error
    assert restored.predictor_attempts == summary.predictor_attempts


def test_report_run_names_the_real_exception_and_flags_total_failure(capsys) -> None:
    summary = RunSummary(
        ticks=3,
        predictor_errors={"always-fails": 3},
        predictor_first_error={"always-fails": "RuntimeError: missing feature column 'foo'"},
        predictor_attempts={"always-fails": 3},
    )

    _report_run(summary)
    captured = capsys.readouterr()
    output = captured.out + captured.err

    assert "Predictor errors" in output
    assert "RuntimeError" in output
    assert "missing feature column 'foo'" in output
    assert "FAILED EVERY TICK" in output


def test_report_run_does_not_shout_about_a_partial_failure(capsys) -> None:
    summary = RunSummary(
        ticks=3,
        predictor_errors={"fails-once": 1},
        predictor_first_error={"fails-once": "ValueError: transient: cold cache on first tick"},
        predictor_attempts={"fails-once": 3},
    )

    _report_run(summary)
    captured = capsys.readouterr()
    output = captured.out + captured.err

    assert "ValueError" in output
    assert "transient: cold cache on first tick" in output
    assert "FAILED EVERY TICK" not in output
