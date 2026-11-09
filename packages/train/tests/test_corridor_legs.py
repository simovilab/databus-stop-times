"""Turning-point leg segmentation (stoptimes.train.sources.corridor_legs) on a synthetic corridor.

The corridor runs north from two low-end terminals (A at 0 m, B at 300 m) to a high-end
terminal H (4 km), with H2 20 m beside it (bUCR: Odontología's two stops), and an
up-direction detour variant through a stop D 300 m east of the corridor (bUCR: the milla).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
from stoptimes.core.domain.models import Stop
from stoptimes.core.spatial.polyline import assign_stops_monotonic, build_polyline
from stoptimes.train.sources.bucr_gtfs import RouteDirectionCandidate
from stoptimes.train.sources.corridor_legs import LegParams, corridor_layout, segment_legs

LAT0, LON0 = 9.93, -84.05
M_PER_DEG = 111_000.0


def _ll(y_m: float, x_m: float = 0.0) -> tuple[float, float]:
    return LAT0 + y_m / M_PER_DEG, LON0 + x_m / (M_PER_DEG * math.cos(math.radians(LAT0)))


PLACES = {
    "A": (0, 0),
    "B": (300, 0),
    "M1": (1500, 0),
    "M2": (3000, 0),
    "H": (4000, 0),
    "H2": (4020, 0),
    "D": (2000, 300),
}


def _candidate(
    shape_id: str, direction: int, waypoints: Sequence[tuple[float, float]], stops: list[str]
):
    pts = []
    for (y0, x0), (y1, x1) in zip(waypoints[:-1], waypoints[1:], strict=True):
        n = max(2, int(math.hypot(y1 - y0, x1 - x0) // 50))
        pts += [_ll(y0 + (y1 - y0) * k / n, x0 + (x1 - x0) * k / n) for k in range(n)]
    pts.append(_ll(*waypoints[-1]))
    poly = build_polyline(pts)
    st = [
        Stop(stop_id=s, stop_sequence=i + 1, lat=_ll(*PLACES[s])[0], lon=_ll(*PLACES[s])[1])
        for i, s in enumerate(stops)
    ]
    return RouteDirectionCandidate(
        "R", direction, shape_id, f"trip-{shape_id}", poly, assign_stops_monotonic(st, poly)
    )


DETOUR = [(0, 0), (1500, 0), (1800, 0), (1800, 300), (2200, 300), (2200, 0), (4000, 0)]
CANDIDATES = [
    _candidate("up_A", 0, [(0, 0), (4000, 0)], ["A", "M1", "M2", "H"]),
    _candidate("up_B", 0, [(300, 0), (4000, 0)], ["B", "M1", "M2", "H"]),
    _candidate("up_A_detour", 0, DETOUR, ["A", "M1", "D", "M2", "H"]),
    _candidate("down_A", 1, [(4020, 0), (0, 0)], ["H2", "M2", "M1", "A"]),
    _candidate("down_B", 1, [(4020, 0), (300, 0)], ["H2", "M2", "M1", "B"]),
]
LAYOUT = corridor_layout(CANDIDATES)


class _Trace:
    """Pings every ``dt`` seconds: drive between waypoints at ``speed`` m/s, or wait."""

    def __init__(self, dt: float = 15.0, speed: float = 8.0) -> None:
        self.dt, self.speed = dt, speed
        self.t = pd.Timestamp("2026-10-20T12:00:00Z")
        self.rows: list[dict] = []
        self.pos = (0.0, 0.0)

    def _ping(self, y: float, x: float, stopped: bool) -> None:
        lat, lon = _ll(y, x)
        self.rows.append({"ts": self.t, "lat": lat, "lon": lon, "stopped": stopped})
        self.t += pd.Timedelta(seconds=self.dt)

    def at(self, place: str) -> _Trace:
        self.pos = tuple(float(v) for v in PLACES[place])
        return self

    def wait(self, seconds: float) -> _Trace:
        for _ in range(int(seconds // self.dt) + 1):
            self._ping(*self.pos, stopped=True)
        return self

    def drive(self, *waypoints: tuple[float, float] | str) -> _Trace:
        for w in waypoints:
            y1, x1 = PLACES[w] if isinstance(w, str) else w
            y0, x0 = self.pos
            n = max(1, int(math.hypot(y1 - y0, x1 - x0) / (self.speed * self.dt)))
            for k in range(1, n + 1):
                self._ping(y0 + (y1 - y0) * k / n, x0 + (x1 - x0) * k / n, stopped=False)
            self.pos = (float(y1), float(x1))
        return self

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def _legs(trace: _Trace, **params):
    day = trace.frame()
    return day, segment_legs(day, LAYOUT, LegParams(**params))


def _place_of(day: pd.DataFrame, i: int) -> str:
    y = (day["lat"].iloc[i] - LAT0) * M_PER_DEG
    return min(("A", "B", "H", "H2"), key=lambda p: abs(PLACES[p][0] - y))


def test_layout_is_derived_from_the_shapes():
    assert set(LAYOUT.low_terminals) == {"A", "B"}
    assert set(LAYOUT.high_terminals) == {"H", "H2"}
    assert set(LAYOUT.detour_stops) == {"D"}
    assert LAYOUT.up_direction == 0
    assert LAYOUT.reference.shape_id in {"up_A", "up_B"}
    assert LAYOUT.variants[(0, "A", False)] == "up_A"
    assert LAYOUT.variants[(0, "A", True)] == "up_A_detour"
    assert LAYOUT.variants[(1, "B", False)] == "down_B"


def test_round_trips_split_at_terminal_departures_with_variants():
    tr = _Trace().at("A").wait(120).drive("H").wait(180).drive("B").wait(150).drive("H").wait(60)
    day, legs = _legs(tr)
    assert [(leg.direction_id, leg.shape_id) for leg in legs] == [
        (0, "up_A"),
        (1, "down_B"),
        (0, "up_B"),
    ]
    assert _place_of(day, legs[0].start) == "A" and day["stopped"].iloc[legs[0].start]
    assert _place_of(day, legs[1].start) in {"H", "H2"}
    assert _place_of(day, legs[2].start) == "B"
    assert all(a.end <= b.start for a, b in zip(legs[:-1], legs[1:], strict=True))


def test_a_quick_turnaround_still_starts_the_next_leg_at_the_terminal():
    """The old dwell rule (>= 90 s) glued a 30 s turnaround to the previous trip."""
    tr = _Trace().at("H").wait(60).drive("A").wait(30).drive("H").wait(30)
    day, legs = _legs(tr)
    assert [leg.shape_id for leg in legs] == ["down_A", "up_A"]
    assert _place_of(day, legs[1].start) == "A"


def test_the_last_layover_in_the_zone_is_the_departure():
    tr = _Trace().at("H").wait(60).drive("A").wait(180).drive("B").wait(150).drive("H").wait(30)
    _, legs = _legs(tr)
    assert legs[-1].shape_id == "up_B"


def test_a_short_stop_after_the_layover_does_not_move_the_departure():
    tr = _Trace().at("H").wait(60).drive("A").wait(180).drive("B").wait(45).drive("H").wait(30)
    _, legs = _legs(tr)
    assert legs[-1].shape_id == "up_A"


def test_visiting_a_detour_stop_selects_the_detour_variant():
    tr = (
        _Trace()
        .at("A")
        .wait(60)
        .drive((1500, 0), (1800, 0), (1800, 300), (2000, 300), (2200, 300), (2200, 0), "H")
        .wait(60)
    )
    _, legs = _legs(tr)
    assert [leg.shape_id for leg in legs] == ["up_A_detour"]


def test_a_long_wait_away_from_terminals_ends_the_leg():
    tr = _Trace().at("A").wait(60).drive((2500, 0)).wait(600).drive("H").wait(60)
    day, legs = _legs(tr)
    assert len(legs) == 1 and legs[0].shape_id == "up_A"
    end_y = (day["lat"].iloc[legs[0].end] - LAT0) * M_PER_DEG
    assert end_y == pytest.approx(2500, abs=5)
    assert day["stopped"].iloc[legs[0].end]  # the first ping of the break, nothing after it


def test_a_long_offline_gap_drops_the_leg():
    tr = _Trace().at("A").wait(60).drive((2000, 0))
    tr.t += pd.Timedelta(minutes=20)
    tr.drive("H").wait(60)
    _, legs = _legs(tr)
    assert legs == []


def test_too_few_pings_is_no_leg():
    day = _Trace().at("A").wait(30).frame()
    assert segment_legs(day, LAYOUT) == []


def test_segmentation_is_deterministic():
    tr = _Trace().at("A").wait(120).drive("H").wait(180).drive("B").wait(150)
    day = tr.frame()
    assert segment_legs(day, LAYOUT) == segment_legs(day.copy(), LAYOUT)
    assert np.all(np.diff(day["ts"].to_numpy()) > np.timedelta64(0, "s"))
