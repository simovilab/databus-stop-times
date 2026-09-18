"""Tests for readers.read_static: wraps GtfsStaticLoader, reports missing tables."""

from __future__ import annotations

import zipfile
from pathlib import Path

from stoptimes.core.gtfs.static import GtfsStaticLoader
from stoptimes.core.readers import read_static

STOPS_TXT = "stop_id,stop_lat,stop_lon,stop_name\ns1,9.93,-84.08,Stop 1\ns2,9.94,-84.09,Stop 2\n"
TRIPS_TXT = "trip_id,route_id,service_id,direction_id,shape_id\nt1,r1,wk,0,sh1\n"
STOP_TIMES_TXT = (
    "trip_id,stop_id,stop_sequence,arrival_time,departure_time\n"
    "t1,s1,1,08:00:00,08:00:00\nt1,s2,2,08:05:00,08:05:00\n"
)
ROUTES_TXT = "route_id,route_short_name\nr1,Red\n"
CALENDAR_TXT = (
    "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n"
    "wk,1,1,1,1,1,0,0,20260101,20261231\n"
)


def _write_zip(path: Path, members: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return path


def test_read_static_complete_feed_wraps_loader_and_reports_all_tables(tmp_path):
    zip_path = _write_zip(
        tmp_path / "gtfs.zip",
        {
            "stops.txt": STOPS_TXT,
            "trips.txt": TRIPS_TXT,
            "stop_times.txt": STOP_TIMES_TXT,
            "routes.txt": ROUTES_TXT,
            "calendar.txt": CALENDAR_TXT,
        },
    )
    loader, feed = read_static(str(zip_path), "mbta")
    assert isinstance(loader, GtfsStaticLoader)
    assert feed.source_id == "mbta"
    assert feed.missing_required() == []
    for table in ("stops", "trips", "stop_times", "routes", "calendar"):
        assert feed.has(table)
    assert not feed.has("shapes")
    assert not feed.has("calendar_dates")
    assert len(loader.trips) == 1


def test_read_static_missing_required_table_reports_without_crashing(tmp_path):
    members = {"stops.txt": STOPS_TXT, "trips.txt": TRIPS_TXT}
    zip_path = _write_zip(tmp_path / "incomplete.zip", members)
    loader, feed = read_static(str(zip_path), "mbta")
    assert loader is None
    assert feed.missing_required() == ["stop_times"]
    assert feed.has("stops")
    assert feed.has("trips")


def test_read_static_reports_calendar_dates_presence_without_parsing(tmp_path):
    zip_path = _write_zip(
        tmp_path / "gtfs.zip",
        {
            "stops.txt": STOPS_TXT,
            "trips.txt": TRIPS_TXT,
            "stop_times.txt": STOP_TIMES_TXT,
            "calendar_dates.txt": "service_id,date,exception_type\nwk,20260704,2\n",
        },
    )
    loader, feed = read_static(str(zip_path), "mbta")
    assert loader is not None
    assert feed.has("calendar_dates")
    # GtfsStaticLoader itself never parses calendar_dates — nothing to assert
    # on `loader` for it; only StaticFeed reports it.
    assert not hasattr(loader, "calendar_dates")
