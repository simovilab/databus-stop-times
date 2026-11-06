"""GtfsStaticLoader.agency_timezone / trip_schedule on a tiny synthetic feed."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from stoptimes.core.features.schedule import TripSchedule
from stoptimes.core.gtfs.static import GtfsStaticLoader

STOPS = "stop_id,stop_lat,stop_lon\na,9.9,-84.0\nb,9.91,-84.0\nc,9.92,-84.0\n"
TRIPS = "trip_id,route_id,service_id\nt1,r,wk\nt2,r,wk\n"
ST = (
    "trip_id,stop_id,stop_sequence,arrival_time,departure_time,shape_dist_traveled\n"
    "t1,c,3,25:10:00,25:10:00,1000\n"
    "t1,a,1,24:50:00,24:50:00,0\n"
    "t1,b,2,,,500\n"
    "t2,a,1,08:00:00,08:00:00,\n"
)
AGENCY = "agency_id,agency_name,agency_url,agency_timezone\n1,X,http://x,America/Costa_Rica\n"


def _zip(path: Path, **extra: str) -> Path:
    members = {"stops.txt": STOPS, "trips.txt": TRIPS, "stop_times.txt": ST, **extra}
    with zipfile.ZipFile(path, "w") as zf:
        for name, text in members.items():
            zf.writestr(name, text)
    return path


def test_agency_timezone_present(tmp_path):
    loader = GtfsStaticLoader(_zip(tmp_path / "f.zip", **{"agency.txt": AGENCY}))
    assert loader.agency_timezone() == "America/Costa_Rica"


def test_agency_timezone_absent(tmp_path):
    assert GtfsStaticLoader(_zip(tmp_path / "f.zip")).agency_timezone() is None


def test_trip_schedule_sorted_interpolated_past_midnight(tmp_path):
    loader = GtfsStaticLoader(_zip(tmp_path / "f.zip"))
    sched = loader.trip_schedule("t1")
    assert isinstance(sched, TripSchedule)
    assert sched.stop_ids == ("a", "b", "c")
    assert sched.stop_sequences == (1, 2, 3)
    assert sched.offsets_s[0] == 24 * 3600 + 50 * 60
    assert sched.offsets_s[2] == 25 * 3600 + 10 * 60
    assert sched.offsets_s[1] == pytest.approx((sched.offsets_s[0] + sched.offsets_s[2]) / 2)
    assert sched.interpolated == (False, True, False)


def test_trip_schedule_cached_and_unknown(tmp_path):
    loader = GtfsStaticLoader(_zip(tmp_path / "f.zip"))
    assert loader.trip_schedule("t2") is loader.trip_schedule("t2")
    with pytest.raises(ValueError):
        loader.trip_schedule("nope")
