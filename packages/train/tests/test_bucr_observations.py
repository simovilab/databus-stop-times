"""Calendar resolution in stoptimes.train.sources.bucr_observations."""

from __future__ import annotations

import zipfile

import pandas as pd
from stoptimes.train.sources.bucr_observations import active_trips_fn


def _zip(tmp_path, files: dict[str, str]):
    path = tmp_path / "gtfs.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for name, text in files.items():
            zf.writestr(name, text)
    return path


CALENDAR = (
    "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n"
    "weekday,1,1,1,1,1,0,0,20260810,20261127\n"
)
TRIPS = pd.DataFrame({"trip_id": ["a", "b"], "service_id": ["weekday", "extra"]})


def test_weekdays_inside_the_calendar_run_and_holidays_do_not(tmp_path):
    gtfs = _zip(
        tmp_path,
        {
            "calendar.txt": CALENDAR,
            "calendar_dates.txt": (
                "service_id,date,exception_type\nweekday,20261012,2\nextra,20261017,1\n"
            ),
        },
    )
    active = active_trips_fn(gtfs, TRIPS)
    assert active("20261020") == {"a"}  # a Tuesday
    assert active("20261012") == set()  # Día de las Culturas, removed
    assert active("20261018") == set()  # a Sunday
    assert active("20261017") == {"b"}  # a Saturday added by exception
    assert active("20261130") == set()  # after the calendar ends (the feed expires 27 Nov)


def test_no_calendar_dates_file(tmp_path):
    active = active_trips_fn(_zip(tmp_path, {"calendar.txt": CALENDAR}), TRIPS)
    assert active("20261012") == {"a"}
