"""Tests for stoptimes.core.diagnose: pure over already-read Observation
frames, no I/O. Fixtures are built by hand (same style as
test_catalog_inspect.py) rather than through the readers, so each check can
be exercised in isolation.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest
from stoptimes.core.contracts.data import StaticFeed
from stoptimes.core.diagnose import DiagnoseParams, Severity, diagnose_feed
from stoptimes.core.gtfs.static import GtfsStaticLoader

UTC = "UTC"


def _ts(*values: str) -> pd.Series:
    return pd.to_datetime(pd.Series(list(values)), utc=True)


def _vp_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    if "ingested_at" in df.columns:
        df["ingested_at"] = pd.to_datetime(df["ingested_at"], utc=True)
    return df


def _tu_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in ("ts", "ingested_at", "arrival_time", "departure_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    return df


# ---------------------------------------------------------------------------
# Presence / missing-piece robustness (the most important property)
# ---------------------------------------------------------------------------


def test_everything_missing_reports_blockers_not_a_crash():
    report = diagnose_feed()
    assert report.verdict_severity is Severity.BLOCKER
    ids = {f.id for f in report.findings}
    assert "vp_missing" in ids
    assert "tu_missing" in ids
    assert "static_missing" in ids


def test_only_trip_updates_present_is_the_newcomer_broken_case():
    """No shapes, no vehicle positions, only trip updates: must diagnose
    cleanly, not crash, and call out that map-matching is impossible."""
    tu_df = _tu_frame(
        [
            {
                "trip_id": "t1",
                "start_date": "20260101",
                "stop_id": "s1",
                "stop_sequence": 1,
                "ts": "2026-01-01T08:00:00Z",
                "arrival_time": "2026-01-01T08:00:00Z",
            }
        ]
    )
    report = diagnose_feed(tu_df=tu_df)
    assert report.vp_present is False
    assert report.tu_present is True
    ids = {f.id for f in report.findings}
    assert "vp_missing" in ids
    assert "static_missing" in ids
    assert report.verdict_severity is Severity.BLOCKER  # no static feed


def test_empty_frames_behave_like_none():
    report = diagnose_feed(vp_df=pd.DataFrame(), tu_df=pd.DataFrame())
    assert report.vp_present is False
    assert report.tu_present is False


def test_clean_feed_has_only_info_findings():
    vp_df = _vp_frame(
        [
            {
                "trip_id": "t1",
                "route_id": "r1",
                "vehicle_id": "v1",
                "lat": 0.0,
                "lon": 0.0,
                "ts": f"2026-01-01T08:00:{i:02d}Z",
                "ingested_at": f"2026-01-01T08:00:{i:02d}Z",
            }
            for i in range(0, 60, 15)
        ]
    )
    report = diagnose_feed(vp_df=vp_df)
    vp_findings = [f for f in report.findings if f.id.startswith("vp_")]
    assert vp_findings, "expected at least the informational gap-distribution finding"
    assert all(f.severity is not Severity.BLOCKER for f in vp_findings)
    assert all(f.id != "vp_stale_timestamps" for f in vp_findings)


# ---------------------------------------------------------------------------
# Vehicle-position checks
# ---------------------------------------------------------------------------


def test_vp_gap_detection_and_distribution():
    rows = [
        {"trip_id": "t1", "route_id": "r1", "vehicle_id": "v1", "lat": 0.0, "lon": 0.0,
         "ts": "2026-01-01T08:00:00Z", "ingested_at": "2026-01-01T08:00:00Z"},
        {"trip_id": "t1", "route_id": "r1", "vehicle_id": "v1", "lat": 0.0, "lon": 0.0,
         "ts": "2026-01-01T08:00:10Z", "ingested_at": "2026-01-01T08:00:10Z"},
        # 20-minute gap
        {"trip_id": "t1", "route_id": "r1", "vehicle_id": "v1", "lat": 0.0, "lon": 0.0,
         "ts": "2026-01-01T08:20:10Z", "ingested_at": "2026-01-01T08:20:10Z"},
    ]
    report = diagnose_feed(vp_df=_vp_frame(rows), params=DiagnoseParams(gap_threshold_s=300.0))
    gap_finding = next(f for f in report.findings if f.id == "vp_sampling_gaps")
    assert gap_finding.severity is Severity.DEGRADED
    assert gap_finding.count == 1
    assert gap_finding.detail["max_s"] == pytest.approx(1200.0)
    assert "v1" in gap_finding.samples[0]


def test_vp_duplicate_positions():
    rows = [
        {"trip_id": "t1", "vehicle_id": "v1", "lat": 0.0, "lon": 0.0, "ts": "2026-01-01T08:00:00Z"},
        {"trip_id": "t1", "vehicle_id": "v1", "lat": 0.0, "lon": 0.0, "ts": "2026-01-01T08:00:00Z"},
    ]
    report = diagnose_feed(vp_df=_vp_frame(rows))
    finding = next(f for f in report.findings if f.id == "vp_duplicate_positions")
    assert finding.count == 1
    assert finding.severity is Severity.DEGRADED


def test_vp_out_of_order_timestamps():
    rows = [
        {"trip_id": "t1", "vehicle_id": "v1", "lat": 0.0, "lon": 0.0,
         "ts": "2026-01-01T08:00:10Z", "ingested_at": "2026-01-01T08:00:00Z"},
        {"trip_id": "t1", "vehicle_id": "v1", "lat": 0.0, "lon": 0.0,
         "ts": "2026-01-01T08:00:00Z", "ingested_at": "2026-01-01T08:00:01Z"},
    ]
    report = diagnose_feed(vp_df=_vp_frame(rows))
    finding = next(f for f in report.findings if f.id == "vp_out_of_order_timestamps")
    assert finding.count == 1


def test_vp_staleness_flagged_and_unchecked_variants():
    stale_rows = [
        {"trip_id": "t1", "vehicle_id": "v1", "lat": 0.0, "lon": 0.0,
         "ts": "2026-01-01T08:00:00Z", "ingested_at": "2026-01-01T08:10:00Z"},
    ]
    report = diagnose_feed(vp_df=_vp_frame(stale_rows), params=DiagnoseParams(stale_threshold_s=120.0))
    assert any(f.id == "vp_stale_timestamps" for f in report.findings)

    no_ingested = [
        {"trip_id": "t1", "vehicle_id": "v1", "lat": 0.0, "lon": 0.0, "ts": "2026-01-01T08:00:00Z"}
    ]
    df = _vp_frame(no_ingested)
    df["ingested_at"] = pd.NaT
    report2 = diagnose_feed(vp_df=df)
    unchecked = next(f for f in report2.findings if f.id == "vp_staleness_unchecked")
    assert unchecked.severity is Severity.INFO


def test_vp_missing_lat_lon():
    rows = [
        {"trip_id": "t1", "vehicle_id": "v1", "lat": 0.0, "lon": 0.0, "ts": "2026-01-01T08:00:00Z"},
        {"trip_id": "t2", "vehicle_id": "v2", "lat": None, "lon": None, "ts": "2026-01-01T08:00:10Z"},
    ]
    report = diagnose_feed(vp_df=_vp_frame(rows))
    finding = next(f for f in report.findings if f.id == "vp_missing_lat_lon")
    assert finding.count == 1
    assert finding.total == 2


# ---------------------------------------------------------------------------
# Trip-update checks
# ---------------------------------------------------------------------------


def test_tu_missing_times():
    rows = [
        {"trip_id": "t1", "stop_id": "s1", "stop_sequence": 1, "ts": "2026-01-01T08:00:00Z",
         "arrival_time": pd.NaT, "departure_time": pd.NaT},
    ]
    report = diagnose_feed(tu_df=_tu_frame(rows))
    finding = next(f for f in report.findings if f.id == "tu_missing_times")
    assert finding.count == 1


def test_tu_duplicates():
    rows = [
        {"trip_id": "t1", "start_date": "20260101", "stop_id": "s1", "stop_sequence": 1,
         "ts": "2026-01-01T08:00:00Z", "arrival_time": "2026-01-01T08:00:00Z"},
        {"trip_id": "t1", "start_date": "20260101", "stop_id": "s1", "stop_sequence": 1,
         "ts": "2026-01-01T08:00:00Z", "arrival_time": "2026-01-01T08:00:00Z"},
    ]
    report = diagnose_feed(tu_df=_tu_frame(rows))
    finding = next(f for f in report.findings if f.id == "tu_duplicate_predictions")
    assert finding.count == 1


def test_tu_out_of_order_arrivals():
    rows = [
        {"trip_id": "t1", "start_date": "20260101", "stop_id": "s1", "stop_sequence": 1,
         "ts": "2026-01-01T08:00:00Z", "arrival_time": "2026-01-01T08:05:00Z"},
        {"trip_id": "t1", "start_date": "20260101", "stop_id": "s2", "stop_sequence": 2,
         "ts": "2026-01-01T08:00:00Z", "arrival_time": "2026-01-01T08:01:00Z"},
    ]
    report = diagnose_feed(tu_df=_tu_frame(rows))
    finding = next(f for f in report.findings if f.id == "tu_out_of_order_arrivals")
    assert finding.count == 1


# ---------------------------------------------------------------------------
# Static-dependent checks
# ---------------------------------------------------------------------------


def _write_zip(path: Path, members: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return path


STOPS_TXT = "stop_id,stop_lat,stop_lon\ns1,0.0,0.0\ns2,0.0,0.01\ns3,0.0,0.02\n"
TRIPS_TXT = "trip_id,route_id,service_id,shape_id\nt1,r1,wk,sh1\nt2,r1,wk,sh1\n"
STOP_TIMES_TXT = (
    "trip_id,stop_id,stop_sequence,shape_dist_traveled\n"
    "t1,s1,1,0\nt1,s2,2,100\nt1,s3,3,200\n"
    "t2,s1,1,0\nt2,s2,2,100\nt2,s3,3,200\n"
)
ROUTES_TXT = "route_id,route_short_name\nr1,Red\n"


def _static_loader(tmp_path: Path, *, with_shapes: bool, shape_dist: bool = True) -> tuple[GtfsStaticLoader, StaticFeed]:
    members = {
        "stops.txt": STOPS_TXT,
        "trips.txt": TRIPS_TXT,
        "stop_times.txt": STOP_TIMES_TXT,
        "routes.txt": ROUTES_TXT,
    }
    if with_shapes:
        dist_col = "shape_dist_traveled" if shape_dist else None
        header = "shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence"
        rows = ["sh1,0.0,0.0,1", "sh1,0.0,0.01,2", "sh1,0.0,0.02,3"]
        if dist_col:
            header += ",shape_dist_traveled"
            rows = ["sh1,0.0,0.0,1,0", "sh1,0.0,0.01,2,100", "sh1,0.0,0.02,3,200"]
        members["shapes.txt"] = header + "\n" + "\n".join(rows) + "\n"

    zip_path = _write_zip(tmp_path / "gtfs.zip", members)
    tables = ["stops", "trips", "stop_times", "routes"] + (["shapes"] if with_shapes else [])
    feed = StaticFeed(source_id="demo", tables=tables)
    loader = GtfsStaticLoader(zip_path)
    return loader, feed


def test_shapes_missing_is_degraded(tmp_path):
    loader, feed = _static_loader(tmp_path, with_shapes=False)
    report = diagnose_feed(static_loader=loader, static_feed=feed)
    finding = next(f for f in report.findings if f.id == "shapes_missing")
    assert finding.severity is Severity.DEGRADED


def test_shape_dist_traveled_missing(tmp_path):
    zip_path = _write_zip(
        tmp_path / "gtfs.zip",
        {
            "stops.txt": STOPS_TXT,
            "trips.txt": TRIPS_TXT,
            "stop_times.txt": STOP_TIMES_TXT,
            "routes.txt": ROUTES_TXT,
            "shapes.txt": (
                "shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence,shape_dist_traveled\n"
                "sh1,0.0,0.0,1,0\nsh1,0.0,0.01,2,\nsh1,0.0,0.02,3,200\n"
            ),
        },
    )
    feed = StaticFeed(source_id="demo", tables=["stops", "trips", "stop_times", "routes", "shapes"])
    loader = GtfsStaticLoader(zip_path)
    report = diagnose_feed(static_loader=loader, static_feed=feed)
    finding = next(f for f in report.findings if f.id == "shape_dist_traveled_missing")
    assert finding.count == 1
    assert finding.total == 3


def test_trip_route_stop_mismatch(tmp_path):
    loader, feed = _static_loader(tmp_path, with_shapes=True)
    vp_df = _vp_frame(
        [
            {"trip_id": "t1", "route_id": "r1", "vehicle_id": "v1", "lat": 0.0, "lon": 0.0,
             "ts": "2026-01-01T08:00:00Z", "stop_id": "s1"},
            {"trip_id": "ghost", "route_id": "ghost_route", "vehicle_id": "v2", "lat": 0.0, "lon": 0.0,
             "ts": "2026-01-01T08:00:00Z", "stop_id": "ghost_stop"},
        ]
    )
    report = diagnose_feed(vp_df=vp_df, static_loader=loader, static_feed=feed)
    ids = {f.id for f in report.findings}
    assert "trip_ids_unmatched_static" in ids
    assert "route_ids_unmatched_static" in ids
    assert "static_trips_never_observed" in ids  # t2 never observed
    unmatched = next(f for f in report.findings if f.id == "trip_ids_unmatched_static")
    assert "ghost" in unmatched.samples


def test_static_required_tables_missing():
    feed = StaticFeed(source_id="demo", tables=["stops"])  # missing trips/stop_times
    report = diagnose_feed(static_loader=None, static_feed=feed)
    finding = next(f for f in report.findings if f.id == "static_required_tables_missing")
    assert finding.severity is Severity.BLOCKER
    assert "trips" in finding.samples


# ---------------------------------------------------------------------------
# Report shape / reproducibility
# ---------------------------------------------------------------------------


def test_report_is_json_serializable_and_carries_params():
    report = diagnose_feed(params=DiagnoseParams(gap_threshold_s=60.0))
    dumped = report.model_dump(mode="json")
    assert dumped["params"]["gap_threshold_s"] == 60.0
    assert isinstance(dumped["findings"], list)


def test_max_samples_bounds_sample_list(tmp_path):
    rows = [
        {"trip_id": f"t{i}", "route_id": "r1", "vehicle_id": f"v{i}", "lat": 0.0, "lon": 0.0,
         "ts": "2026-01-01T08:00:00Z"}
        for i in range(50)
    ]
    loader, feed = _static_loader(tmp_path, with_shapes=True)
    report = diagnose_feed(
        vp_df=_vp_frame(rows), static_loader=loader, static_feed=feed, params=DiagnoseParams(max_samples=3)
    )
    finding = next(f for f in report.findings if f.id == "trip_ids_unmatched_static")
    assert len(finding.samples) == 3
