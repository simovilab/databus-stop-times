"""Tests for `stoptimes data` (T1's CLI command file): add/list/inspect/remove.

Separate file from `test_cli.py` (shared, not owned by this track) per
docs/PLAN.md §10 — `test_cli.py::test_data_add_is_planned` /
`test_data_inspect_is_planned` now fail because `data add`/`data inspect`
are real commands, not `planned()` stubs; that's an expected, orchestrator-
resolved consequence of this track's deliverable (see docs/notes/p2-t1.md),
not something fixed here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from stoptimes.cli.main import app
from typer.testing import CliRunner

runner = CliRunner()

FIXTURES = Path(__file__).resolve().parents[3] / "packages" / "core" / "tests" / "fixtures"
VP_JSON = FIXTURES / "gtfsrt_vp.json"
AVL_CSV = FIXTURES / "avl_sample.csv"


@pytest.fixture
def stoptimes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("STOPTIMES_HOME", str(tmp_path))
    from stoptimes.core.config import reset_config

    reset_config()
    yield tmp_path
    reset_config()


def test_data_add_gtfsrt_json_registers_a_dataset(stoptimes_home):
    result = runner.invoke(
        app,
        ["data", "add", "mbta_vp_sample", str(VP_JSON), "--source", "mbta", "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["name"] == "mbta_vp_sample"
    assert payload["row_count"] == 2
    assert payload["kind"] == "observations"


def test_data_add_then_list_shows_it(stoptimes_home):
    runner.invoke(app, ["data", "add", "mbta_vp_sample", str(VP_JSON), "--source", "mbta"])
    result = runner.invoke(app, ["data", "list", "--json"])
    assert result.exit_code == 0, result.output
    entries = json.loads(result.output)
    assert any(e["name"] == "mbta_vp_sample" for e in entries)


def test_data_add_duplicate_without_overwrite_fails(stoptimes_home):
    runner.invoke(app, ["data", "add", "mbta_vp_sample", str(VP_JSON), "--source", "mbta"])
    result = runner.invoke(app, ["data", "add", "mbta_vp_sample", str(VP_JSON), "--source", "mbta"])
    assert result.exit_code == 1
    assert "already exists" in result.output


def test_data_add_avl_csv_requires_mapping(stoptimes_home):
    result = runner.invoke(
        app,
        ["data", "add", "bucr_avl", str(AVL_CSV), "--source", "bucr", "--format", "avl_csv"],
    )
    assert result.exit_code == 1
    assert "mapping" in result.output.lower()


def test_data_add_avl_csv_with_mapping(stoptimes_home, tmp_path):
    mapping_file = tmp_path / "mapping.yaml"
    mapping_file.write_text(
        "timestamp: cr_datetime\nlat: latitude\nlon: longitude\nvehicle: plate_number\n"
        "source_timezone: America/Costa_Rica\n"
    )
    result = runner.invoke(
        app,
        [
            "data",
            "add",
            "bucr_avl",
            str(AVL_CSV),
            "--source",
            "bucr",
            "--format",
            "avl_csv",
            "--mapping",
            str(mapping_file),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["row_count"] == 3


def test_data_inspect_reports_coverage(stoptimes_home):
    runner.invoke(app, ["data", "add", "mbta_vp_sample", str(VP_JSON), "--source", "mbta"])
    result = runner.invoke(app, ["data", "inspect", "mbta_vp_sample", "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["rows"] == 2
    assert report["trips_seen"] == 2


def test_data_inspect_missing_dataset_fails(stoptimes_home):
    result = runner.invoke(app, ["data", "inspect", "nope", "--json"])
    assert result.exit_code == 1


def test_data_remove_round_trips(stoptimes_home):
    runner.invoke(app, ["data", "add", "mbta_vp_sample", str(VP_JSON), "--source", "mbta"])
    result = runner.invoke(app, ["data", "remove", "mbta_vp_sample", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["removed"] is True
    result = runner.invoke(app, ["data", "remove", "mbta_vp_sample", "--json"])
    assert result.exit_code == 1


def test_data_add_static_dataset(stoptimes_home, tmp_path):
    import zipfile

    zip_path = tmp_path / "gtfs.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("stops.txt", "stop_id,stop_lat,stop_lon\ns1,1.0,2.0\n")
        zf.writestr("trips.txt", "trip_id,route_id,service_id\nt1,r1,wk\n")
        zf.writestr("stop_times.txt", "trip_id,stop_id,stop_sequence\nt1,s1,1\n")

    result = runner.invoke(
        app,
        [
            "data",
            "add",
            "mbta_static",
            str(zip_path),
            "--source",
            "mbta",
            "--kind",
            "static",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["kind"] == "static"
    assert "stops" in payload["extra"]["tables"]



def test_data_help_still_lists_everything(stoptimes_home):
    result = runner.invoke(app, ["data", "--help"])
    assert result.exit_code == 0
    for name in ("add", "list", "inspect", "remove", "arrivals"):
        assert name in result.output
