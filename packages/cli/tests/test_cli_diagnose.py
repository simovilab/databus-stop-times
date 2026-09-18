"""`stoptimes diagnose` — the CLI/reader boundary (track U3).

Most of the behaviour (checks, severities, findings) is unit-tested against
`stoptimes.core.diagnose.diagnose_feed` directly in
`packages/core/tests/test_diagnose.py`; this file covers what only exists at
the CLI layer: locating/reading observations, `--from`/`--to` windowing an
archive_parquet read, and `--json` output shape.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from stoptimes.cli.main import app
from typer.testing import CliRunner

pytest.importorskip("pandas")

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("STOPTIMES_HOME", str(tmp_path / "home"))


def _write_archive(root: Path) -> Path:
    """Three days of vehicle positions, one Hive-partitioned parquet file
    per day, in the layout `stoptimes.record` writes."""
    import pandas as pd

    for day in (17, 18, 19):
        rows = [
            {
                "source_id": "demo",
                "kind": "vehicle_position",
                "trip_id": f"t{day}",
                "route_id": "r1",
                "vehicle_id": f"v{day}",
                "ts": pd.Timestamp(f"2026-09-{day:02d}T08:00:00Z"),
                "lat": 0.0,
                "lon": 0.0,
                "bearing": 90.0,
            }
        ]
        partition = root / "year=2026" / "month=09" / f"day={day:02d}" / "route_id=r1"
        partition.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(partition / "part-0.parquet")
    return root


def test_diagnose_with_no_input_exits_zero_and_reports_blockers():
    result = runner.invoke(app, ["diagnose", "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["verdict_severity"] == "blocker"
    ids = {f["id"] for f in report["findings"]}
    assert {"vp_missing", "static_missing"} <= ids


def test_from_to_window_prunes_archive_files_before_reading(tmp_path):
    """Rows outside [--from, --to] must not reach the report at all — the
    window is applied by pruning the partitioned file list, not by
    filtering rows after a full read."""
    archive = _write_archive(tmp_path / "archive")

    unbounded = runner.invoke(app, ["diagnose", "--vehicle-positions", str(archive), "--json"])
    assert unbounded.exit_code == 0, unbounded.output
    assert json.loads(unbounded.stdout)["vp_rows"] == 3

    bounded = runner.invoke(
        app,
        [
            "diagnose",
            "--vehicle-positions",
            str(archive),
            "--from",
            "2026-09-18",
            "--to",
            "2026-09-18",
            "--json",
        ],
    )
    assert bounded.exit_code == 0, bounded.output
    report = json.loads(bounded.stdout)
    assert report["vp_rows"] == 1
    assert report["window"] == {"from": "2026-09-18", "to": "2026-09-18"}


def test_unbounded_archive_read_warns_about_memory(tmp_path):
    archive = _write_archive(tmp_path / "archive")
    result = runner.invoke(app, ["diagnose", "--vehicle-positions", str(archive)])
    assert result.exit_code == 0, result.output
    assert "no --from/--to bound" in result.output


def test_bounded_archive_read_does_not_warn(tmp_path):
    archive = _write_archive(tmp_path / "archive")
    result = runner.invoke(
        app,
        ["diagnose", "--vehicle-positions", str(archive), "--from", "2026-09-18", "--to", "2026-09-18"],
    )
    assert result.exit_code == 0, result.output
    assert "no --from/--to bound" not in result.output


def test_unknown_location_is_a_usage_error(tmp_path):
    result = runner.invoke(
        app, ["diagnose", "--vehicle-positions", str(tmp_path / "does-not-exist.csv")]
    )
    assert result.exit_code == 1
