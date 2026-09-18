"""Tests for the real `stoptimes record start`/`status` implementation (T5).

Deliberately a separate file from ``test_cli.py`` (shared across tracks,
not owned by T5): that file's ``test_record_start_is_planned`` /
`test_record_help` assume the pre-Phase-2 stub, which `record start`/`status`
still honor when called with no ``--source`` (see docs/notes/p2-t5.md).
Everything here exercises the real implementation added on top of that.

Offline: HTTP is mocked with respx, no real feed is polled, no real S3.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx
import yaml
from stoptimes.cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture
def source_yaml(tmp_path):
    """A minimal, non-agency-specific source config + one mocked JSON VP feed."""
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "acme.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "acme",
                "gtfs_static_url": "http://example.test/gtfs.zip",
                "vehicle_positions_url": "http://example.test/vp.json",
                "trip_updates_url": "http://example.test/tu.json",
                "poll_interval_s": 1,
                "headers": {},
                "arrival_radius_m": 40.0,
                "dwell_s": 0.0,
                "shape_threshold_m": 30.0,
            }
        )
    )
    return sources_dir


_VP_BODY = json.dumps(
    {
        "header": {"timestamp": 1000},
        "entity": [
            {
                "id": "1",
                "vehicle": {
                    "trip": {"trip_id": "t1", "route_id": "r1"},
                    "vehicle": {"id": "v1"},
                    "position": {"latitude": 1.0, "longitude": 2.0},
                    "timestamp": 1000,
                },
            }
        ],
    }
).encode()

_TU_BODY = json.dumps({"header": {"timestamp": 1000}, "entity": []}).encode()


# ── stub back-compat (bare invocations stay "planned") ──────────────────────




def test_record_status_help():
    result = runner.invoke(app, ["record", "status", "--help"])
    assert result.exit_code == 0
    assert "--source" in result.output
    assert "--json" in result.output


# ── record start (real) ──────────────────────────────────────────────────


def test_record_start_unknown_source_errors(tmp_path):
    result = runner.invoke(
        app,
        [
            "record",
            "start",
            "--source",
            "nope",
            "--sources-dir",
            str(tmp_path),
            "--duration",
            "1",
        ],
    )
    assert result.exit_code == 1
    assert "No source config found" in result.output


def test_record_start_duration_exits_cleanly(source_yaml, tmp_path):
    with respx.mock(assert_all_called=False) as mock:
        mock.get("http://example.test/vp.json").mock(
            return_value=httpx.Response(200, content=_VP_BODY)
        )
        mock.get("http://example.test/tu.json").mock(
            return_value=httpx.Response(200, content=_TU_BODY)
        )
        result = runner.invoke(
            app,
            [
                "record",
                "start",
                "--source",
                "acme",
                "--sources-dir",
                str(source_yaml),
                "--out",
                str(tmp_path / "archive"),
                "--spool-dir",
                str(tmp_path / "spool"),
                "--status-dir",
                str(tmp_path / "status"),
                "--duration",
                "1.5",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "stopped (duration_elapsed)" in result.output
    assert (tmp_path / "status" / "acme.json").exists()


def test_record_start_dry_run_writes_nothing(source_yaml, tmp_path):
    with respx.mock(assert_all_called=False) as mock:
        mock.get("http://example.test/vp.json").mock(
            return_value=httpx.Response(200, content=_VP_BODY)
        )
        mock.get("http://example.test/tu.json").mock(
            return_value=httpx.Response(200, content=_TU_BODY)
        )
        result = runner.invoke(
            app,
            [
                "record",
                "start",
                "--source",
                "acme",
                "--sources-dir",
                str(source_yaml),
                "--out",
                str(tmp_path / "archive"),
                "--spool-dir",
                str(tmp_path / "spool"),
                "--duration",
                "1",
                "--dry-run",
            ],
        )

    assert result.exit_code == 0, result.output
    assert not (tmp_path / "archive").exists()


def test_record_start_feeds_filter_only_polls_selected(source_yaml, tmp_path):
    with respx.mock(assert_all_called=False) as mock:
        vp_route = mock.get("http://example.test/vp.json").mock(
            return_value=httpx.Response(200, content=_VP_BODY)
        )
        tu_route = mock.get("http://example.test/tu.json").mock(
            return_value=httpx.Response(200, content=_TU_BODY)
        )
        result = runner.invoke(
            app,
            [
                "record",
                "start",
                "--source",
                "acme",
                "--sources-dir",
                str(source_yaml),
                "--out",
                str(tmp_path / "archive"),
                "--spool-dir",
                str(tmp_path / "spool"),
                "--feeds",
                "vp",
                "--duration",
                "1",
            ],
        )

    assert result.exit_code == 0, result.output
    assert vp_route.call_count >= 1
    assert tu_route.call_count == 0


# ── record status (real) ─────────────────────────────────────────────────


def test_record_status_json_reports_spool_state(source_yaml, tmp_path):
    with respx.mock(assert_all_called=False) as mock:
        mock.get("http://example.test/vp.json").mock(
            return_value=httpx.Response(200, content=_VP_BODY)
        )
        mock.get("http://example.test/tu.json").mock(
            return_value=httpx.Response(200, content=_TU_BODY)
        )
        runner.invoke(
            app,
            [
                "record",
                "start",
                "--source",
                "acme",
                "--sources-dir",
                str(source_yaml),
                "--out",
                str(tmp_path / "archive"),
                "--spool-dir",
                str(tmp_path / "spool"),
                "--status-dir",
                str(tmp_path / "status"),
                "--duration",
                "1",
            ],
        )

    result = runner.invoke(
        app,
        [
            "record",
            "status",
            "--source",
            "acme",
            "--sources-dir",
            str(source_yaml),
            "--spool-dir",
            str(tmp_path / "spool"),
            "--status-dir",
            str(tmp_path / "status"),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["source"] == "acme"
    assert report["last_status"]["state"] == "stopped"
