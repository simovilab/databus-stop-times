"""Tests for weekly static GTFS snapshots (rt_pipeline.storage.static_gtfs).

`requests.get`, `subprocess.run`, and credential resolution are all
monkeypatched -- no network or `mc` dependency, same spirit as the other
storage tests running against a local tmpdir instead of real S3.
"""

from __future__ import annotations

import datetime as dt
import io
import zipfile

import pytest
import requests

from rt_pipeline.storage import static_gtfs as sg


def _zip_bytes(members: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return buf.getvalue()


_VALID_GTFS = {
    "stops.txt": "stop_id,stop_name\n1,Main St\n",
    "routes.txt": "route_id,route_short_name\nR1,1\n",
}


class _FakeResponse:
    def __init__(self, content: bytes, *, status: int = 200):
        self.content = content
        self._status = status

    def raise_for_status(self) -> None:
        if self._status >= 400:
            raise requests.HTTPError(f"HTTP {self._status}")


def test_fetch_returns_bytes_for_valid_gtfs(monkeypatch):
    body = _zip_bytes(_VALID_GTFS)
    monkeypatch.setattr(sg.requests, "get", lambda url, timeout: _FakeResponse(body))

    result = sg.fetch("https://example.org/gtfs.zip")

    assert result == body


def test_fetch_raises_on_non_zip(monkeypatch):
    monkeypatch.setattr(
        sg.requests, "get", lambda url, timeout: _FakeResponse(b"<html>not a zip</html>")
    )

    with pytest.raises(sg.StaticGtfsError, match="valid zip"):
        sg.fetch("https://example.org/gtfs.zip")


def test_fetch_raises_on_missing_required_members(monkeypatch):
    body = _zip_bytes({"stops.txt": "stop_id\n1\n"})  # routes.txt missing
    monkeypatch.setattr(sg.requests, "get", lambda url, timeout: _FakeResponse(body))

    with pytest.raises(sg.StaticGtfsError, match="missing required GTFS files"):
        sg.fetch("https://example.org/gtfs.zip")


def test_fetch_raises_on_http_error(monkeypatch):
    monkeypatch.setattr(
        sg.requests, "get", lambda url, timeout: _FakeResponse(b"", status=404)
    )

    with pytest.raises(requests.HTTPError):
        sg.fetch("https://example.org/gtfs.zip")


class _FakeCompletedProcess:
    def __init__(self, returncode: int, stderr: bytes = b""):
        self.returncode = returncode
        self.stderr = stderr


def test_put_snapshot_builds_dated_key_and_pipes_content(monkeypatch):
    calls = {}

    def fake_run(cmd, input, capture_output):
        calls["cmd"] = cmd
        calls["input"] = input
        return _FakeCompletedProcess(0)

    monkeypatch.setattr(sg.subprocess, "run", fake_run)
    monkeypatch.setattr(
        "rt_pipeline.compaction.credentials.load_credentials", lambda alias: None
    )

    key = sg.put_snapshot(b"zip-bytes", "mbta", dt.date(2026, 8, 17))

    assert key == "feeds/mbta/gtfs_static/2026-08-17.zip"
    assert calls["cmd"] == ["mc", "pipe", "simovilab/transit/feeds/mbta/gtfs_static/2026-08-17.zip"]
    assert calls["input"] == b"zip-bytes"


def test_put_snapshot_raises_on_mc_failure(monkeypatch):
    monkeypatch.setattr(
        sg.subprocess,
        "run",
        lambda cmd, input, capture_output: _FakeCompletedProcess(1, b"access denied"),
    )
    monkeypatch.setattr(
        "rt_pipeline.compaction.credentials.load_credentials", lambda alias: None
    )

    with pytest.raises(sg.StaticGtfsError, match="access denied"):
        sg.put_snapshot(b"zip-bytes", "bucr", dt.date(2026, 8, 17))


class _FakeLsCompletedProcess:
    def __init__(self, returncode: int, stdout: bytes = b"", stderr: bytes = b""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


_LS_OUTPUT = (
    b"[2026-08-14 12:36:26 CST]  18MiB STANDARD 2026-08-14.zip\n"
    b"[2026-08-16 22:00:03 CST]  18MiB STANDARD 2026-08-17.zip\n"
    b"[2026-08-23 22:00:04 CST]  18MiB STANDARD 2026-08-24.zip\n"
)


def test_list_snapshots_parses_and_sorts_dates(monkeypatch):
    calls = {}

    def fake_run(cmd, capture_output):
        calls["cmd"] = cmd
        return _FakeLsCompletedProcess(0, stdout=_LS_OUTPUT)

    monkeypatch.setattr(sg.subprocess, "run", fake_run)
    monkeypatch.setattr(
        "rt_pipeline.compaction.credentials.load_credentials", lambda alias: None
    )

    result = sg.list_snapshots("mbta")

    assert calls["cmd"] == ["mc", "ls", "simovilab/transit/feeds/mbta/gtfs_static/"]
    assert result == [
        dt.date(2026, 8, 14),
        dt.date(2026, 8, 17),
        dt.date(2026, 8, 24),
    ]


def test_list_snapshots_returns_empty_when_prefix_missing(monkeypatch):
    monkeypatch.setattr(
        sg.subprocess,
        "run",
        lambda cmd, capture_output: _FakeLsCompletedProcess(
            1, stderr=b"mc: <ERROR> Unable to list folder. Object does not exist"
        ),
    )
    monkeypatch.setattr(
        "rt_pipeline.compaction.credentials.load_credentials", lambda alias: None
    )

    assert sg.list_snapshots("bucr") == []


def test_list_snapshots_returns_empty_for_no_output(monkeypatch):
    monkeypatch.setattr(
        sg.subprocess,
        "run",
        lambda cmd, capture_output: _FakeLsCompletedProcess(0, stdout=b""),
    )
    monkeypatch.setattr(
        "rt_pipeline.compaction.credentials.load_credentials", lambda alias: None
    )

    assert sg.list_snapshots("bucr") == []


def test_list_snapshots_raises_on_other_mc_failure(monkeypatch):
    monkeypatch.setattr(
        sg.subprocess,
        "run",
        lambda cmd, capture_output: _FakeLsCompletedProcess(1, stderr=b"access denied"),
    )
    monkeypatch.setattr(
        "rt_pipeline.compaction.credentials.load_credentials", lambda alias: None
    )

    with pytest.raises(sg.StaticGtfsError, match="access denied"):
        sg.list_snapshots("mbta")


def test_get_snapshot_builds_key_and_returns_bytes(monkeypatch):
    calls = {}

    def fake_run(cmd, capture_output):
        calls["cmd"] = cmd
        return _FakeLsCompletedProcess(0, stdout=b"zip-bytes-here")

    monkeypatch.setattr(sg.subprocess, "run", fake_run)
    monkeypatch.setattr(
        "rt_pipeline.compaction.credentials.load_credentials", lambda alias: None
    )

    result = sg.get_snapshot("mbta", dt.date(2026, 8, 17))

    assert calls["cmd"] == [
        "mc",
        "cat",
        "simovilab/transit/feeds/mbta/gtfs_static/2026-08-17.zip",
    ]
    assert result == b"zip-bytes-here"


def test_get_snapshot_raises_on_missing_date(monkeypatch):
    monkeypatch.setattr(
        sg.subprocess,
        "run",
        lambda cmd, capture_output: _FakeLsCompletedProcess(
            1, stderr=b"mc: <ERROR> Unable to stat: Object does not exist"
        ),
    )
    monkeypatch.setattr(
        "rt_pipeline.compaction.credentials.load_credentials", lambda alias: None
    )

    with pytest.raises(sg.StaticGtfsError, match="Object does not exist"):
        sg.get_snapshot("mbta", dt.date(2099, 1, 1))


def test_latest_snapshot_on_or_before_returns_exact_match(monkeypatch):
    monkeypatch.setattr(
        sg, "list_snapshots", lambda agency, **kw: [
            dt.date(2026, 8, 14), dt.date(2026, 8, 17), dt.date(2026, 8, 24)
        ]
    )

    assert sg.latest_snapshot_on_or_before(
        "mbta", dt.date(2026, 8, 17)
    ) == dt.date(2026, 8, 17)


def test_latest_snapshot_on_or_before_returns_previous_when_between(monkeypatch):
    monkeypatch.setattr(
        sg, "list_snapshots", lambda agency, **kw: [
            dt.date(2026, 8, 14), dt.date(2026, 8, 17), dt.date(2026, 8, 24)
        ]
    )

    assert sg.latest_snapshot_on_or_before(
        "mbta", dt.date(2026, 8, 20)
    ) == dt.date(2026, 8, 17)


def test_latest_snapshot_on_or_before_returns_none_when_target_before_first(
    monkeypatch,
):
    monkeypatch.setattr(
        sg, "list_snapshots", lambda agency, **kw: [
            dt.date(2026, 8, 14), dt.date(2026, 8, 17), dt.date(2026, 8, 24)
        ]
    )

    assert (
        sg.latest_snapshot_on_or_before("mbta", dt.date(2026, 8, 1)) is None
    )


def test_latest_snapshot_on_or_before_returns_none_when_no_snapshots(monkeypatch):
    monkeypatch.setattr(sg, "list_snapshots", lambda agency, **kw: [])

    assert sg.latest_snapshot_on_or_before("mbta", dt.date(2026, 8, 20)) is None
