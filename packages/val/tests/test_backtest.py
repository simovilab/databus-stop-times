"""Tests for etaval/backtest.py — A6/M6 walk-forward backtest.

Follows the synthetic Hive-partitioned Parquet fixture pattern used by
tests/test_feeds_replay_archive.py (never depends on a real archive on disk),
extended to 5 ticks matching all 4 mini_gtfs.zip stops so the engine actually
detects arrivals and scores predictions in each window (mirroring
tests/test_engine_replay.py's snapshot scenario).
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from etaval.backtest import (
    SPLIT_NOTE,
    WalkForwardResult,
    WindowResult,
    split_windows,
    walk_forward,
)
from etaval.cli.main import app
from etaval.feeds.source_config import load_named_source
from etaval.stats.bootstrap import CIResult

_FIXTURES = Path(__file__).parent / "fixtures"
_MINI_GTFS = _FIXTURES / "mini_gtfs.zip"

runner = CliRunner()

# Base tick timestamps, chosen well clear of any window-edge collision below.
T0 = 1704153600.0  # 2024-01-02T00:00:00Z
TICKS = [T0, T0 + 100.0, T0 + 200.0, T0 + 300.0, T0 + 400.0]

# Vehicle positions matching mini_gtfs.zip's stops.txt exactly (s1..s4), same
# progression as tests/test_engine_replay.py's 5-tick scenario.
_POSITIONS = [
    (42.3601, -71.0589, 1, "s1"),
    (42.3610, -71.0580, 2, "s2"),
    (42.3620, -71.0570, 3, "s3"),
    (42.3630, -71.0560, 4, "s4"),
    (42.3630, -71.0560, 4, "s4"),  # past s4, same position
]


def _write_hive_parquet(base_dir: Path, df: pd.DataFrame, partition_cols: list[str]) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(base_dir), engine="pyarrow", partition_cols=partition_cols, index=False)
    return base_dir


def _vp_rows() -> list[dict]:
    rows = []
    for t, (lat, lon, seq, stop_id) in zip(TICKS, _POSITIONS, strict=True):
        rows.append(
            {
                "vehicle_id": "v1",
                "ts": t,
                "lat": lat,
                "lon": lon,
                "speed": 3.0,
                "trip_id": "trip1",
                "route_id": "r1",
                "current_stop_sequence": seq,
                "current_status": "IN_TRANSIT_TO",
                "stop_id": stop_id,
            }
        )
    return rows


def _tu_rows() -> list[dict]:
    """TU predictions mirroring test_engine_replay: predict each remaining stop."""
    remaining_stops = [
        [("s2", 2, TICKS[1]), ("s3", 3, TICKS[2]), ("s4", 4, TICKS[3])],
        [("s3", 3, TICKS[2]), ("s4", 4, TICKS[3])],
        [("s4", 4, TICKS[3])],
        [],
        [],
    ]
    rows = []
    for t, preds in zip(TICKS, remaining_stops, strict=True):
        for stop_id, seq, arr_t in preds:
            rows.append(
                {
                    "ts": t,
                    "trip_id": "trip1",
                    "route_id": "r1",
                    "stop_sequence": seq,
                    "stop_id": stop_id,
                    "arrival_delay": 0,
                    "arrival_time": arr_t,
                    "departure_time": arr_t + 30,
                }
            )
    return rows


def _with_partitions(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        d = dict(r)
        date = dt.datetime.fromtimestamp(d["ts"], tz=dt.UTC).date()
        d.setdefault("year", date.year)
        d.setdefault("month", date.month)
        d.setdefault("day", date.day)
        d.setdefault("feed_name", "test-feed")
        d.setdefault("bearing", 0.0)
        d.setdefault("ingested_at", d["ts"])
        out.append(d)
    return pd.DataFrame(out)


def _write_archive(tmp_path: Path) -> tuple[Path, Path]:
    vp_df = _with_partitions(_vp_rows())
    tu_df = _with_partitions(_tu_rows())
    vp_dir = _write_hive_parquet(
        tmp_path / "vp", vp_df, partition_cols=["year", "month", "day", "route_id"]
    )
    tu_dir = _write_hive_parquet(
        tmp_path / "tu", tu_df, partition_cols=["year", "month", "day", "route_id"]
    )
    return vp_dir, tu_dir


# ── split_windows ──────────────────────────────────────────────────────────────


class TestSplitWindows:
    def test_equal_count_contiguous_and_covers(self) -> None:
        windows = split_windows(0.0, 100.0, n_windows=4)
        assert len(windows) == 4
        assert windows[0][0] == 0.0
        assert windows[-1][1] == 100.0
        # contiguous: each window's end == next window's start
        for i in range(len(windows) - 1):
            assert windows[i][1] == windows[i + 1][0]
        # equal widths
        widths = [b - a for a, b in windows]
        assert all(w == pytest.approx(25.0) for w in widths)

    def test_equal_count_single_window(self) -> None:
        windows = split_windows(10.0, 20.0, n_windows=1)
        assert windows == [(10.0, 20.0)]

    def test_fixed_width_contiguous_and_covers(self) -> None:
        windows = split_windows(0.0, 100.0, width=30.0)
        assert windows[0][0] == 0.0
        assert windows[-1][1] == 100.0
        for i in range(len(windows) - 1):
            assert windows[i][1] == windows[i + 1][0]
        # 100/30 -> 4 windows, last one truncated to 10s wide
        assert len(windows) == 4
        assert windows[-1][1] - windows[-1][0] == pytest.approx(10.0)

    def test_fixed_width_exact_division_no_empty_trailing_window(self) -> None:
        windows = split_windows(0.0, 90.0, width=30.0)
        assert len(windows) == 3
        assert windows[-1] == (60.0, 90.0)

    def test_start_must_be_less_than_end(self) -> None:
        with pytest.raises(ValueError, match="must be <"):
            split_windows(10.0, 10.0, n_windows=2)
        with pytest.raises(ValueError, match="must be <"):
            split_windows(10.0, 5.0, n_windows=2)

    def test_requires_exactly_one_of_n_windows_or_width(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            split_windows(0.0, 10.0)
        with pytest.raises(ValueError, match="exactly one"):
            split_windows(0.0, 10.0, n_windows=2, width=5.0)

    def test_n_windows_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="n_windows"):
            split_windows(0.0, 10.0, n_windows=0)

    def test_width_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="width"):
            split_windows(0.0, 10.0, width=0.0)
        with pytest.raises(ValueError, match="width"):
            split_windows(0.0, 10.0, width=-5.0)


# ── walk_forward end-to-end ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_walk_forward_end_to_end(tmp_path: Path) -> None:
    vp_dir, tu_dir = _write_archive(tmp_path)
    cfg = load_named_source("mbta")

    # Window edges deliberately avoid landing exactly on a tick timestamp.
    result = await walk_forward(
        cfg,
        vp_path=vp_dir,
        tu_path=tu_dir,
        start=T0 - 100.0,
        end=T0 + 450.0,
        predictor_ids=["gtfsrt", "baseline:constant_speed"],
        n_windows=2,
        seed=7,
        static_gtfs_zip=_MINI_GTFS,
        n_boot=200,
    )

    assert isinstance(result, WalkForwardResult)
    assert len(result.windows) == 2
    assert result.seed == 7
    assert result.split_note == SPLIT_NOTE
    assert "70/10/20" in result.split_note
    assert result.provenance is not None
    assert result.provenance.seed == 7
    assert result.provenance.predictor_ids == ["gtfsrt", "baseline:constant_speed"]

    total_results = 0
    for w in result.windows:
        assert isinstance(w, WindowResult)
        assert w.window_end > w.window_start
        assert 0.0 <= w.eval_fraction <= 1.0
        assert set(w.mae_by_predictor) == {"gtfsrt", "baseline:constant_speed"}
        for ci_res in w.mae_by_predictor.values():
            assert isinstance(ci_res, CIResult)
        total_results += w.n_results

    # At least one window produced scored results (both predictors + detectors
    # fire on the s1->s4 progression, as in test_engine_replay.py).
    assert total_results > 0

    aggregate = result.aggregate()
    assert set(aggregate) <= {"gtfsrt", "baseline:constant_speed"}
    for stats in aggregate.values():
        assert "mean_mae" in stats
        assert "std_mae" in stats
        assert "n_windows" in stats
        assert stats["mean_mae"] >= 0.0
        assert stats["std_mae"] >= 0.0

    # ── to_dict / write_json round-trip ───────────────────────────────────────
    payload = result.to_dict()
    assert payload["seed"] == 7
    assert payload["split_note"] == SPLIT_NOTE
    assert payload["provenance"]["seed"] == 7
    assert len(payload["windows"]) == 2
    assert "aggregate" in payload

    out_path = tmp_path / "wf.json"
    written = result.write_json(out_path)
    assert written == out_path
    on_disk = json.loads(out_path.read_text())
    assert on_disk == payload


@pytest.mark.asyncio
async def test_walk_forward_default_predictor_ids(tmp_path: Path) -> None:
    vp_dir, tu_dir = _write_archive(tmp_path)
    cfg = load_named_source("mbta")

    result = await walk_forward(
        cfg,
        vp_path=vp_dir,
        tu_path=tu_dir,
        start=T0 - 10.0,
        end=T0 + 410.0,
        n_windows=1,
        seed=0,
        static_gtfs_zip=_MINI_GTFS,
    )
    assert len(result.windows) == 1
    from etaval.engine.assembly import DEFAULT_PREDICTOR_IDS

    assert set(result.windows[0].mae_by_predictor) == set(DEFAULT_PREDICTOR_IDS)


@pytest.mark.asyncio
async def test_walk_forward_requires_static_gtfs(tmp_path: Path) -> None:
    vp_dir, tu_dir = _write_archive(tmp_path)
    cfg = load_named_source("mbta")

    with pytest.raises(ValueError, match="static_gtfs"):
        await walk_forward(
            cfg,
            vp_path=vp_dir,
            tu_path=tu_dir,
            start=T0 - 10.0,
            end=T0 + 410.0,
            n_windows=1,
            seed=0,
        )


# ── CLI ───────────────────────────────────────────────────────────────────────


def test_cli_walk_forward_end_to_end(tmp_path: Path) -> None:
    vp_dir, tu_dir = _write_archive(tmp_path)
    out_path = tmp_path / "wf_cli.json"

    result = runner.invoke(
        app,
        [
            "walk-forward",
            str(out_path),
            "--source",
            "mbta",
            "--replay-vp",
            str(vp_dir),
            "--replay-tu",
            str(tu_dir),
            "--replay-gtfs",
            str(_MINI_GTFS),
            "--replay-start",
            str(T0 - 100.0),
            "--replay-end",
            str(T0 + 450.0),
            "--windows",
            "2",
            "--seed",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["seed"] == 3
    assert len(payload["windows"]) == 2
    assert "split_note" in payload


def test_cli_walk_forward_requires_gtfs_source(tmp_path: Path) -> None:
    vp_dir, tu_dir = _write_archive(tmp_path)
    out_path = tmp_path / "wf_cli.json"

    result = runner.invoke(
        app,
        [
            "walk-forward",
            str(out_path),
            "--source",
            "mbta",
            "--replay-vp",
            str(vp_dir),
            "--replay-start",
            str(T0),
            "--replay-end",
            str(T0 + 400.0),
            "--windows",
            "2",
        ],
    )
    assert result.exit_code == 1
    assert "--replay-gtfs" in result.output


def test_cli_walk_forward_requires_windows_xor_width(tmp_path: Path) -> None:
    vp_dir, _tu_dir = _write_archive(tmp_path)
    out_path = tmp_path / "wf_cli.json"

    result = runner.invoke(
        app,
        [
            "walk-forward",
            str(out_path),
            "--source",
            "mbta",
            "--replay-vp",
            str(vp_dir),
            "--replay-gtfs",
            str(_MINI_GTFS),
            "--replay-start",
            str(T0),
            "--replay-end",
            str(T0 + 400.0),
        ],
    )
    assert result.exit_code == 1
    assert "--windows" in result.output or "--width" in result.output
