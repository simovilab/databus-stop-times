"""test_cli_replay — `stoptimes val replay` (T6 deliverable 2).

Full CliRunner invocation, offline: reads an archived VP parquet dataset
written in the exact Hive layout `stoptimes.record.storage.schema` uses
(this test builds it directly via the existing `test_feeds_replay_archive`
helpers -- val does not import `record`, only its documented column/layout
contract), no network, no wall clock.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from stoptimes.val.cli.main import app
from tests.test_feeds_replay_archive import _base_vp_rows, _write_vp_archive

runner = CliRunner()

_FIXTURES = Path(__file__).parent / "fixtures"
_MINI_GTFS = _FIXTURES / "mini_gtfs.zip"


def test_replay_command_scores_archive_deterministically(tmp_path: Path) -> None:
    vp_dir = _write_vp_archive(tmp_path, _base_vp_rows())
    out = tmp_path / "results.parquet"

    result = runner.invoke(
        app,
        [
            "replay",
            "--source", "mbta",
            "--archive", str(vp_dir),
            "--gtfs", str(_MINI_GTFS),
            "--out", str(out),
            "--seed", "42",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Replaying: source='mbta'" in result.output
    assert "Provenance: etaval" in result.output
    assert out.exists()

    summary_sidecar = Path(f"{out}.summary.json")
    assert summary_sidecar.is_file()
    summary_data = json.loads(summary_sidecar.read_text())
    assert summary_data["provenance"]["seed"] == 42


def test_replay_is_deterministic_across_runs(tmp_path: Path) -> None:
    """Same archive in, same scores out -- run it twice, compare the exported parquet."""
    vp_dir = _write_vp_archive(tmp_path, _base_vp_rows())
    out1 = tmp_path / "run1.parquet"
    out2 = tmp_path / "run2.parquet"

    for out in (out1, out2):
        result = runner.invoke(
            app,
            [
                "replay",
                "--source", "mbta",
                "--archive", str(vp_dir),
                "--gtfs", str(_MINI_GTFS),
                "--out", str(out),
                "--seed", "1",
            ],
        )
        assert result.exit_code == 0, result.output

    import pandas as pd

    df1 = pd.read_parquet(out1).sort_values(list(pd.read_parquet(out1).columns)).reset_index(drop=True)
    df2 = pd.read_parquet(out2).sort_values(list(pd.read_parquet(out2).columns)).reset_index(drop=True)
    pd.testing.assert_frame_equal(df1, df2)


def test_replay_requires_gtfs_or_gtfs_dir(tmp_path: Path) -> None:
    vp_dir = _write_vp_archive(tmp_path, _base_vp_rows())

    result = runner.invoke(
        app,
        ["replay", "--source", "mbta", "--archive", str(vp_dir)],
    )

    assert result.exit_code == 1
    assert "--gtfs or --gtfs-dir is required" in result.output


def test_replay_rejects_unknown_predictor_id(tmp_path: Path) -> None:
    vp_dir = _write_vp_archive(tmp_path, _base_vp_rows())

    result = runner.invoke(
        app,
        [
            "replay",
            "--source", "mbta",
            "--archive", str(vp_dir),
            "--gtfs", str(_MINI_GTFS),
            "--predictor", "not-a-real-predictor",
        ],
    )

    assert result.exit_code == 1
    assert "unknown predictor IDs" in result.output


def test_replay_accepts_stoptimes_prefixed_predictor_id(tmp_path: Path) -> None:
    """A `stoptimes:<ref>` predictor id validates even though it's not a
    literal PREDICTOR_REGISTRY key (resolved dynamically)."""
    vp_dir = _write_vp_archive(tmp_path, _base_vp_rows())

    result = runner.invoke(
        app,
        [
            "replay",
            "--source", "mbta",
            "--archive", str(vp_dir),
            "--gtfs", str(_MINI_GTFS),
            "--predictor", "stoptimes:mbta",
        ],
    )

    # Resolves and runs (an empty registry just means the predictor never
    # scores anything -- that's not a validation error).
    assert result.exit_code == 0, result.output
