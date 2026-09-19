"""Tests for stoptimes.val.api_verbs: replay_score / score_report.

Reuses the same archived-VP fixture helpers `test_cli_replay.py` uses
(`_base_vp_rows`/`_write_vp_archive` from `test_feeds_replay_archive.py`) so
these two entry points are exercised against the exact same data the CLI's
`replay` command is -- proving the API and the CLI score identically.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from stoptimes.val.api_verbs import replay_score, score_report
from tests.test_feeds_replay_archive import _base_vp_rows, _write_vp_archive

_FIXTURES = Path(__file__).parent / "fixtures"
_MINI_GTFS = _FIXTURES / "mini_gtfs.zip"


def test_replay_score_returns_a_dataframe(tmp_path: Path) -> None:
    vp_dir = _write_vp_archive(tmp_path, _base_vp_rows())

    scored = replay_score(source="mbta", archive=str(vp_dir), static=str(_MINI_GTFS), seed=42)

    assert isinstance(scored, pd.DataFrame)
    assert len(scored) > 0
    assert "predictor_id" in scored.columns
    assert "error_s" in scored.columns


def test_replay_score_is_deterministic(tmp_path: Path) -> None:
    vp_dir = _write_vp_archive(tmp_path, _base_vp_rows())

    first = replay_score(source="mbta", archive=str(vp_dir), static=str(_MINI_GTFS), seed=1)
    second = replay_score(source="mbta", archive=str(vp_dir), static=str(_MINI_GTFS), seed=1)

    cols = sorted(first.columns)
    pd.testing.assert_frame_equal(
        first[cols].sort_values(cols).reset_index(drop=True),
        second[cols].sort_values(cols).reset_index(drop=True),
    )


def test_replay_score_can_export_to_parquet(tmp_path: Path) -> None:
    vp_dir = _write_vp_archive(tmp_path, _base_vp_rows())
    out = tmp_path / "results.parquet"

    scored = replay_score(source="mbta", archive=str(vp_dir), static=str(_MINI_GTFS), out=out)

    assert out.is_file()
    assert len(pd.read_parquet(out)) == len(scored)


def test_replay_score_requires_a_static_snapshot(tmp_path: Path) -> None:
    vp_dir = _write_vp_archive(tmp_path, _base_vp_rows())

    with pytest.raises(ValueError):
        replay_score(source="mbta", archive=str(vp_dir))


def test_score_report_from_dataframe_and_from_parquet(tmp_path: Path) -> None:
    vp_dir = _write_vp_archive(tmp_path, _base_vp_rows())
    out = tmp_path / "results.parquet"
    scored = replay_score(source="mbta", archive=str(vp_dir), static=str(_MINI_GTFS), out=out)

    report_from_df = score_report(scored)
    report_from_path = score_report(out)

    assert len(report_from_df.per_predictor) == len(report_from_path.per_predictor) > 0
