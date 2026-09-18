"""`stoptimes serve` — CLI wiring for the T4 serving pipeline.

Owned by track T4 (docs/PLAN.md §10): the shared `packages/cli/tests/test_cli.py`
still asserts the pre-T4 "planned" phrasing for a bare `stoptimes serve` (no
`--source`) stays true even after this command does real work, so those two
assertions aren't duplicated destructively here -- this file covers the real
behavior once `--source` is given.

No network: a custom `--pipeline` file overrides `ingest`/`predict` with
synthetic data, exercising the override mechanism the whole product is built
around while keeping the test offline.
"""

from __future__ import annotations

import json
import textwrap

from stoptimes.cli.main import app
from typer.testing import CliRunner

runner = CliRunner()

_CUSTOM_PIPELINE = textwrap.dedent(
    """
    import pandas as pd
    from stoptimes.core.contracts.data import Observation, ObservationKind
    from stoptimes.core.contracts.prediction import StopTimePrediction
    from stoptimes.serve.pipeline import build_default_pipeline

    pipeline = build_default_pipeline()

    @pipeline.stage("ingest")
    def _ingest(_payload, ctx):
        return [
            Observation(
                kind=ObservationKind.TRIP_UPDATE,
                source_id=ctx.source.id,
                trip_id="t1",
                route_id="r1",
                direction_id=0,
                start_date="20260917",
                stop_id="s1",
                stop_sequence=1,
                ts=pd.Timestamp(ctx.now),
            )
        ]

    @pipeline.stage("predict")
    def _predict(features, ctx):
        return [
            StopTimePrediction(
                trip_id=row["trip_id"],
                start_date=row["start_date"],
                stop_id=row["stop_id"],
                stop_sequence=int(row["stop_sequence"]),
                horizon_s=30.0,
                predicted_arrival_time=ctx.now,
                model_id="test-pipeline",
                generated_at=ctx.now,
            )
            for _, row in features.iterrows()
        ]
    """
)


def test_serve_help_is_fast_and_lists_options():
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    for opt in ("--source", "--registry", "--model", "--artifact", "--pipeline", "--sink", "--format", "--once", "--interval", "--json"):
        assert opt in result.output


def test_serve_requires_source():
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 2
    assert "--source" in result.output


def test_serve_unknown_source_exits_1():
    result = runner.invoke(app, ["serve", "--source", "does-not-exist-agency"])
    assert result.exit_code == 1
    assert "does-not-exist-agency" in result.output


def test_serve_rejects_bad_format():
    result = runner.invoke(app, ["serve", "--source", "mbta", "--format", "xml"])
    assert result.exit_code == 2


def test_serve_rejects_once_and_interval_together():
    result = runner.invoke(app, ["serve", "--source", "mbta", "--once", "--interval", "5"])
    assert result.exit_code == 2


def test_serve_once_with_custom_pipeline_and_file_sink(tmp_path):
    pipeline_file = tmp_path / "custom_pipeline.py"
    pipeline_file.write_text(_CUSTOM_PIPELINE)
    out_file = tmp_path / "tripupdates.json"

    result = runner.invoke(
        app,
        [
            "serve",
            "--source",
            "mbta",
            "--pipeline",
            f"{pipeline_file}:pipeline",
            "--sink",
            f"file:{out_file}",
            "--once",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary["source"] == "mbta"
    assert summary["trip_update_count"] == 1
    assert summary["published"] is True
    assert summary["errors"] == []

    feed = json.loads(out_file.read_bytes())
    assert feed["header"]["gtfs_realtime_version"] == "2.0"
    entity = feed["entity"][0]
    assert entity["trip_update"]["trip"]["trip_id"] == "t1"
    assert entity["trip_update"]["stop_time_update"][0]["stop_id"] == "s1"


def test_serve_once_human_readable_output(tmp_path):
    pipeline_file = tmp_path / "custom_pipeline.py"
    pipeline_file.write_text(_CUSTOM_PIPELINE)
    out_file = tmp_path / "tripupdates.pb"

    result = runner.invoke(
        app,
        [
            "serve",
            "--source",
            "mbta",
            "--pipeline",
            f"{pipeline_file}:pipeline",
            "--sink",
            f"file:{out_file}",
            "--format",
            "pb",
            "--once",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "1 trip update(s)" in result.output
    assert out_file.exists()
    assert out_file.read_bytes() != b""
