"""Pipeline stage composition, overriding, and per-trip error containment.

Uses a synthetic `Source` + injected `Observation`s (no network) and a fake
model, per docs/PLAN.md's "synthetic feeds and a fake model everywhere, no
network" rule for T4.
"""

from __future__ import annotations

import textwrap
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from stoptimes.core.contracts.data import Observation, ObservationKind
from stoptimes.core.contracts.pipeline import STAGE_ORDER, PipelineContext
from stoptimes.core.contracts.source import Source
from stoptimes.serve.pipeline import build_default_pipeline, load_pipeline


class FakeModel:
    family = "fake"
    required_features = ["stop_sequence"]

    def __init__(self, *, fail_for_stop_sequence: int | None = None):
        self._fail_for = fail_for_stop_sequence

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        row = X.iloc[0]
        if self._fail_for is not None and int(row["stop_sequence"]) == self._fail_for:
            raise ValueError("synthetic model failure")
        return np.array([60.0 + 10.0 * float(row["stop_sequence"])])


def _source() -> Source:
    return Source(id="testagency", name="Test Agency", gtfs_static_url="/dev/null")


def _tu_observation(trip_id: str, stop_id: str, stop_sequence: int, start_date: str = "20260917") -> Observation:
    return Observation(
        kind=ObservationKind.TRIP_UPDATE,
        source_id="testagency",
        trip_id=trip_id,
        route_id="r1",
        direction_id=0,
        start_date=start_date,
        stop_id=stop_id,
        stop_sequence=stop_sequence,
        ts=pd.Timestamp("2026-09-17T12:00:00Z"),
    )


def _ctx(model, observations) -> PipelineContext:
    return PipelineContext(
        source=_source(),
        static_feed=None,
        model=model,
        now=datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc),
        config={"observations": observations},
    )


def test_stage_order_matches_contract():
    assert STAGE_ORDER == ("ingest", "match", "features", "predict", "postprocess", "publish")


def test_default_pipeline_runs_end_to_end_and_groups_by_trip():
    pipeline = build_default_pipeline()
    observations = [
        _tu_observation("t1", "s1", 1),
        _tu_observation("t1", "s2", 2),
        _tu_observation("t2", "s1", 1),
    ]
    ctx = _ctx(FakeModel(), observations)
    pipeline.run(ctx)

    trip_updates = ctx.state["trip_updates"]
    assert {tu.trip_id for tu in trip_updates} == {"t1", "t2"}
    t1 = next(tu for tu in trip_updates if tu.trip_id == "t1")
    assert [stu.stop_sequence for stu in t1.stop_time_updates] == [1, 2]
    assert ctx.state.get("errors", []) == []


def test_vehicle_position_observations_are_reported_not_fatal():
    pipeline = build_default_pipeline()
    vp = Observation(
        kind=ObservationKind.VEHICLE_POSITION,
        source_id="testagency",
        trip_id="t3",
        lat=0.0,
        lon=0.0,
    )
    ctx = _ctx(FakeModel(), [vp, _tu_observation("t1", "s1", 1)])
    pipeline.run(ctx)

    assert ctx.state["trip_updates"][0].trip_id == "t1"
    errors = ctx.state["errors"]
    assert any(e["stage"] == "match" and e["trip_id"] == "t3" for e in errors)


def test_predict_errors_are_contained_per_trip():
    pipeline = build_default_pipeline()
    observations = [_tu_observation("ok", "s1", 1), _tu_observation("bad", "s1", 2)]
    ctx = _ctx(FakeModel(fail_for_stop_sequence=2), observations)
    pipeline.run(ctx)

    trip_updates = ctx.state["trip_updates"]
    assert {tu.trip_id for tu in trip_updates} == {"ok"}
    errors = ctx.state["errors"]
    assert any(e["stage"] == "predict" and e["trip_id"] == "bad" for e in errors)


def test_overriding_predict_alone_leaves_other_stages_default():
    pipeline = build_default_pipeline()
    calls = []

    @pipeline.stage("predict")
    def custom_predict(features, ctx):
        calls.append(len(features))
        from stoptimes.core.contracts.prediction import StopTimePrediction

        return [
            StopTimePrediction(
                trip_id=row["trip_id"],
                start_date=row["start_date"],
                stop_id=row["stop_id"],
                stop_sequence=int(row["stop_sequence"]),
                horizon_s=42.0,
                predicted_arrival_time=ctx.now,
                model_id="custom",
                generated_at=ctx.now,
            )
            for _, row in features.iterrows()
        ]

    for name in ("ingest", "match", "features", "postprocess", "publish"):
        assert pipeline.is_default(name)
    assert not pipeline.is_default("predict")

    ctx = _ctx(FakeModel(), [_tu_observation("t1", "s1", 1)])
    pipeline.run(ctx)

    assert calls == [1]
    assert ctx.state["trip_updates"][0].stop_time_updates[0].horizon_s == 42.0


def test_load_pipeline_from_module_attr():
    pipeline = load_pipeline("stoptimes.serve.pipeline:build_default_pipeline")
    assert hasattr(pipeline, "run")
    for name in STAGE_ORDER:
        assert pipeline.is_default(name)


def test_load_pipeline_from_file_path(tmp_path):
    user_file = tmp_path / "my_pipeline.py"
    user_file.write_text(
        textwrap.dedent(
            """
            from stoptimes.serve.pipeline import build_default_pipeline

            pipeline = build_default_pipeline()
            """
        )
    )
    loaded = load_pipeline(f"{user_file}:pipeline")
    assert hasattr(loaded, "run")


def test_load_pipeline_rejects_spec_without_colon():
    with pytest.raises(ValueError):
        load_pipeline("no_colon_here")
