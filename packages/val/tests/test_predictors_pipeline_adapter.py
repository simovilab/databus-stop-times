"""test_predictors_pipeline_adapter — PipelineTripUpdatesPredictor (T6 deliverable 3).

Offline: no real `stoptimes.serve.pipeline.load_pipeline` is required to
exist (it's still T4's Phase 2 placeholder) -- these tests monkeypatch
`pipeline_adapter._import_pipeline_module` directly (rather than fighting
namespace-package import semantics via `sys.modules`) to inject a fake
`stoptimes.serve.pipeline` module, and separately assert the two guard
error paths.
"""

from __future__ import annotations

import types
from datetime import UTC, datetime

import pytest

import stoptimes.val.predictors.pipeline_adapter as pipeline_adapter
from stoptimes.core.domain.models import Stop, TripInstance, VehicleState
from stoptimes.val.predictors.pipeline_adapter import PipelineTripUpdatesPredictor

TRIP = TripInstance(
    source="test",
    trip_id="t1",
    vehicle_id="v1",
    route_id="22",
    direction_id=0,
    shape_id="s1",
    start_date="20260701",
    started_at=1_000_000.0,
)

STATE = VehicleState(
    epoch=1_000_100.0,
    lat=42.1,
    lon=-71.0,
    progress_m=200.0,
    speed=5.0,
    current_status="IN_TRANSIT_TO",
)


def _stops() -> list[Stop]:
    return [
        Stop(stop_id="s1", stop_sequence=1, lat=42.2, lon=-71.1),
        Stop(stop_id="s2", stop_sequence=2, lat=42.3, lon=-71.2),
    ]


class _FakePipeline:
    def __init__(self, rows):
        self._rows = rows
        self.last_ctx = None

    def run(self, ctx):
        self.last_ctx = ctx
        return self._rows


def _fake_module(**attrs):
    return lambda: types.SimpleNamespace(**attrs)


def test_predictor_id_includes_filename() -> None:
    predictor = PipelineTripUpdatesPredictor("/tmp/my_pipeline.py")
    assert predictor.predictor_id == "pipeline:my_pipeline.py"


def test_empty_upcoming_stops_short_circuits_without_loading(monkeypatch) -> None:
    def _boom():
        raise AssertionError("_import_pipeline_module should not be called")

    monkeypatch.setattr(pipeline_adapter, "_import_pipeline_module", _boom)

    predictor = PipelineTripUpdatesPredictor("pipeline.py")
    assert predictor.predict(TRIP, [], STATE, None) == []


def test_predict_extracts_matching_stop_predictions(monkeypatch) -> None:
    rows = [
        {
            "stop_id": "s1",
            "predicted_arrival_time": datetime.fromtimestamp(STATE.epoch + 60, tz=UTC),
        },
        {
            "stop_id": "s2",
            "predicted_arrival_time": datetime.fromtimestamp(STATE.epoch + 180, tz=UTC),
        },
        {"stop_id": "unknown-stop", "predicted_arrival_time": datetime.now(tz=UTC)},
    ]
    fake_pipeline = _FakePipeline(rows)
    monkeypatch.setattr(
        pipeline_adapter, "_import_pipeline_module", _fake_module(load_pipeline=lambda path: fake_pipeline)
    )

    predictor = PipelineTripUpdatesPredictor("pipeline.py")
    preds = predictor.predict(TRIP, _stops(), STATE, None)

    assert len(preds) == 2
    assert preds[0].stop_id == "s1"
    assert preds[0].predicted_arrival_epoch == pytest.approx(STATE.epoch + 60)
    assert preds[1].stop_id == "s2"

    # Context threaded through to the pipeline carries the run's logical clock.
    assert fake_pipeline.last_ctx.now == datetime.fromtimestamp(STATE.epoch, tz=UTC)
    assert fake_pipeline.last_ctx.source == "test"


def test_predict_falls_back_to_ctx_state_predictions(monkeypatch) -> None:
    """A pipeline whose `publish` stage hands off to a sink (returns None)
    can still be scored if it stashes rows in ctx.state["predictions"]."""

    class _PublishToSinkPipeline:
        def run(self, ctx):
            ctx.state["predictions"] = [
                {
                    "stop_id": "s1",
                    "predicted_arrival_time": STATE.epoch + 30.0,
                }
            ]
            return None

    monkeypatch.setattr(
        pipeline_adapter,
        "_import_pipeline_module",
        _fake_module(load_pipeline=lambda path: _PublishToSinkPipeline()),
    )

    predictor = PipelineTripUpdatesPredictor("pipeline.py")
    preds = predictor.predict(TRIP, _stops(), STATE, None)

    assert len(preds) == 1
    assert preds[0].predicted_arrival_epoch == pytest.approx(STATE.epoch + 30.0)


def test_predict_raises_import_error_when_serve_not_installed(monkeypatch) -> None:
    def _raise():
        raise ImportError("no module named stoptimes.serve.pipeline")

    monkeypatch.setattr(pipeline_adapter, "_import_pipeline_module", _raise)

    predictor = PipelineTripUpdatesPredictor("pipeline.py")
    with pytest.raises(ImportError, match="stoptimes-serve"):
        predictor.predict(TRIP, _stops(), STATE, None)


def test_predict_raises_attribute_error_when_load_pipeline_missing(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_adapter, "_import_pipeline_module", _fake_module())  # no load_pipeline attr

    predictor = PipelineTripUpdatesPredictor("pipeline.py")
    with pytest.raises(AttributeError, match="load_pipeline"):
        predictor.predict(TRIP, _stops(), STATE, None)


def test_attach_static_loader_passed_through_as_static_feed(monkeypatch) -> None:
    fake_pipeline = _FakePipeline([])
    monkeypatch.setattr(
        pipeline_adapter, "_import_pipeline_module", _fake_module(load_pipeline=lambda path: fake_pipeline)
    )

    predictor = PipelineTripUpdatesPredictor("pipeline.py")
    sentinel = object()
    predictor.attach_static_loader(sentinel)
    predictor.predict(TRIP, _stops(), STATE, None)

    assert fake_pipeline.last_ctx.static_feed is sentinel
