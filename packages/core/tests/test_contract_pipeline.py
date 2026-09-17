"""Tests for contracts.pipeline: stage order, context, and that hooks compose
through the reference SequentialPipeline."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from stoptimes.core.contracts.pipeline import (
    STAGE_ORDER,
    PipelineContext,
    SequentialPipeline,
)


def _ctx(**overrides) -> PipelineContext:
    base = dict(source=None, static_feed=None, model=None, now=datetime(2026, 9, 17, tzinfo=timezone.utc))
    base.update(overrides)
    return PipelineContext(**base)


def test_stage_order_is_fixed():
    assert STAGE_ORDER == ("ingest", "match", "features", "predict", "postprocess", "publish")


def test_missing_stage_raises_keyerror():
    pipeline = SequentialPipeline()
    with pytest.raises(KeyError):
        pipeline.run(_ctx())


def test_hooks_compose_in_order_and_share_context_state():
    pipeline = SequentialPipeline()
    calls: list[str] = []

    for name in STAGE_ORDER:

        def make_hook(stage_name):
            def hook(payload, ctx):
                calls.append(stage_name)
                ctx.state.setdefault("trail", []).append(stage_name)
                return (payload or []) + [stage_name]

            return hook

        pipeline.stage(name)(make_hook(name))

    ctx = _ctx()
    result = pipeline.run(ctx)

    assert calls == list(STAGE_ORDER)
    assert result == list(STAGE_ORDER)
    assert ctx.state["trail"] == list(STAGE_ORDER)


def test_stage_decorator_returns_hook_unchanged_and_is_independently_callable():
    pipeline = SequentialPipeline()

    @pipeline.stage("predict")
    def predict(payload, ctx):
        return "predicted"

    # still directly callable/testable without going through the pipeline
    assert predict(None, _ctx()) == "predicted"


def test_re_registering_a_stage_replaces_the_previous_hook():
    pipeline = SequentialPipeline()
    pipeline.stage("ingest")(lambda payload, ctx: "first")
    pipeline.stage("ingest")(lambda payload, ctx: "second")
    pipeline.stage("match")(lambda payload, ctx: payload)
    pipeline.stage("features")(lambda payload, ctx: payload)
    pipeline.stage("predict")(lambda payload, ctx: payload)
    pipeline.stage("postprocess")(lambda payload, ctx: payload)
    pipeline.stage("publish")(lambda payload, ctx: payload)

    assert pipeline.run(_ctx()) == "second"


def test_context_now_is_the_clock_stages_must_use():
    ctx = _ctx(now=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert ctx.now.tzinfo is not None
