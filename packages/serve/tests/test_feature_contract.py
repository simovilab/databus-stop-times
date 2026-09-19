"""The docs/PLAN.md §7.4 defect: an artifact's declared `required_features`
must be checked against what the serving pipeline can actually compute --
never silently served on a missing input (`None`) and never refused only
per-row, after the fact.

Two things this file must prove, per the fix's own bar:

- A model declaring a feature `core.features.registry` knows about but
  `default_match`/`default_features` cannot compute is refused up front,
  naming that feature -- not returned as NaN, not left to error per row.
- A model declaring only producible features is served end to end and gets
  *real*, finite values for those features (not merely "the run didn't
  crash") -- checked on the `default_features` intermediate, not just the
  final prediction, per §7.4.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from stoptimes.core.contracts.data import Observation, ObservationKind
from stoptimes.core.contracts.pipeline import PipelineContext
from stoptimes.core.contracts.source import Source
from stoptimes.serve.pipeline import (
    MatchedObservation,
    default_features,
    default_predict,
    producible_features,
    unproducible_registered_features,
)


def _source() -> Source:
    return Source(id="testagency", name="Test Agency", gtfs_static_url="/dev/null")


def _ctx(model) -> PipelineContext:
    return PipelineContext(
        source=_source(),
        static_feed=None,
        model=model,
        now=datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc),
    )


def _matched(**extra) -> MatchedObservation:
    observation = Observation(
        kind=ObservationKind.TRIP_UPDATE,
        source_id="testagency",
        trip_id="t1",
        route_id="r1",
        direction_id=0,
        start_date="20260917",
        stop_id="s1",
        stop_sequence=1,
        ts=pd.Timestamp("2026-09-17T12:00:00Z"),
    )
    return MatchedObservation(
        trip_id="t1",
        route_id="r1",
        direction_id=0,
        start_date="20260917",
        vehicle_id=None,
        stop_id="s1",
        stop_sequence=1,
        observation=observation,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# producible_features / unproducible_registered_features
# ---------------------------------------------------------------------------


def test_producible_features_includes_temporal_and_match_spatial_and_identity():
    producible = producible_features()
    assert "hour" in producible  # temporal, computed live
    assert "distance_to_stop" in producible  # what default_match attaches
    assert "stops_ahead" in producible
    assert "trip_id" in producible  # identity, always present


def test_producible_features_excludes_train_only_columns():
    producible = producible_features()
    assert "shape_progress" not in producible
    assert "vp_lat" not in producible
    assert "recent_seg_traversal_s" not in producible
    assert "from_stop_id" not in producible


def test_unproducible_registered_features_flags_only_registered_unproducible_names():
    missing = unproducible_registered_features(["hour", "distance_to_stop", "shape_progress", "not_a_real_feature"])
    assert missing == ["shape_progress"]


def test_unproducible_registered_features_empty_for_fully_producible_list():
    assert unproducible_registered_features(["hour", "distance_to_stop", "trip_id"]) == []


# ---------------------------------------------------------------------------
# default_features: real registry wiring, asserted on the intermediate frame
# ---------------------------------------------------------------------------


def test_default_features_computes_temporal_columns_via_registry():
    matched = [_matched(distance_to_stop=123.4, stops_ahead=2.0)]
    features = default_features(matched, _ctx(model=None))

    assert len(features) == 1
    row = features.iloc[0]
    # Real, finite values -- not merely "the column exists".
    for col in ("hour", "day_of_week", "hour_sin", "hour_cos", "dow_sin", "dow_cos"):
        assert col in features.columns
        assert np.isfinite(float(row[col]))
    assert row["hour"] == 12  # 2026-09-17T12:00:00Z, agency tz UTC
    assert bool(row["is_weekend"]) is False
    # Spatial extras from `match` pass through untouched.
    assert row["distance_to_stop"] == 123.4
    assert row["stops_ahead"] == 2.0


def test_default_features_empty_input_returns_empty_frame():
    features = default_features([], _ctx(model=None))
    assert features.empty


# ---------------------------------------------------------------------------
# A model requiring only producible features: served end to end, real values.
# ---------------------------------------------------------------------------


class _ProducibleFeatureModel:
    family = "fake-producible"
    required_features = ["distance_to_stop", "hour"]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        row = X.iloc[0]
        assert pd.notna(row["distance_to_stop"])
        assert pd.notna(row["hour"])
        return np.array([float(row["distance_to_stop"]) + float(row["hour"])])


def test_model_requiring_only_producible_features_serves_real_values():
    model = _ProducibleFeatureModel()
    ctx = _ctx(model)
    matched = [_matched(distance_to_stop=200.0, stops_ahead=1.0)]

    features = default_features(matched, ctx)
    # Assert on the intermediate the stage actually produced, not only the
    # final prediction (docs/PLAN.md §7.4): both declared inputs are present
    # and finite before predict ever runs.
    row = features.iloc[0]
    assert np.isfinite(float(row["distance_to_stop"]))
    assert np.isfinite(float(row["hour"]))

    predictions = default_predict(features, ctx)
    assert len(predictions) == 1
    assert predictions[0].horizon_s == pytest.approx(200.0 + 12.0)


# ---------------------------------------------------------------------------
# A model requiring an unproducible feature: default_features/default_predict
# alone do NOT refuse (that's model_loading's job at load time) -- but they
# must not fabricate a value either. This documents why per-row silence was
# the bug: refusal belongs at load time, not here.
# ---------------------------------------------------------------------------


class _UnproducibleFeatureModel:
    family = "fake-unproducible"
    required_features = ["shape_progress"]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        row = X.iloc[0]
        assert row.get("shape_progress") is None
        return np.array([1.0])


def test_default_features_does_not_fabricate_an_unproducible_column():
    matched = [_matched(distance_to_stop=10.0)]
    features = default_features(matched, _ctx(model=_UnproducibleFeatureModel()))
    assert "shape_progress" not in features.columns
