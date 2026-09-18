"""Tests for stoptimes.core.features.registry."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from stoptimes.core.features.registry import (
    FeatureNotFoundError,
    MissingFeatureInputsError,
    build,
    get,
    list_features,
)


def test_list_features_returns_sorted_names():
    names = [f.name for f in list_features()]
    assert names == sorted(names)
    assert "hour" in names
    assert "distance_to_stop" in names


def test_list_features_filters_by_tag():
    spatial = list_features(tag="spatial")
    assert spatial
    assert all("spatial" in f.tags for f in spatial)
    assert "distance_to_stop" in [f.name for f in spatial]

    rolling = list_features(tag="rolling")
    assert "route_congestion_index" in [f.name for f in rolling]
    assert all("rolling" in f.tags for f in rolling)

    segment = list_features(tag="segment")
    assert {"from_stop_id", "segment_index"} <= {f.name for f in segment}


def test_get_unknown_feature_raises_with_available_names():
    with pytest.raises(FeatureNotFoundError, match="unknown feature 'nope'"):
        get("nope")


def test_every_feature_has_a_description_and_tags():
    for feature in list_features():
        assert feature.description
        assert feature.tags


def test_build_temporal_from_timestamp():
    frame = pd.DataFrame(
        {
            "vp_ts": [
                datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc),  # Thursday
                datetime(2026, 9, 20, 20, 0, tzinfo=timezone.utc),  # Sunday
            ]
        }
    )
    out = build(["hour", "is_weekend", "hour_sin", "hour_cos"], frame)
    assert "hour" in out.columns
    assert out["is_weekend"].tolist() == [False, True]
    assert "hour_sin" in out.columns and "hour_cos" in out.columns


def test_build_temporal_reuses_existing_columns_without_a_timestamp():
    frame = pd.DataFrame(
        {
            "hour": [8],
            "day_of_week": [1],
            "is_weekend": [False],
            "is_holiday": [False],
            "time_of_day_bin": ["morning"],
            "is_peak_hour": [True],
        }
    )
    out = build(["hour", "hour_sin"], frame)
    assert out["hour"].tolist() == [8]
    assert "hour_sin" in out.columns


def test_build_missing_timestamp_and_columns_raises_clear_error():
    frame = pd.DataFrame({"other": [1, 2]})
    with pytest.raises(MissingFeatureInputsError, match="timestamp"):
        build(["hour"], frame)


def test_build_passthrough_feature_requires_its_column():
    frame = pd.DataFrame({"vp_lat": [1.0]})
    with pytest.raises(MissingFeatureInputsError, match="distance_to_stop"):
        build(["distance_to_stop"], frame)


def test_build_passthrough_feature_passes_when_column_present():
    frame = pd.DataFrame({"distance_to_stop": [50.0]})
    out = build(["distance_to_stop"], frame)
    assert out["distance_to_stop"].tolist() == [50.0]


def test_build_does_not_mutate_input_frame():
    frame = pd.DataFrame({"vp_ts": [datetime(2026, 1, 1, tzinfo=timezone.utc)]})
    build(["hour"], frame)
    assert list(frame.columns) == ["vp_ts"]


def test_build_unknown_feature_name_raises():
    frame = pd.DataFrame({"a": [1]})
    with pytest.raises(FeatureNotFoundError):
        build(["not-a-real-feature"], frame)


def test_build_composes_multiple_features():
    frame = pd.DataFrame(
        {
            "vp_ts": [datetime(2026, 9, 17, 8, 0, tzinfo=timezone.utc)],
            "distance_to_stop": [50.0],
        }
    )
    out = build(["hour", "distance_to_stop"], frame)
    assert "hour" in out.columns
    assert "distance_to_stop" in out.columns
