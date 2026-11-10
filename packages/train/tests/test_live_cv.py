"""Leak-free cross-validation of live frames: fold plans, per-fold frames, scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from stoptimes.core.features.live import LIVE_FEATURES
from stoptimes.core.gtfs.static import GtfsStaticLoader
from stoptimes.train.live_cv import (
    Fold,
    _model_params,
    _requested_days,
    aggregate,
    build_frame_cache,
    day_grouped_kfold,
    fold_frame,
    holdout,
    rolling_origin_by_week,
    run_cv,
    score_fold,
)
from stoptimes.train.live_frame import LiveFrameParams, build_live_dataset, prepare_trip_records

from .test_live_frame import PACES, TZ, _drive, _write_feed


def _weekdays(start: str, end: str) -> list[str]:
    return [d.strftime("%Y%m%d") for d in pd.date_range(start, end) if d.dayofweek < 5]


# ---------------------------------------------------------------------------
# Fold plans
# ---------------------------------------------------------------------------


def test_rolling_origin_tests_each_week_on_all_earlier_weeks_and_stops_on_the_last():
    days = _weekdays("2026-08-10", "2026-09-18")  # six ISO weeks, W33..W38
    folds = rolling_origin_by_week(days, min_train_weeks=4)
    assert [f.name for f in folds] == ["2026-W37", "2026-W38"]
    first = folds[0]
    assert first.fit_days == tuple(_weekdays("2026-08-10", "2026-08-28"))  # W33-W35
    assert first.es_days == tuple(
        _weekdays("2026-08-31", "2026-09-04")
    )  # W36, the last training week
    assert first.test_days == tuple(_weekdays("2026-09-07", "2026-09-11"))
    assert not first.es_is_test
    for f in folds:  # strictly causal: every training day precedes every test day
        assert max(f.fit_days + f.es_days) < min(f.test_days)
        assert max(f.fit_days) < min(f.es_days)


def test_rolling_origin_uses_iso_weeks_across_the_year_boundary():
    days = _weekdays("2026-12-07", "2027-01-08")  # 2027-01-01 is in ISO week 2026-W53
    folds = rolling_origin_by_week(days, min_train_weeks=3)
    assert [f.name for f in folds] == ["2026-W53", "2027-W01"]
    assert folds[0].test_days == tuple(_weekdays("2026-12-28", "2027-01-01"))


def test_rolling_origin_rejects_bad_input():
    with pytest.raises(ValueError, match="min_train_weeks"):
        rolling_origin_by_week(_weekdays("2026-08-10", "2026-09-18"), min_train_weeks=1)
    with pytest.raises(ValueError, match="sorted"):
        rolling_origin_by_week(["20260812", "20260811"])
    assert rolling_origin_by_week(_weekdays("2026-08-10", "2026-08-28"), min_train_weeks=4) == []


def test_kfold_tests_every_day_once_and_stops_on_held_out_training_days():
    days = _weekdays("2026-08-10", "2026-11-06")
    folds = day_grouped_kfold(days, k=5, seed=7)
    assert len(folds) == 5
    tested = [d for f in folds for d in f.test_days]
    assert sorted(tested) == days  # a partition of the days
    for f in folds:
        train = set(f.fit_days) | set(f.es_days)
        assert train | set(f.test_days) == set(days)
        assert not set(f.es_days) & set(f.test_days) and not set(f.fit_days) & set(f.es_days)
        assert len(f.es_days) == round(0.15 * len(train))
    assert day_grouped_kfold(days, k=5, seed=7) == folds  # seeded
    assert day_grouped_kfold(days, k=5, seed=8) != folds


def test_kfold_rejects_bad_input_and_keeps_at_least_one_es_day():
    with pytest.raises(ValueError, match="k >= 2"):
        day_grouped_kfold(_weekdays("2026-08-10", "2026-08-12"), k=5)
    with pytest.raises(ValueError, match="es_fraction"):
        day_grouped_kfold(_weekdays("2026-08-10", "2026-09-18"), k=3, es_fraction=1.0)
    folds = day_grouped_kfold(_weekdays("2026-08-10", "2026-08-18"), k=3, es_fraction=0.01)
    assert all(len(f.es_days) == 1 for f in folds)


def test_holdout_reproduces_the_production_split():
    days = _weekdays("2026-10-19", "2026-10-30")
    (fold,) = holdout(days, "20261026")
    assert fold.fit_days == tuple(_weekdays("2026-10-19", "2026-10-23"))
    assert fold.test_days == fold.es_days == tuple(_weekdays("2026-10-26", "2026-10-30"))
    assert fold.es_is_test


def test_fold_rejects_overlaps():
    with pytest.raises(ValueError, match="overlap"):
        Fold("x", ("a", "b"), ("b",), ("c",))
    with pytest.raises(ValueError, match="early-stopping days overlap"):
        Fold("x", ("a",), ("c",), ("c",))
    with pytest.raises(ValueError, match="non-empty"):
        Fold("x", (), ("b",), ("c",))


# ---------------------------------------------------------------------------
# Per-fold frames (synthetic feed shared with test_live_frame)
# ---------------------------------------------------------------------------

DAYS = [f"202610{d:02d}" for d in range(1, 11)]


@pytest.fixture(scope="module")
def by_day(tmp_path_factory):
    feed = GtfsStaticLoader(_write_feed(tmp_path_factory.mktemp("cvfeed") / "gtfs.zip"))
    rows = []
    for i, day in enumerate(DAYS):
        rows += _drive(day, "t_day", 8, 0, delay_s=20.0 * i, pace=PACES[i])
        if i < 8:  # route rB never runs on the last two days
            rows += _drive(day, "t_late", 25, 30, delay_s=-10.0 * i, pace=PACES[i])
    records, _ = prepare_trip_records(
        pd.DataFrame(rows), feed, config_timezone="UTC", holiday_region="US_MA"
    )
    return records


@pytest.fixture(scope="module")
def cache(by_day):
    return build_frame_cache(DAYS, lambda d: by_day[d], tz=TZ, params=LiveFrameParams())


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["day", "trip_id", "t", "stop_idx"]).reset_index(drop=True)


@pytest.mark.parametrize("fit_n", [6, 8])
def test_fold_frame_equals_a_production_build_on_the_fit_days(cache, by_day, fit_n):
    fold = Fold("f", tuple(DAYS[:fit_n]), (DAYS[fit_n],), tuple(DAYS[fit_n + 1 :]))
    frame = fold_frame(cache, fold)
    ref = build_live_dataset(
        fold.fit_days,
        (*fold.es_days, *fold.test_days),
        lambda d: by_day[d],
        tz=TZ,
        holiday_region="US_MA",
        params=LiveFrameParams(),
    ).frame
    got, want = _sorted(frame), _sorted(ref)
    assert len(got) == len(want)
    for col in [*LIVE_FEATURES, "time_to_arrival_seconds"]:
        np.testing.assert_array_equal(got[col].to_numpy(), want[col].to_numpy(), err_msg=col)
    assert (got.loc[want["split"] == "train", "split"] == "fit").all()
    assert set(got.loc[want["split"] == "val", "split"]) == {"es", "test"}
    assert set(frame.loc[frame.split == "es", "day"]) == set(fold.es_days)


def test_fold_assets_never_see_test_days(cache, by_day):
    # Same fit days, different test days: fit and es rows are identical (nothing flows back).
    a = fold_frame(cache, Fold("a", tuple(DAYS[:6]), (DAYS[6],), (DAYS[7],)))
    b = fold_frame(cache, Fold("b", tuple(DAYS[:6]), (DAYS[6],), tuple(DAYS[7:])))
    keep = lambda f: _sorted(f[f.split != "test"])  # noqa: E731
    pd.testing.assert_frame_equal(keep(a), keep(b))
    # The vocabulary is fitted on fit days: a route seen only on test days would be unknown.
    c = fold_frame(cache, Fold("c", tuple(DAYS[8:]), (DAYS[0],), (DAYS[1],)))
    assert c.loc[c.route_id == "rB", "route_code"].isna().all()
    assert c.loc[c.route_id == "rA", "route_code"].notna().all()


def test_holdout_fold_scores_and_stops_on_the_same_test_rows(cache):
    (fold,) = holdout(DAYS, DAYS[8])
    frame = fold_frame(cache, fold)
    assert set(frame["split"]) == {"fit", "test"}


# ---------------------------------------------------------------------------
# Scoring and aggregation
# ---------------------------------------------------------------------------


def test_score_fold_hand_computed():
    test = pd.DataFrame(
        {
            "time_to_arrival_seconds": [60.0, 200.0, 400.0, 900.0, 1500.0],
            "sched_rem": [0.0, 200.0, 400.0, 900.0, 1500.0],
            "prior_rem": [70.0, np.nan, 400.0, 900.0, 1500.0],
        }
    )
    pred = np.array([70.0, 190.0, 430.0, 900.0, 1400.0])
    s = score_fold(test, pred)
    assert s["model_mae"] == pytest.approx((10 + 10 + 30 + 0 + 100) / 5)
    assert s["model_medae"] == pytest.approx(10.0)
    assert s["model_bias"] == pytest.approx((10 - 10 + 30 + 0 - 100) / 5)
    assert s["sched_mae"] == pytest.approx(12.0)
    assert s["prior_mae"] == pytest.approx(2.5) and s["prior_coverage"] == pytest.approx(0.8)
    assert s["model_mae_hlt2"] == 10 and s["model_mae_h2_5"] == 10 and s["model_mae_h5_10"] == 30
    assert s["model_mae_h10_20"] == 0 and s["model_mae_h20p"] == 100


def test_aggregate_pools_by_row_count():
    rows = [
        {"model_mae": 10.0, "sched_mae": 20.0, "prior_mae": 15.0},
        {"model_mae": 30.0, "sched_mae": 40.0, "prior_mae": 25.0},
    ]
    sums = [
        {"model": (100.0, 10), "sched": (200.0, 10), "prior": (150.0, 10)},
        {"model": (900.0, 30), "sched": (1200.0, 30), "prior": (750.0, 30)},
    ]
    agg = aggregate(rows, sums)
    assert agg["model_mae"]["mean"] == 20.0 and agg["model_mae"]["pooled"] == pytest.approx(25.0)
    assert agg["model_mae"]["min"] == 10.0 and agg["model_mae"]["max"] == 30.0
    assert agg["model_mae"]["std"] == pytest.approx(np.std([10, 30], ddof=1))


def test_run_cv_end_to_end_on_the_synthetic_feed(cache):
    folds = [Fold("f0", tuple(DAYS[:5]), (DAYS[5],), tuple(DAYS[6:8])), *holdout(DAYS, DAYS[8])]
    params = {"n_estimators": 20, "early_stopping_rounds": 5, "max_depth": 3, "n_jobs": 1}
    table, summary = run_cv(cache, folds, list(LIVE_FEATURES), params, seed=1)
    assert list(table["fold"]) == ["f0", f"holdout-{DAYS[8]}"]
    assert (table["n_test"] > 0).all() and (table["best_iteration"] >= 0).all()
    assert table.loc[1, "n_es"] == table.loc[1, "n_test"]  # holdout stops on test
    assert summary["n_folds"] == 2 and np.isfinite(summary["model_mae"]["pooled"])
    with pytest.raises(ValueError, match="not live features"):
        run_cv(cache, folds, ["nope"], params, seed=1)


def test_cli_helpers():
    class _Model:
        params = {"n_jobs": 12, "max_depth": 8}

    class _Exp:
        model = _Model()

    assert _model_params(_Exp(), None, None)["n_jobs"] == 6  # capped
    assert _model_params(_Exp(), '{"max_depth": 4}', 2) == {"n_jobs": 2, "max_depth": 4}
    assert _requested_days("2026-11-06:2026-11-09", weekdays_only=True) == ["20261106", "20261109"]
    assert len(_requested_days("2026-11-06:2026-11-09", weekdays_only=False)) == 4
