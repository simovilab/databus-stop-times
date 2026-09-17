"""
Tests for models/common/splits.py (roadmap 3.2): calendar-boundary temporal
splits with group purge, embargo, and a stable deterministic sort.

All synthetic and fast -- no real datasets loaded here (see
scratchpad validation done separately for datasets/mbta_segments_corpus.parquet
and datasets/bucr_segments_corpus.parquet).

Run with:
    uv run --no-sync python -m pytest packages/models/tests/test_splits.py -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stoptimes.models.common.splits import (
    SplitError,
    SplitResult,
    assert_no_group_overlap,
    day_grouped_split,
    fraction_split,
    service_day,
    verify,
)


# --------------------------------------------------------------------------
# service_day
# --------------------------------------------------------------------------

def test_service_day_offset_moves_late_night_row():
    # A 00:15 vp_ts with offset_hours=3 should land on the PREVIOUS calendar
    # date -- it's still "yesterday's" service day.
    ts = pd.Series(pd.to_datetime(["2026-01-02 00:15:00"]))
    result = service_day(ts, offset_hours=3)
    assert result.iloc[0] == pd.Timestamp("2026-01-01").date()


def test_service_day_after_offset_stays_same_date():
    ts = pd.Series(pd.to_datetime(["2026-01-02 10:00:00"]))
    result = service_day(ts, offset_hours=3)
    assert result.iloc[0] == pd.Timestamp("2026-01-02").date()


def test_service_day_requires_datetime():
    with pytest.raises(SplitError):
        service_day(pd.Series(["not-a-date"]), offset_hours=3)


# --------------------------------------------------------------------------
# Synthetic corpus builder
# --------------------------------------------------------------------------

def _make_corpus(
    n_days: int = 12,
    rows_per_day: int = 100,
    n_trips: int = 40,
    seed: int = 0,
    start: str = "2026-01-01",
    reuse_trip_across_days: bool = True,
) -> pd.DataFrame:
    """Synthetic vp-row corpus with trip_id (and optionally trip reused
    across service dates, mimicking MBTA's real hazard: the same trip_id
    string is scheduled on many different calendar days)."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, periods=n_days, freq="D")
    rows = []
    for day in days:
        for _ in range(rows_per_day):
            hour = int(rng.integers(4, 24))  # avoid the 0-3am offset zone by default
            minute = int(rng.integers(0, 60))
            ts = day + pd.Timedelta(hours=hour, minutes=minute)
            if reuse_trip_across_days:
                trip_id = f"trip{int(rng.integers(0, n_trips))}"
            else:
                trip_id = f"trip{day.strftime('%Y%m%d')}_{int(rng.integers(0, n_trips))}"
            rows.append(
                {
                    "vp_ts": ts,
                    "trip_id": trip_id,
                    "service_date": day.date(),
                    "value": float(rng.random()),
                }
            )
    df = pd.DataFrame(rows)
    return df


# --------------------------------------------------------------------------
# day_grouped_split: boundaries
# --------------------------------------------------------------------------

def test_day_boundaries_land_where_expected():
    df = _make_corpus(n_days=10, rows_per_day=50, n_trips=500, reuse_trip_across_days=False)
    res = day_grouped_split(df, offset_hours=3, test_days=2, val_days=2, embargo_days=0)

    all_days = sorted(df["vp_ts"].apply(lambda t: (t - pd.Timedelta(hours=3)).date()).unique())
    expected_test = all_days[-2:]
    expected_val = all_days[-4:-2]
    expected_train = all_days[:-4]

    assert res.report["day_boundaries"]["test"] == (expected_test[0], expected_test[-1])
    assert res.report["day_boundaries"]["val"] == (expected_val[0], expected_val[-1])
    assert res.report["day_boundaries"]["train"] == (expected_train[0], expected_train[-1])

    # Every row lands in exactly one split and none are silently lost, up to
    # whatever purge drops (with default trip_id reuse, purge is expected).
    total_out = len(res.train) + len(res.val) + len(res.test) + sum(res.report["rows_purged"].values())
    assert total_out == len(df)


def test_day_grouped_split_row_membership_matches_service_day():
    df = _make_corpus(n_days=8, rows_per_day=40, reuse_trip_across_days=False)
    res = day_grouped_split(df, offset_hours=3, test_days=1, val_days=1, embargo_days=0, purge=False)

    train_days = set(res.train["vp_ts"].apply(lambda t: (t - pd.Timedelta(hours=3)).date()))
    val_days = set(res.val["vp_ts"].apply(lambda t: (t - pd.Timedelta(hours=3)).date()))
    test_days = set(res.test["vp_ts"].apply(lambda t: (t - pd.Timedelta(hours=3)).date()))

    assert train_days.isdisjoint(val_days)
    assert val_days.isdisjoint(test_days)
    assert train_days.isdisjoint(test_days)


# --------------------------------------------------------------------------
# embargo
# --------------------------------------------------------------------------

def test_embargo_removes_exactly_expected_days_and_nothing_more():
    df = _make_corpus(n_days=12, rows_per_day=50, reuse_trip_across_days=False)

    res_no_embargo = day_grouped_split(df, offset_hours=3, test_days=2, val_days=2, embargo_days=0)
    res_embargo = day_grouped_split(df, offset_hours=3, test_days=2, val_days=2, embargo_days=1)

    assert res_embargo.report["days_embargoed"] == {"train": 1, "val": 1}
    assert res_no_embargo.report["days_embargoed"] == {"train": 0, "val": 0}

    # test window itself is untouched by embargo
    assert res_embargo.report["day_boundaries"]["test"] == res_no_embargo.report["day_boundaries"]["test"]

    # val loses exactly 1 leading day, train loses exactly 1 trailing day,
    # relative to the no-embargo split.
    val_days_no_embargo = pd.date_range(
        res_no_embargo.report["day_boundaries"]["val"][0],
        res_no_embargo.report["day_boundaries"]["val"][1],
    ).date
    val_days_embargo = pd.date_range(
        res_embargo.report["day_boundaries"]["val"][0],
        res_embargo.report["day_boundaries"]["val"][1],
    ).date
    assert len(val_days_embargo) == len(val_days_no_embargo) - 1

    train_days_no_embargo = pd.date_range(
        res_no_embargo.report["day_boundaries"]["train"][0],
        res_no_embargo.report["day_boundaries"]["train"][1],
    ).date
    train_days_embargo = pd.date_range(
        res_embargo.report["day_boundaries"]["train"][0],
        res_embargo.report["day_boundaries"]["train"][1],
    ).date
    assert len(train_days_embargo) == len(train_days_no_embargo) - 1


def test_embargo_too_large_raises():
    df = _make_corpus(n_days=6, rows_per_day=30)
    with pytest.raises(SplitError):
        day_grouped_split(df, offset_hours=3, test_days=2, val_days=2, embargo_days=10)


# --------------------------------------------------------------------------
# purge
# --------------------------------------------------------------------------

def _make_corpus_with_one_straddling_trip(n_days=10, rows_per_day=50, n_trips=200, seed=0):
    """Mostly distinct-per-day trips (so purge has nothing to do for them),
    plus one deliberately shared trip_id ("straddler") that appears on both
    an early (train-range) day and a late (test-range) day, so purge has an
    unambiguous, deterministic case to act on."""
    df = _make_corpus(n_days=n_days, rows_per_day=rows_per_day, n_trips=n_trips,
                       seed=seed, reuse_trip_across_days=False)
    days = sorted(df["vp_ts"].apply(lambda t: (t - pd.Timedelta(hours=3)).date()).unique())
    early_day, late_day = days[0], days[-1]
    day_series = df["vp_ts"].apply(lambda t: (t - pd.Timedelta(hours=3)).date())
    early_idx = df[day_series == early_day].index[:3]
    late_idx = df[day_series == late_day].index[:3]
    df.loc[early_idx, "trip_id"] = "straddler"
    df.loc[late_idx, "trip_id"] = "straddler"
    return df


def test_purge_trip_straddling_boundary_lands_in_exactly_one_split():
    df = _make_corpus_with_one_straddling_trip(n_days=10, rows_per_day=50)
    res = day_grouped_split(
        df, offset_hours=3, test_days=2, val_days=2, embargo_days=0,
        purge=True, group_cols=("trip_id",),
    )
    # No overlap, and the report's rows_purged accounts for the difference
    # between input size and output size.
    assert_no_group_overlap(res, ("trip_id",))
    total_purged = sum(res.report["rows_purged"].values())
    assert total_purged == 3  # exactly the 3 early rows of "straddler" get dropped from train
    assert res.report["rows_purged"]["train"] == 3
    assert "straddler" not in set(res.train["trip_id"])
    assert "straddler" in set(res.test["trip_id"])
    assert len(res.train) + len(res.val) + len(res.test) + total_purged == len(df)


def test_purge_group_cols_trip_and_service_date_keeps_reused_trip_in_multiple_splits():
    # MBTA hazard: same trip_id reused across service dates. With
    # group_cols=("trip_id",) alone, that trip_id must NOT straddle splits
    # (it will get purged down to one split). With group_cols=("trip_id",
    # "service_date"), the same trip_id string legitimately appears in
    # multiple splits because each (trip_id, service_date) pair is a
    # distinct group -- purge must NOT force it into one split.
    df = _make_corpus_with_one_straddling_trip(n_days=10, rows_per_day=50)
    # Make the "straddler" trip legitimately span train/val/test service
    # dates -- and give it a distinct service_date per occurrence so
    # (trip_id, service_date) grouping treats each occurrence separately.
    df["service_date"] = df["vp_ts"].apply(lambda t: (t - pd.Timedelta(hours=3)).date())

    res_trip_only = day_grouped_split(
        df, offset_hours=3, test_days=2, val_days=2, embargo_days=0,
        purge=True, group_cols=("trip_id",),
    )
    assert_no_group_overlap(res_trip_only, ("trip_id",))

    res_trip_and_date = day_grouped_split(
        df, offset_hours=3, test_days=2, val_days=2, embargo_days=0,
        purge=True, group_cols=("trip_id", "service_date"),
    )
    assert_no_group_overlap(res_trip_and_date, ("trip_id", "service_date"))

    # The same trip_id string DOES appear across multiple splits under
    # (trip_id, service_date) grouping -- that's expected and correct.
    trips_train = set(res_trip_and_date.train["trip_id"])
    trips_test = set(res_trip_and_date.test["trip_id"])
    assert trips_train & trips_test, (
        "expected trip_id reuse across service dates to survive purge when "
        "group_cols includes service_date"
    )

    # And purging under (trip_id, service_date) should drop strictly fewer
    # (or equal) rows than purging under trip_id alone, since it is a
    # coarser -- less restrictive -- grouping.
    purged_trip_only = sum(res_trip_only.report["rows_purged"].values())
    purged_trip_and_date = sum(res_trip_and_date.report["rows_purged"].values())
    assert purged_trip_and_date <= purged_trip_only


def test_assert_no_group_overlap_raises_with_helpful_message():
    train = pd.DataFrame({"trip_id": ["a", "a"], "vp_ts": pd.to_datetime(["2026-01-01", "2026-01-01"])})
    val = pd.DataFrame({"trip_id": ["a"], "vp_ts": pd.to_datetime(["2026-01-02"])})
    test = pd.DataFrame({"trip_id": ["b"], "vp_ts": pd.to_datetime(["2026-01-03"])})
    res = SplitResult(train=train, val=val, test=test, report={})
    with pytest.raises(SplitError, match="group overlap"):
        assert_no_group_overlap(res, ("trip_id",))


# --------------------------------------------------------------------------
# fraction_split: stable secondary sort / determinism
# --------------------------------------------------------------------------

def test_fraction_split_shuffled_input_produces_identical_splits():
    df = _make_corpus(n_days=6, rows_per_day=80, n_trips=25)
    df = df.reset_index(drop=True)
    df["row_id"] = df.index  # give an explicit row id so shuffling is well-defined

    shuffled = df.sample(frac=1.0, random_state=123).reset_index(drop=True)

    res_a = fraction_split(df, train_frac=0.6, val_frac=0.2, group_cols=("trip_id",), purge=False)
    res_b = fraction_split(shuffled, train_frac=0.6, val_frac=0.2, group_cols=("trip_id",), purge=False)

    # Compare by row_id sets (order-independent identity), not raw frame
    # equality, since the two inputs began in different orders but the same
    # *rows* -- keyed by row_id -- must land in the same splits.
    def ids(frame):
        return sorted(frame["row_id"].tolist())

    assert ids(res_a.train) == ids(res_b.train)
    assert ids(res_a.val) == ids(res_b.val)
    assert ids(res_a.test) == ids(res_b.test)


def test_fraction_split_with_explicit_row_id_is_order_independent():
    df = _make_corpus(n_days=5, rows_per_day=60, n_trips=20).reset_index(drop=True)
    df["rid"] = df.index

    shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)

    res_a = fraction_split(df, train_frac=0.7, val_frac=0.15, group_cols=("trip_id",),
                            purge=False, row_id_col="rid")
    res_b = fraction_split(shuffled, train_frac=0.7, val_frac=0.15, group_cols=("trip_id",),
                            purge=False, row_id_col="rid")

    assert sorted(res_a.train["rid"]) == sorted(res_b.train["rid"])
    assert sorted(res_a.val["rid"]) == sorted(res_b.val["rid"])
    assert sorted(res_a.test["rid"]) == sorted(res_b.test["rid"])


def test_fraction_split_chronological_order_holds():
    df = _make_corpus(n_days=6, rows_per_day=50)
    res = fraction_split(df, train_frac=0.7, val_frac=0.15, group_cols=("trip_id",), purge=False)
    assert res.train["vp_ts"].max() <= res.val["vp_ts"].min()
    assert res.val["vp_ts"].max() <= res.test["vp_ts"].min()


def test_fraction_split_purges_group_overlap():
    df = _make_corpus(n_days=6, rows_per_day=80, n_trips=400, reuse_trip_across_days=False)
    res = fraction_split(df, train_frac=0.6, val_frac=0.2, group_cols=("trip_id",), purge=True)
    assert_no_group_overlap(res, ("trip_id",))


def test_fraction_split_invalid_fracs_raise():
    df = _make_corpus(n_days=3, rows_per_day=20)
    with pytest.raises(SplitError):
        fraction_split(df, train_frac=0.9, val_frac=0.2)  # sums >= 1
    with pytest.raises(SplitError):
        fraction_split(df, train_frac=0, val_frac=0.2)
    with pytest.raises(SplitError):
        fraction_split(df, train_frac=0.5, val_frac=0)


# --------------------------------------------------------------------------
# min_rows_per_day filtering
# --------------------------------------------------------------------------

def test_min_rows_per_day_filters_sparse_days():
    df = _make_corpus(n_days=8, rows_per_day=100, n_trips=1000, reuse_trip_across_days=False)
    # Make one middle day sparse.
    sparse_day = sorted(df["vp_ts"].apply(lambda t: (t - pd.Timedelta(hours=3)).date()).unique())[3]
    mask_sparse_day = df["vp_ts"].apply(lambda t: (t - pd.Timedelta(hours=3)).date()) == sparse_day
    idx_to_drop = df[mask_sparse_day].index[10:]  # leave only 10 rows that day
    df = df.drop(index=idx_to_drop)

    res = day_grouped_split(df, offset_hours=3, test_days=2, val_days=2, min_rows_per_day=50)
    assert res.report["days_dropped_sparse"] == 1

    res_no_filter = day_grouped_split(df, offset_hours=3, test_days=2, val_days=2, min_rows_per_day=0)
    assert res_no_filter.report["days_dropped_sparse"] == 0


# --------------------------------------------------------------------------
# empty / degenerate inputs
# --------------------------------------------------------------------------

def test_day_grouped_split_empty_df_raises():
    df = pd.DataFrame({"vp_ts": pd.to_datetime([]), "trip_id": []})
    with pytest.raises(SplitError):
        day_grouped_split(df, offset_hours=3, test_days=1, val_days=1)


def test_day_grouped_split_not_enough_days_raises():
    df = _make_corpus(n_days=3, rows_per_day=20)
    with pytest.raises(SplitError):
        day_grouped_split(df, offset_hours=3, test_days=2, val_days=2)  # needs > 4 days


def test_fraction_split_empty_df_raises():
    df = pd.DataFrame({"vp_ts": pd.to_datetime([]), "trip_id": []})
    with pytest.raises(SplitError):
        fraction_split(df, train_frac=0.7, val_frac=0.15)


def test_day_grouped_split_requires_datetime_column():
    df = pd.DataFrame({"vp_ts": ["not-a-date"] * 5, "trip_id": ["a"] * 5})
    with pytest.raises(SplitError):
        day_grouped_split(df, offset_hours=3, test_days=1, val_days=1)


def test_verify_reports_all_checks_true_on_healthy_split():
    df = _make_corpus(n_days=10, rows_per_day=50, reuse_trip_across_days=False)
    res = day_grouped_split(df, offset_hours=3, test_days=2, val_days=2, embargo_days=1)
    checks = verify(res, group_cols=("trip_id",))
    assert checks["train_nonempty"]
    assert checks["val_nonempty"]
    assert checks["test_nonempty"]
    assert checks["no_group_overlap"]
    assert checks["chronological_order"]
