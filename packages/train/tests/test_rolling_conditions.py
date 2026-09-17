"""Tests for ``stoptimes.train.datasets.rolling_conditions``.

The headline requirement of that module is causality: a feature computed for
a row at time `t` must be identical no matter what happens at or after `t`.
Every test here is either a direct proof of that (the shuffle-future test,
the boundary test) or a check that the arithmetic underneath it is right
(hand-computed EWMA/expanding values, cold-start fallback, reorder-invariance).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stoptimes.train.datasets.rolling_conditions import (
    ROLLING_CONDITIONS_FEATURES,
    RollingConditionsConfig,
    add_rolling_conditions_features,
)

UTC = "UTC"
BASE = pd.Timestamp("2026-08-14 12:00:00", tz=UTC)


def _row(trip, veh, route, seg_idx, from_s, to_s, vp_offset_s, arr_offset_s):
    return dict(
        trip_id=trip, route_id=route, vehicle_id=veh, stop_id=to_s,
        from_stop_id=from_s, segment_index=seg_idx,
        vp_ts=BASE + pd.Timedelta(seconds=vp_offset_s),
        actual_arrival=BASE + pd.Timedelta(seconds=arr_offset_s),
    )


def _corpus() -> pd.DataFrame:
    """Two trip instances on the same route/segment, one vehicle each.

    Trip A: seg0 (S0->S1) arrives at +60s, seg1 (S1->S2) arrives at +160s
    (duration 100s), with an extra mid-segment ping at +130s.
    Trip B: seg0 (S0->S1) arrives at +560s, seg1 (S1->S2) arrives at +700s
    (duration 140s). Trip B's seg1 query rows should be able to see Trip A's
    already-completed seg1 traversal (100s, completed at +160s) since it
    happened strictly before every Trip B seg1 timestamp.
    """
    rows = [
        _row("A", "v1", "R1", 0, "S0", "S1", 0, 60),
        _row("A", "v1", "R1", 1, "S1", "S2", 60, 160),
        _row("A", "v1", "R1", 1, "S1", "S2", 130, 160),
        _row("B", "v2", "R1", 0, "S0", "S1", 500, 560),
        _row("B", "v2", "R1", 1, "S1", "S2", 560, 700),
        _row("B", "v2", "R1", 1, "S1", "S2", 630, 700),
    ]
    return pd.DataFrame(rows)


def _cfg(**overrides) -> RollingConditionsConfig:
    base = dict(offset_hours=7, min_observations=1)
    base.update(overrides)
    return RollingConditionsConfig(**base)


# --------------------------------------------------------------------------
# Causality: the headline test.
# --------------------------------------------------------------------------

def test_shuffle_future_leaves_past_features_unchanged():
    """Perturbing observations that complete strictly after a row's `vp_ts`
    must leave that row's features byte-identical. This is proven by
    computing features once, then mutating every completion (`actual_arrival`
    + implicitly its duration) that lies in the *second half* of the corpus
    in time, and re-running: rows drawn from the first half must not move at
    all, even though the corpus-level statistics for shared keys (segment,
    route, vehicle) did change.
    """
    # `_corpus()` alone isn't enough to prove the mutation propagates anywhere
    # (Trip B is the last trip in it, so nothing downstream ever reads its
    # completions) -- add a third trip, E, querying the same segment as B
    # *after* B completes, so B's completion is genuinely load-bearing for
    # E's features and the mutation has somewhere real to show up.
    df = pd.concat([
        _corpus(),
        pd.DataFrame([_row("E", "v5", "R1", 1, "S1", "S2", 1000, 1100)]),
    ], ignore_index=True)
    cfg = _cfg(min_observations=1)

    out_before = add_rolling_conditions_features(df, cfg)

    # Perturb every row whose OWN segment completes at or after Trip B's
    # first timestamp (+500s) -- i.e. corrupt the "future" relative to Trip
    # A's rows, and also relative to Trip B's own earlier rows within itself.
    cutoff = BASE + pd.Timedelta(seconds=500)
    future_mask = df["actual_arrival"] >= cutoff
    assert future_mask.any() and (~future_mask).any()

    df_mutated = df.copy()
    df_mutated.loc[future_mask, "actual_arrival"] = (
        df_mutated.loc[future_mask, "actual_arrival"] + pd.Timedelta(hours=999)
    )

    out_after = add_rolling_conditions_features(df_mutated, cfg)

    past_rows = df["vp_ts"] < cutoff
    assert past_rows.sum() >= 2  # sanity: the shuffle actually exercises something

    for col in ROLLING_CONDITIONS_FEATURES:
        left = out_before.loc[past_rows, col].to_numpy()
        right = out_after.loc[past_rows, col].to_numpy()
        np.testing.assert_array_equal(
            left, right, err_msg=f"feature {col!r} leaked future information"
        )

    # And prove the mutation actually did something to the *future* rows'
    # own downstream state, so this isn't a vacuously-passing no-op test.
    future_rows = df["vp_ts"] >= cutoff
    changed_any = False
    for col in ROLLING_CONDITIONS_FEATURES:
        left = out_before.loc[future_rows, col].to_numpy()
        right = out_after.loc[future_rows, col].to_numpy()
        if not _nan_aware_equal(left, right):
            changed_any = True
    assert changed_any, "mutation had no effect anywhere -- test is not exercising anything"


def _nan_aware_equal(a: np.ndarray, b: np.ndarray) -> bool:
    both_nan = np.isnan(a) & np.isnan(b)
    return bool(np.all(both_nan | (a == b)))


# --------------------------------------------------------------------------
# Boundary: arrival exactly at t excluded, t - eps included.
# --------------------------------------------------------------------------

def test_boundary_exact_arrival_excluded_epsilon_before_included():
    # Segment "S1>S2" completes (Trip A) at +160s. Build two query rows, on a
    # different trip/vehicle, for the SAME segment: one querying at exactly
    # +160s (the traversal completes AT t -- must NOT be visible), one
    # querying one microsecond later (the traversal completed at t - eps --
    # must be visible).
    rows = [
        _row("A", "v1", "R1", 0, "S0", "S1", 0, 60),
        _row("A", "v1", "R1", 1, "S1", "S2", 60, 160),
    ]
    df = pd.DataFrame(rows)

    exact = _row("C", "v3", "R1", 1, "S1", "S2", 999, 999)
    exact["vp_ts"] = BASE + pd.Timedelta(seconds=160)  # == completion instant
    after = _row("C", "v3", "R1", 1, "S1", "S2", 999, 999)
    after["vp_ts"] = BASE + pd.Timedelta(seconds=160) + pd.Timedelta(microseconds=1)

    df = pd.concat([df, pd.DataFrame([exact, after])], ignore_index=True)
    out = add_rolling_conditions_features(df, _cfg(min_observations=1))

    exact_ts = BASE + pd.Timedelta(seconds=160)
    after_ts = exact_ts + pd.Timedelta(microseconds=1)
    exact_row = out[(out["trip_id"] == "C") & (out["vp_ts"] == exact_ts)]
    after_row = out[(out["trip_id"] == "C") & (out["vp_ts"] == after_ts)]

    assert exact_row["leader_last_duration_s"].isna().all(), (
        "a traversal completing at exactly t must be excluded"
    )
    assert after_row["leader_last_duration_s"].notna().all(), (
        "a traversal completing at t - eps (query one us later) must be included"
    )
    assert after_row["leader_last_duration_s"].iloc[0] == 100.0


# --------------------------------------------------------------------------
# Cold start: documented NaN fallback, not a silently-leaked global mean.
# --------------------------------------------------------------------------

def test_cold_start_segment_gets_nan_not_global_mean():
    df = _corpus()
    cfg = _cfg(min_observations=1)
    out = add_rolling_conditions_features(df, cfg)

    # Trip A, segment 0 (S0->S1): the very first traversal of that segment in
    # the whole corpus -- no history exists yet.
    first = out[(out["trip_id"] == "A") & (out["segment_index"] == 0)]
    assert first["recent_seg_traversal_s"].isna().all()
    assert first["recent_seg_ratio"].isna().all()
    assert first["headway_since_last_traversal_s"].isna().all()
    assert first["leader_last_duration_s"].isna().all()

    # Sanity: this is NOT the corpus-wide mean duration sneaking in as a
    # fallback -- there is no numeric value at all, it is exactly NaN.
    assert not (first["recent_seg_traversal_s"] == df["actual_arrival"].count()).any()


def test_ratio_respects_min_observations_guard():
    """With min_observations=2, a segment's SECOND traversal still has only
    one prior observation and must stay cold (NaN ratio), even though
    `recent_seg_traversal_s` (the EWMA) could technically already report a
    number from that one observation."""
    rows = [
        _row("A", "v1", "R1", 0, "S0", "S1", 0, 60),
        _row("A", "v1", "R1", 1, "S1", "S2", 60, 160),   # 1st traversal of S1>S2
        _row("B", "v2", "R1", 0, "S9", "S8", 500, 560),
        _row("B", "v2", "R1", 1, "S1", "S2", 700, 900),  # 2nd traversal, query after
    ]
    df = pd.DataFrame(rows)
    out = add_rolling_conditions_features(df, _cfg(min_observations=2))
    second = out[(out["trip_id"] == "B") & (out["segment_index"] == 1)]
    assert second["recent_seg_ratio"].isna().all(), (
        "ratio must stay cold below min_observations even with 1 prior obs"
    )


# --------------------------------------------------------------------------
# EWMA / expanding arithmetic matches a hand-computed example.
# --------------------------------------------------------------------------

def test_ewma_and_ratio_match_hand_computation():
    """Three sequential traversals of the same segment by three different
    trips: durations 100s, 200s, 300s. A 4th trip's query, occurring after
    all three, must see:
      - recent_seg_traversal_s = EWM(span=2, adjust=False) of [100, 200, 300]
      - recent_seg_ratio = that EWMA / expanding_mean([100, 200, 300])
      - seg_n_obs_before = 3
    """
    rows = [
        _row("A", "v1", "R1", 0, "S0", "S1", 0, 100),
        _row("B", "v2", "R1", 0, "S0", "S1", 200, 400),   # duration 200
        _row("C", "v3", "R1", 0, "S0", "S1", 500, 900),   # duration 400... fix below
    ]
    # Build durations 100, 200, 300 exactly via explicit arrival offsets:
    # trip A: vp 0 -> arrival 100 (dur 100, no prior segment so this is seg0:
    # NaN duration by design). Use segment_index 1 with a real "from" arrival
    # instead, so durations are well-defined and equal to 100/200/300.
    rows = [
        _row("A", "v1", "R1", 0, "SX", "S0", 0, 0),      # seg0: establishes arrival at S0 (t=0)
        _row("A", "v1", "R1", 1, "S0", "S1", 0, 100),    # seg1 duration = 100
        _row("B", "v2", "R1", 0, "SX", "S0", 150, 150),
        _row("B", "v2", "R1", 1, "S0", "S1", 150, 350),  # seg1 duration = 200
        _row("C", "v3", "R1", 0, "SX", "S0", 400, 400),
        _row("C", "v3", "R1", 1, "S0", "S1", 400, 700),  # seg1 duration = 300
        _row("D", "v4", "R1", 0, "SX", "S0", 900, 900),
        _row("D", "v4", "R1", 1, "S0", "S1", 900, 1200),  # query row, after all 3
    ]
    df = pd.DataFrame(rows)
    out = add_rolling_conditions_features(df, _cfg(min_observations=1, seg_ewma_span=2.0))

    d_row = out[(out["trip_id"] == "D") & (out["segment_index"] == 1)].iloc[0]

    expected_ewma = pd.Series([100.0, 200.0, 300.0]).ewm(
        span=2.0, adjust=False, min_periods=1
    ).mean().iloc[-1]
    expected_mean = pd.Series([100.0, 200.0, 300.0]).mean()

    assert d_row["seg_n_obs_before"] == 3
    assert d_row["recent_seg_traversal_s"] == pytest.approx(expected_ewma)
    assert d_row["recent_seg_ratio"] == pytest.approx(expected_ewma / expected_mean)


# --------------------------------------------------------------------------
# Deterministic under reordering of input rows.
# --------------------------------------------------------------------------

def test_deterministic_under_row_reordering():
    df = _corpus()
    cfg = _cfg(min_observations=1)

    out_in_order = add_rolling_conditions_features(df, cfg)

    shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    out_shuffled = add_rolling_conditions_features(shuffled, cfg)

    # Re-align by a natural key present in both (trip_id, segment_index, vp_ts
    # is unique in this fixture) before comparing.
    key = ["trip_id", "segment_index", "vp_ts"]
    a = out_in_order.sort_values(key, kind="mergesort").reset_index(drop=True)
    b = out_shuffled.sort_values(key, kind="mergesort").reset_index(drop=True)

    for col in ROLLING_CONDITIONS_FEATURES:
        np.testing.assert_array_equal(
            a[col].to_numpy(), b[col].to_numpy(),
            err_msg=f"feature {col!r} is not deterministic under row reordering",
        )


# --------------------------------------------------------------------------
# Input validation / config guards.
# --------------------------------------------------------------------------

def test_missing_required_column_raises():
    df = _corpus().drop(columns=["actual_arrival"])
    with pytest.raises(ValueError):
        add_rolling_conditions_features(df, _cfg())


def test_config_rejects_invalid_min_observations():
    with pytest.raises(ValueError):
        RollingConditionsConfig(offset_hours=7, min_observations=0)


def test_config_rejects_invalid_span():
    with pytest.raises(ValueError):
        RollingConditionsConfig(offset_hours=7, seg_ewma_span=0)


def test_does_not_mutate_input():
    df = _corpus()
    original = df.copy(deep=True)
    add_rolling_conditions_features(df, _cfg())
    pd.testing.assert_frame_equal(df, original)
