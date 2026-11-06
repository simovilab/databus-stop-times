import numpy as np
import pandas as pd
import pytest

from stoptimes.core.features import prior as P
from stoptimes.core.features.prior import (
    SegmentStats, daytype_of, fit, fit_out_of_fold, local_offset_s, prior_remaining, segments_from_arrivals,
)

TZ = "America/New_York"


def seg_rows(n, tt, *, route="R", frm="A", to="B", hour=8, dtype=0, day="2026-01-05", seglen=500.0):
    return pd.DataFrame({
        "route_id": route, "from_stop_id": frm, "to_stop_id": to, "travel_s": float(tt) if np.isscalar(tt) else tt,
        "seglen_m": seglen, "hour": hour, "daytype": dtype, "service_date": day,
    }, index=range(n))


def look(st, hour=8, dtype=0, frm="A", to="B", route="R", seglen=500.0):
    return float(st.lookup(route, frm, to, dtype, hour, seglen))


# ------------------------------------------------------------------ segments_from_arrivals

def arrivals(day, epoch0, trip="t1", stops=("A", "B", "C"), step=100.0, prog_step=400.0):
    return pd.DataFrame({
        "trip_key": trip, "service_date": day, "route_id": "R", "stop_index": range(len(stops)), "stop_id": list(stops),
        "stop_progress_m": [i * prog_step for i in range(len(stops))], "arrival_epoch": [epoch0 + i * step for i in range(len(stops))],
    })


def epoch(local, tz=TZ):
    return pd.Timestamp(local, tz=tz).timestamp()


def test_segments_basic_filters_and_columns():
    a = arrivals("2026-01-05", epoch("2026-01-05 08:30"))
    a.loc[2, "arrival_epoch"] = np.nan
    b = arrivals("2026-01-05", epoch("2026-01-05 09:00"), trip="t2", step=2000.0)  # > 1800 dropped
    c = arrivals("2026-01-05", epoch("2026-01-05 10:00"), trip="t3", step=0.0)  # zero dropped
    s = segments_from_arrivals(pd.concat([a, b, c]), TZ)
    assert list(s.columns) == list(P.SEGMENT_COLUMNS)
    assert len(s) == 1 and s.iloc[0].from_stop_id == "A" and s.iloc[0].travel_s == 100.0
    assert s.iloc[0].seglen_m == 400.0 and s.iloc[0].hour == 8 and s.iloc[0].daytype == 0


def test_segments_require_consecutive_index_and_same_trip():
    a = arrivals("2026-01-05", epoch("2026-01-05 08:30"), stops=("A", "B", "C", "D"))
    a = a[a.stop_index != 1]  # gap: no A->C segment
    s = segments_from_arrivals(a, TZ)
    assert list(zip(s.from_stop_id, s.to_stop_id)) == [("C", "D")]


def test_daytype_and_hour_non_dst_and_dst_days():
    sat = arrivals("2026-01-10", epoch("2026-01-10 23:59"))
    sun = arrivals("2026-01-11", epoch("2026-01-11 00:10"), trip="t2")
    s = segments_from_arrivals(pd.concat([sat, sun]), TZ).set_index("service_date")
    assert s.loc["2026-01-10", "daytype"].iloc[0] == 1 and s.loc["2026-01-10", "hour"].iloc[0] == 23
    assert s.loc["2026-01-11", "daytype"].iloc[0] == 2 and s.loc["2026-01-11", "hour"].iloc[0] == 0
    # DST spring-forward Sunday 2026-03-08: offset is the post-change one for the whole service day
    assert local_offset_s("2026-03-08", TZ) == -4 * 3600 and local_offset_s("2026-03-07", TZ) == -5 * 3600
    dst = arrivals("2026-03-08", epoch("2026-03-08 15:30"))
    assert segments_from_arrivals(dst, TZ).hour.iloc[0] == 15
    # Same instants, a tz without DST (and a different offset): hour follows the parameter
    kolkata = arrivals("2026-03-08", epoch("2026-03-08 15:30", "Asia/Kolkata"))
    assert segments_from_arrivals(kolkata, "Asia/Kolkata").hour.iloc[0] == 15
    assert segments_from_arrivals(kolkata, "UTC").hour.iloc[0] == 10


def test_daytype_of_accepts_gtfs_and_iso_dates():
    assert daytype_of("20260110") == 1 and daytype_of("2026-01-12") == 0


# ------------------------------------------------------------------ fit fallback levels

def test_level0_exact_cell_needs_5():
    st = fit(pd.concat([seg_rows(5, 100.0), seg_rows(20, 300.0, hour=9)], ignore_index=True))
    assert look(st, hour=8) == 100.0 and look(st, hour=9) == 300.0


def test_level1_hour_window_when_cell_too_thin():
    # hour 8 has 4 (<5), neighbours 7 and 9 add 4 each -> window of 12 >= 8
    d = pd.concat([seg_rows(4, 100.0, hour=7), seg_rows(4, 200.0, hour=8), seg_rows(4, 300.0, hour=9)], ignore_index=True)
    st = fit(d)
    assert look(st, hour=8) == 200.0  # median of window 12 values: [100x4, 200x4, 300x4]
    assert look(st, hour=7) == 150.0  # window hours 6..8: 100x4 + 200x4


def test_level1b_ignores_daytype():
    d = pd.concat([seg_rows(4, 100.0, dtype=0), seg_rows(4, 120.0, dtype=1)], ignore_index=True)
    st = fit(d)
    assert look(st, dtype=2, hour=8) == 110.0  # unseen daytype borrowed from (seg, hour+-1) pooled
    assert look(st, dtype=0, hour=8) == 110.0  # dtype 0 window n=4 <8 so pooled too


def test_level2_segment_daytype_and_level3_segment():
    d = pd.concat([seg_rows(3, 100.0, hour=2, dtype=0), seg_rows(1, 500.0, hour=20, dtype=0), seg_rows(1, 900.0, hour=12, dtype=1)], ignore_index=True)
    st = fit(d)
    assert look(st, hour=14, dtype=0) == 100.0   # hour 14 has no data: (seg, daytype=0) median of 4 rows is 100
    # (seg,daytype 1) has n=1 < 3 -> falls to overall segment median
    assert look(st, hour=14, dtype=1) == float(np.median([100, 100, 100, 500, 900]))
    assert look(st, hour=14, dtype=2) == float(np.median([100, 100, 100, 500, 900]))


def test_level4_length_over_route_speed_and_unseen_route_default():
    d = pd.concat([seg_rows(10, 100.0, seglen=500.0), seg_rows(10, 100.0, route="Q", frm="X", to="Y", seglen=1000.0)], ignore_index=True)
    st = fit(d)
    # route R speed 5 m/s, Q 10 m/s, default = median over all rows = 7.5
    assert st.route_speed == {"R": 5.0, "Q": 10.0} and st.speed_default == 7.5
    assert look(st, frm="A", to="Z", seglen=1000.0) == 1000.0 / 5.0 + 15.0
    assert look(st, route="Q", frm="X", to="Z", seglen=2000.0) == 2000.0 / 10.0 + 15.0
    assert look(st, route="N", frm="A", to="B", seglen=15.0) == 30.0 / 7.5 + 15.0  # min length 30, default speed


def test_empty_fit_uses_global_defaults():
    st = fit(seg_rows(0, 1.0))
    assert look(st) == 500.0 / 5.0 + 15.0


def test_lookup_vectorised_and_broadcast():
    st = fit(seg_rows(10, 100.0))
    out = st.lookup("R", ["A", "A", "Q"], ["B", "Z", "B"], 0, [8, 9, 8], 500.0)
    assert out.shape == (3,)
    assert out[0] == 100.0 and out[1] == 500.0 / 5.0 + 15.0 and out[2] == 500.0 / 5.0 + 15.0


# ------------------------------------------------------------------ out of fold

def test_out_of_fold_excludes_own_fold():
    days = [f"2026-01-{d:02d}" for d in range(5, 11)]  # 6 weekdays/sat, k=3 -> 2 days per fold
    rows = [seg_rows(10, 100.0 * (i + 1), day=d, dtype=0) for i, d in enumerate(days)]
    segs = pd.concat(rows, ignore_index=True)
    oof = fit_out_of_fold(segs, days, k=3)
    assert dict(oof.day_fold) == {d: i % 3 for i, d in enumerate(days)}
    # fold 0 = days 0 and 3 (tt 100, 400); fitted on the rest: 200,300,500,600 -> median 400
    assert look(oof.by_fold[0]) == float(np.median([200, 300, 500, 600]))
    assert look(oof.for_day(days[3])) == look(oof.by_fold[0])
    assert look(oof.full) == float(np.median([100, 200, 300, 400, 500, 600]))
    assert oof.by_fold[0].n_obs == 40 and oof.full.n_obs == 60


def test_out_of_fold_ignores_days_not_listed():
    segs = pd.concat([seg_rows(10, 100.0, day="2026-01-05"), seg_rows(10, 9999.0, day="2026-01-06")], ignore_index=True)
    oof = fit_out_of_fold(segs, ["2026-01-05"], k=2)
    assert look(oof.full) == 100.0 and oof.by_fold[0].n_obs == 0 and oof.by_fold[1].n_obs == 10


def test_fold_map_uses_given_order_and_is_immutable():
    oof = fit_out_of_fold(seg_rows(10, 1.0), ["2026-01-05", "2026-01-05", "2026-01-06"], k=6)
    assert dict(oof.day_fold) == {"2026-01-05": 0, "2026-01-06": 1}
    with pytest.raises(TypeError):
        oof.day_fold["x"] = 3  # type: ignore[index]


# ------------------------------------------------------------------ persistence

def _stats():
    rng = np.random.default_rng(0)
    parts = [seg_rows(30, rng.uniform(60, 300, 30), frm=f, to=t, hour=int(h), dtype=int(d), seglen=400.0 + 10 * i)
             for i, (f, t) in enumerate([("A", "B"), ("B", "C"), ("C", "D")]) for h in (7, 8, 17) for d in (0, 1)]
    return fit(pd.concat(parts, ignore_index=True))


def test_save_load_bit_identical_and_stable_hash(tmp_path):
    st = _stats()
    h1 = st.save(tmp_path / "a.bin")
    h2 = st.save(tmp_path / "b.bin")
    assert h1 == h2 and (tmp_path / "a.bin").read_bytes() == (tmp_path / "b.bin").read_bytes()
    import hashlib
    assert h1 == hashlib.sha256((tmp_path / "a.bin").read_bytes()).hexdigest()
    ld = SegmentStats.load(tmp_path / "a.bin")
    q = dict(route_id="R", from_stop_id=np.array(["A", "B", "C", "Z"]), to_stop_id=np.array(["B", "C", "D", "Q"]), daytype=1, hour=np.array([7, 8, 17, 3]), seglen_m=450.0)
    assert np.array_equal(st.lookup(**q), ld.lookup(**q))
    assert ld.save(tmp_path / "c.bin") == h1
    assert np.array_equal(st.table, ld.table) and st.keys == ld.keys and dict(st.route_speed) == dict(ld.route_speed)


def test_load_rejects_bad_version_magic_and_truncation(tmp_path):
    st = _stats()
    raw = st.to_bytes()
    with pytest.raises(ValueError, match="magic"):
        SegmentStats.from_bytes(b"nope" + raw)
    bumped = raw.replace(b'"version":1', b'"version":9')
    with pytest.raises(ValueError, match="version"):
        SegmentStats.from_bytes(bumped)
    with pytest.raises(ValueError, match="truncated"):
        SegmentStats.from_bytes(raw[:-4])


def test_stats_are_immutable():
    st = _stats()
    with pytest.raises(Exception):
        st.n_obs = 1  # type: ignore[misc]
    with pytest.raises(ValueError):
        st.table[0, 0, 0] = 1.0
    with pytest.raises(TypeError):
        st.route_speed["R"] = 1.0  # type: ignore[index]


def test_ids_with_odd_characters_survive_round_trip(tmp_path):
    st = fit(seg_rows(10, 77.0, route="CR-Fitchburg", frm="place-|x", to="Ünï 1"))
    st.save(tmp_path / "s.bin")
    ld = SegmentStats.load(tmp_path / "s.bin")
    assert float(ld.lookup("CR-Fitchburg", "place-|x", "Ünï 1", 0, 8, 500.0)) == 77.0


# ------------------------------------------------------------------ prior_remaining

def _loop_reference(stats, route, stop_ids, sprog, prog, t, tz, day, targets):
    """Straight transcription of the study recursion for one instant."""
    off = local_offset_s(day, tz)
    dtp = daytype_of(day)
    n = len(stop_ids)
    j0 = int(np.searchsorted(np.asarray(sprog), prog, side="right"))
    s = j0 - 1

    def h(k, when):
        hr = int(np.floor((when + off) / 3600.0)) % 24
        return float(stats.lookup(route, stop_ids[k], stop_ids[k + 1], dtp, hr, sprog[k + 1] - sprog[k]))

    cur = 0.0
    if s >= 0:
        frac = min(max((prog - sprog[s]) / max(sprog[s + 1] - sprog[s], 1.0), 0.0), 1.0)
        cur = (1 - frac) * h(s, t)
    eta = {j0: cur}
    for j in range(j0 + 1, n):
        eta[j] = eta[j - 1] + h(j - 1, t + eta[j - 1])
    return np.array([eta.get(j, np.nan) for j in targets])


def _trip_stats():
    # 5 stops A..E; segment times differ by hour so rollover is visible: hour 8 -> 900 s, hour 9 -> 100 s
    parts = []
    for (f, to) in [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]:
        parts.append(seg_rows(10, 900.0, frm=f, to=to, hour=8))
        parts.append(seg_rows(10, 100.0, frm=f, to=to, hour=9))
    return fit(pd.concat(parts, ignore_index=True))


STOPS = ["A", "B", "C", "D", "E"]
SPROG = [0.0, 500.0, 1100.0, 1600.0, 2400.0]


@pytest.mark.parametrize("prog", [-20.0, 0.0, 120.0, 500.0, 777.0, 1599.0, 2000.0])
def test_prior_remaining_matches_reference_loop(prog):
    st = _trip_stats()
    day = "2026-01-05"
    t = epoch("2026-01-05 08:50")  # 900 s hops cross into hour 9 mid path
    targets = [0, 1, 2, 3, 4]
    got = prior_remaining(st, "R", STOPS, SPROG, prog, t, TZ, day, targets)
    want = _loop_reference(st, "R", STOPS, SPROG, prog, t, TZ, day, targets)
    np.testing.assert_allclose(got, want, equal_nan=True)


def test_prior_remaining_hour_rollover_and_semantics():
    st = _trip_stats()
    t = epoch("2026-01-05 08:50")
    # at stop A exactly (progress 0 -> j0=1, s=0, frac=0): cur = H(A->B @ 08:50) = 900
    got = prior_remaining(st, "R", STOPS, SPROG, 0.0, t, TZ, "2026-01-05", [0, 1, 2, 3])
    assert np.isnan(got[0])
    assert got[1] == 900.0
    assert got[2] == 900.0 + 100.0   # t+900 = 09:05 -> hour 9
    assert got[3] == 1000.0 + 100.0


def test_prior_remaining_past_last_stop_and_out_of_range_targets():
    st = _trip_stats()
    assert np.isnan(prior_remaining(st, "R", STOPS, SPROG, 2400.0, 0.0, TZ, "2026-01-05", [4])).all()
    got = prior_remaining(st, "R", STOPS, SPROG, 100.0, epoch("2026-01-05 08:00"), TZ, "2026-01-05", [9, -1, 2])
    assert np.isnan(got[:2]).all() and np.isfinite(got[2])


def test_prior_remaining_respects_timezone_param():
    st = _trip_stats()
    t = epoch("2026-01-05 08:50")
    a = prior_remaining(st, "R", STOPS, SPROG, 0.0, t, TZ, "2026-01-05", [1])
    b = prior_remaining(st, "R", STOPS, SPROG, 0.0, t, "UTC", "2026-01-05", [1])  # 13:50 UTC -> no hour-8 entry: fallback
    assert a[0] == 900.0 and b[0] != 900.0
