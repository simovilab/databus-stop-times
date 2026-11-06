"""StoptimesArtifactPredictor on a live-feature artifact: a stateful per-run session fed by predict().

Synthetic feed and a tiny real xgboost artifact (see livekit.py); nothing here reads ground truth.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

pytest.importorskip("xgboost")

from stoptimes.core.domain.models import TripInstance, VehicleState
from stoptimes.val.predictors import artifact as artifact_module
from stoptimes.val.predictors.artifact import StoptimesArtifactPredictor


def trip(trip_id="trip1", vehicle="v1"):
    return TripInstance(
        source="synth", trip_id=trip_id, vehicle_id=vehicle, route_id="R1", direction_id=0,
        shape_id="", start_date="20261014", started_at=0.0,
    )


def vs(p):
    return VehicleState(epoch=p.epoch, lat=p.lat, lon=p.lon)


@pytest.fixture
def make_predictor(kit, tmp_path):
    meta = kit.write_live_registry(tmp_path)

    def build(attach=True):
        predictor = StoptimesArtifactPredictor(meta.artifact_id, registry_dir=tmp_path)
        if attach:
            predictor.attach_static_loader(kit.StubStatic())
        return predictor

    return build


def run(predictor, t, pings, *, history=False):
    """One predict() per ping, as the engine's tick loop does; returns {ping index: predictions}."""
    out = {}
    for i, p in enumerate(pings):
        hist = [vs(q) for q in pings[: i + 1]] if history else None
        out[i] = predictor.predict(t, [], vs(p), None, position_history=hist)
    return out


def as_arrays(preds):
    return np.array([[q.predicted_arrival_epoch, q.stop_sequence, q.dist_remaining_m] for q in preds])


def test_needs_the_runs_static_loader(make_predictor, kit):
    predictor = make_predictor(attach=False)
    with pytest.raises(RuntimeError, match="attach_static_loader"):
        predictor.predict(trip(), [], vs(kit.drive("trip1")[0]), None)


def test_predicts_downstream_stops_with_the_artifacts_locale(make_predictor, kit):
    predictor = make_predictor()
    pings = kit.drive("trip1")
    preds = run(predictor, trip(), pings[:30])[29]
    assert len(preds) >= 3
    assert [q.stop_sequence for q in preds] == sorted(q.stop_sequence for q in preds)
    assert all(q.made_at_epoch == pings[29].epoch for q in preds)
    assert preds[0].stops_remaining == len(preds) and preds[-1].stops_remaining == 1
    assert all(q.dist_remaining_m > 0 for q in preds)
    assert predictor._live[next(iter(predictor._live))][0].tz == "America/New_York"


def test_prediction_at_t_never_depends_on_a_later_ping(make_predictor, kit):
    pings = kit.drive("trip1", seed=3)
    k = 24
    a = run(make_predictor(), trip(), pings[: k + 1])[k]
    mutated = pings[: k + 1] + [type(p)(p.epoch, p.lat + 0.02, p.lon - 0.02) for p in pings[k + 1 : k + 10]]
    b = run(make_predictor(), trip(), mutated)[k]
    np.testing.assert_array_equal(as_arrays(a), as_arrays(b))
    # also within one run: the earlier prediction was made before the later pings existed, and re-asking
    # at the same instant after feeding the future gives the same answer
    predictor = make_predictor()
    run(predictor, trip(), pings[: k + 1])
    run_future = run(predictor, trip(), pings[: k + 12])
    again = predictor.predict(trip(), [], vs(pings[k]), None, now=pings[k].epoch)
    np.testing.assert_array_equal(as_arrays(again), as_arrays(a))
    assert len(run_future[k + 11]) > 0


def test_history_and_per_call_feeding_agree(make_predictor, kit):
    pings = kit.drive("trip1", seed=5)[:30]
    one = run(make_predictor(), trip(), pings)[29]
    hist = run(make_predictor(), trip(), pings, history=True)[29]
    np.testing.assert_array_equal(as_arrays(one), as_arrays(hist))


def test_a_ping_skipped_between_ticks_still_reaches_the_state(make_predictor, kit):
    pings = kit.drive("trip1", seed=6)[:30]
    every = run(make_predictor(), trip(), pings)[29]
    predictor = make_predictor()
    for i in (0, 9, 29):  # the engine only saw these ticks, but handed the whole history
        out = predictor.predict(trip(), [], vs(pings[i]), None, position_history=[vs(q) for q in pings[: i + 1]])
    np.testing.assert_array_equal(as_arrays(out), as_arrays(every))


def test_cold_start_mid_trip_predicts_ahead_only(make_predictor, kit):
    pings = kit.drive("trip1", start_m=3000.0, seed=7)
    preds = run(make_predictor(), trip(), pings[:5])[4]
    assert preds and min(q.stop_sequence for q in preds) >= 7  # s6 is at 3000 m, already behind
    assert all(np.isfinite(q.predicted_arrival_epoch) for q in preds)


def test_state_is_isolated_between_trips(make_predictor, kit):
    p1 = kit.drive("trip1", seed=1)[:30]
    p2 = kit.drive("trip2", seed=2, delay_s=-1200)[:30]
    solo = run(make_predictor(), trip("trip1", "v1"), p1)[29]
    mixed = make_predictor()
    last = None
    for a, b in zip(p1, p2):  # interleaved, as one engine tick feeds many trips
        last = mixed.predict(trip("trip1", "v1"), [], vs(a), None)
        mixed.predict(trip("trip2", "v2"), [], vs(b), None)
    assert last[0].predicted_arrival_epoch == pytest.approx(solo[0].predicted_arrival_epoch)
    np.testing.assert_allclose(as_arrays(last)[:, 1:], as_arrays(solo)[:, 1:])


def test_a_trip_that_went_quiet_gets_no_prediction_once_the_clock_moves_on(make_predictor, kit):
    predictor = make_predictor()
    p1 = kit.drive("trip1", seed=1)[:20]
    p2 = kit.drive("trip2", seed=2, delay_s=-1200)
    run(predictor, trip("trip1", "v1"), p1)
    late = [p for p in p2 if p.epoch > p1[-1].epoch + 40][:2]  # another bus pings 40+ s later
    for p in late:
        predictor.predict(trip("trip2", "v2"), [], vs(p), None)
    assert predictor.predict(trip("trip1", "v1"), [], vs(p1[-1]), None) == []  # trip1's last ping is stale


def test_unknown_trip_yields_nothing(make_predictor, kit):
    p = kit.drive("trip1")[0]
    assert make_predictor().predict(trip("ghost", "v9"), [], vs(p), None) == []


def test_the_predictor_never_touches_ground_truth_modules():
    source = inspect.getsource(artifact_module)
    imports = [line for line in source.splitlines() if line.startswith(("from ", "import "))]
    assert not any("detector" in line or "groundtruth" in line or "arrival" in line for line in imports)
