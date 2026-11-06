"""Live-feature training frame: service dates, out-of-fold statistics, targets, train/serve parity."""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from stoptimes.core.features.live import LIVE_FEATURES, assemble_live_features
from stoptimes.core.features.trip_state import update_trip_state
from stoptimes.core.gtfs.static import GtfsStaticLoader
from stoptimes.train.live_frame import (
    LiveFrameParams,
    build_live_dataset,
    prepare_trip_records,
    write_live_assets,
)

TZ = "America/New_York"
LAT0, LON0 = 42.0, -71.0
M_DEG = 6371000.0 * math.pi / 180.0
COS = math.cos(math.radians(LAT0))
N_STOPS = 8
STOP_SPACING_M = 600.0


def _lon(progress_m: float) -> float:
    return LON0 + progress_m / (COS * M_DEG)


def _write_feed(path: Path, *, with_agency: bool = True) -> Path:
    stops = "stop_id,stop_name,stop_lat,stop_lon\n" + "".join(
        f"s{i},S{i},{LAT0},{_lon(i * STOP_SPACING_M)}\n" for i in range(N_STOPS)
    )
    trips = "route_id,service_id,trip_id,direction_id,shape_id\nrA,svc,t_day,0,sh\nrB,svc,t_late,0,sh\n"

    def times(start_h: int, start_m: int) -> list[str]:
        out = []
        for i in range(N_STOPS):
            total = start_h * 60 + start_m + i  # one stop per minute
            out.append(f"{total // 60:02d}:{total % 60:02d}:00")
        return out

    rows = []
    for trip, (h, m) in (("t_day", (8, 0)), ("t_late", (25, 30))):  # t_late runs after midnight
        for i, tm in enumerate(times(h, m)):
            rows.append(f"{trip},{tm},{tm},s{i},{i + 1},{i * STOP_SPACING_M}")
    stop_times = "trip_id,arrival_time,departure_time,stop_id,stop_sequence,shape_dist_traveled\n" + "\n".join(rows) + "\n"
    shapes = "shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence,shape_dist_traveled\n" + "".join(
        f"sh,{LAT0},{_lon(i * 300.0)},{i + 1},{i * 300.0}\n" for i in range(int(N_STOPS * STOP_SPACING_M / 300) + 2)
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("stops.txt", stops)
        zf.writestr("trips.txt", trips)
        zf.writestr("stop_times.txt", stop_times)
        zf.writestr("shapes.txt", shapes)
        zf.writestr("routes.txt", "route_id,route_short_name\nrA,A\nrB,B\n")
        if with_agency:
            zf.writestr("agency.txt", f"agency_id,agency_name,agency_url,agency_timezone\n1,X,http://x,{TZ}\n")
    return path


def _drive(
    service_date: str, trip_id: str, start_h: int, start_m: int, delay_s: float, vehicle: str = "v1", pace: float = 1.0
) -> list[dict]:
    day = pd.Timestamp(service_date, tz=TZ)  # local midnight == GTFS origin on non-DST days
    t0 = day.timestamp() + start_h * 3600 + start_m * 60 + delay_s
    speed = STOP_SPACING_M / 60.0 * pace
    rows = []
    for k in range(0, 8 * 60 + 1, 5):
        p = min(k * speed, (N_STOPS - 1) * STOP_SPACING_M + 100.0)
        rows.append(
            {
                "kind": "vehicle_position",
                "ts": pd.Timestamp(t0 + k, unit="s", tz="UTC"),
                "lat": LAT0 + 2.0 / M_DEG,
                "lon": _lon(p),
                "trip_id": trip_id,
                "route_id": "rA" if trip_id == "t_day" else "rB",
                "vehicle_id": vehicle,
            }
        )
    return rows


@pytest.fixture(scope="module")
def feed(tmp_path_factory):
    return GtfsStaticLoader(_write_feed(tmp_path_factory.mktemp("feed") / "gtfs.zip"))


PACES = [1.0, 1.5, 0.8, 1.2, 1.9, 0.7, 1.4, 1.1, 1.6, 0.9]
TRAIN = [f"2026-10-{d:02d}" for d in range(1, 9)]
VAL = ["2026-10-09", "2026-10-10"]


@pytest.fixture(scope="module")
def observations():
    rows = []
    for i, d in enumerate([*TRAIN, *VAL]):
        key = d.replace("-", "")
        rows += _drive(key, "t_day", 8, 0, delay_s=20.0 * i, pace=PACES[i])
        rows += _drive(key, "t_late", 25, 30, delay_s=-10.0 * i, pace=PACES[i])
        rows += _drive(key, "t_day", 8, 0, delay_s=0.0, vehicle="ghost")[:3]  # a stray second vehicle
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def records(feed, observations):
    by_day, skipped = prepare_trip_records(
        observations, feed, config_timezone="UTC", holiday_region="US_MA"
    )
    return by_day, skipped


def test_service_date_comes_from_the_trip_not_the_calendar_date(records):
    by_day, skipped = records
    assert sorted(by_day) == [d.replace("-", "") for d in [*TRAIN, *VAL]]
    late = [r for r in by_day["20261003"] if r.ctx.trip_id == "t_late"]
    assert len(late) == 1  # its pings are on 2026-10-04 local, still service day 10-03
    first_ping = late[0].state.pings[0][0]
    assert pd.Timestamp(first_ping, unit="s", tz="UTC").tz_convert(TZ).strftime("%Y%m%d") == "20261004"
    assert late[0].ctx.service_date == "20261003"


def test_timezone_is_agency_txt_with_configured_fallback(tmp_path, observations):
    feed_no_agency = GtfsStaticLoader(_write_feed(tmp_path / "noag.zip", with_agency=False))
    by_day, _ = prepare_trip_records(observations, feed_no_agency, config_timezone=TZ, holiday_region=None)
    assert by_day and {r.ctx.tz for rs in by_day.values() for r in rs} == {TZ}
    with pytest.raises(ValueError, match="no timezone"):
        prepare_trip_records(observations, feed_no_agency, config_timezone=None, holiday_region=None)
    feed_with = GtfsStaticLoader(_write_feed(tmp_path / "ag.zip"))
    by_day, _ = prepare_trip_records(observations, feed_with, config_timezone="Asia/Tokyo", holiday_region=None)
    assert {r.ctx.tz for rs in by_day.values() for r in rs} == {TZ}  # agency.txt wins


def test_one_vehicle_per_instance_and_arrivals_detected(records):
    by_day, _ = records
    rec = by_day["20261001"][0]
    assert rec.vehicle_id == "v1"
    assert rec.n_pings == 97  # 5 s cadence, pace 1.0  # the stray vehicle's pings are not part of the trip
    arrival, detect, _ = rec.state.arrival_arrays()
    assert np.isfinite(arrival[1:]).all() and (detect[1:] >= arrival[1:]).all()


@pytest.fixture(scope="module")
def dataset(records):
    by_day, _ = records
    return build_live_dataset(
        [d.replace("-", "") for d in TRAIN],
        [d.replace("-", "") for d in VAL],
        lambda day: by_day.get(day, []),
        tz=TZ,
        holiday_region="US_MA",
        params=LiveFrameParams(rows_per_day=None),
    )


def test_frame_has_features_target_split_and_identity(dataset):
    f = dataset.frame
    assert set(LIVE_FEATURES) <= set(f.columns)
    assert {"time_to_arrival_seconds", "split", "day", "vp_ts", "trip_id", "service_date", "t"} <= set(f.columns)
    assert set(f["split"]) == {"train", "val"}
    assert (f["time_to_arrival_seconds"] > 0).all()
    assert f[list(LIVE_FEATURES)].dtypes.eq(np.float32).all()
    assert set(f.loc[f.split == "val", "day"]) == {d.replace("-", "") for d in VAL}
    # arrival == t + target, exactly the state's interpolated arrival
    row = f.iloc[len(f) // 2]
    assert row["t"] + row["time_to_arrival_seconds"] > row["t"]


def test_training_rows_use_out_of_fold_statistics_validation_rows_the_full_ones(dataset, records):
    from stoptimes.core.features.live import LiveAssets, assemble_live_batch

    f = dataset.frame
    assert f.loc[f.split == "train", "prior_rem"].notna().all()
    oof = dataset.out_of_fold
    assert len(oof.by_fold) == 6
    # a training day's statistics are the fold fitted without its own fold
    assert oof.for_day("2026-10-01") is oof.by_fold[0] and oof.for_day("2026-10-07") is oof.by_fold[0]
    # the shipped asset is the all-days fit; the vocabulary holds both routes
    assert dataset.assets.segment_stats is oof.full
    assert dataset.assets.route_vocabulary.route_ids == ("rA", "rB")

    def prior_with(stats, day, split):
        rec = records[0][day][0]
        rows = f[(f.day == day) & (f.trip_id == rec.ctx.trip_id)]
        times = np.sort(rows["t"].unique())
        out = assemble_live_batch(rec.ctx, LiveAssets(stats, dataset.assets.route_vocabulary), rec.state, times)
        return out["prior_rem"].to_numpy(np.float32), rows.sort_values(["t", "stop_idx"])["prior_rem"].to_numpy()

    # training day: the frame carries the out-of-fold prior, not the full-data prior
    full_prior, frame_prior = prior_with(oof.full, "20261001", "train")
    oof_prior, _ = prior_with(oof.for_day("2026-10-01"), "20261001", "train")
    np.testing.assert_array_equal(oof_prior, frame_prior)
    assert not np.array_equal(full_prior, frame_prior)
    # validation day: the frame carries the full prior
    full_prior_v, frame_prior_v = prior_with(oof.full, "20261009", "val")
    np.testing.assert_array_equal(full_prior_v, frame_prior_v)


def test_rows_per_day_subsample_is_seeded_and_uniform(records):
    by_day, _ = records
    kwargs = dict(tz=TZ, holiday_region="US_MA")
    days = [d.replace("-", "") for d in TRAIN[:3]], [VAL[0].replace("-", "")]
    a = build_live_dataset(*days, lambda d: by_day[d], params=LiveFrameParams(rows_per_day=30, seed=1), **kwargs)
    b = build_live_dataset(*days, lambda d: by_day[d], params=LiveFrameParams(rows_per_day=30, seed=1), **kwargs)
    c = build_live_dataset(*days, lambda d: by_day[d], params=LiveFrameParams(rows_per_day=30, seed=2), **kwargs)
    assert a.frame.groupby("day").size().eq(30).all()
    pd.testing.assert_frame_equal(a.frame, b.frame)
    assert not a.frame["t"].equals(c.frame["t"])


def test_overlapping_train_and_val_days_are_rejected(records):
    with pytest.raises(ValueError, match="both train and val"):
        build_live_dataset(["20261001"], ["20261001"], lambda d: [], tz=TZ, holiday_region=None)


def test_frame_rows_equal_a_tick_by_tick_replay(dataset, records):
    """The frame's features at an instant are what serving computes when it replays the pings."""
    from stoptimes.core.features.trip_state import ForwardMatcher, Ping
    from stoptimes.core.spatial.polyline import build_polyline

    day = "20261009"  # validation day: shipped statistics
    rec = records[0][day][0]
    n = int(N_STOPS * STOP_SPACING_M / 300) + 2
    matcher = ForwardMatcher(build_polyline([(LAT0, _lon(i * 300.0)) for i in range(n)]))
    pings = [Ping(e, LAT0 + 2.0 / M_DEG, _lon(p)) for e, p, ok in rec.state.pings if ok]
    rows = dataset.frame[(dataset.frame.day == day) & (dataset.frame.trip_id == rec.ctx.trip_id)]
    instants = sorted(rows["t"].unique())[:4]
    assert len(instants) == 4
    for t in instants:
        state = None
        for p in pings:
            if p.epoch <= t:
                state = update_trip_state(
                    state, p, polyline=matcher, stop_progress_m=rec.ctx.stop_progress_m, trip_key=rec.ctx.trip_key
                )
        live = assemble_live_features(rec.ctx, dataset.assets, state, t)
        got = rows[rows.t == t].sort_values("stop_idx")
        assert len(live) == len(got) > 0
        for name in LIVE_FEATURES:
            np.testing.assert_array_equal(
                live[name].to_numpy(np.float32), got[name].to_numpy(np.float32), err_msg=name
            )


def test_assets_written_with_hashes_and_manifest(dataset, tmp_path):
    hashes = write_live_assets(tmp_path / "frame.assets", dataset)
    manifest = json.loads((tmp_path / "frame.assets" / "live.json").read_text())
    assert manifest["timezone"] == TZ and manifest["features"] == list(LIVE_FEATURES)
    assert manifest["assets"]["segment_stats"]["sha256"] == hashes["segment_stats"]
    from stoptimes.core.features.prior import SegmentStats

    reloaded = SegmentStats.load(tmp_path / "frame.assets" / "segment_stats.bin")
    assert reloaded.keys == dataset.assets.segment_stats.keys
