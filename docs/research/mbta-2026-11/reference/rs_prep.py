"""Phase 2c prep: per-trip matched pings -> (a) per-stop arrival EVENTS with their detection time, (b) per-instant kinematics.

Everything here is a pure function of (raw VP frame, GT stop table); the perturb-the-future test (2c_test_causality.py) feeds it a
VP frame whose rows after t are rewritten and asserts the instant-level features at t do not move.

Event definition (causality rule of the task): the arrival of trip Y at stop i is  a = interp over the bracketing valid pings
(k-1, k) with prog[k-1] < S_i <= prog[k]  (identical to the ground truth arr_interp, same gap<=60 s rule). It is *detectable* at
detect = ts[k] (the first ping after the crossing).  A consumer at time t may use the event only if detect <= t.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from common import E2E
from gt_core import match_pings, interp_arrivals

RS = f"{E2E}/models/research"
STATUS_CODE = {"INCOMING_AT": 0, "STOPPED_AT": 1, "IN_TRANSIT_TO": 2}


def load_shapes(tag, shape_ids, cache):
    from gtfs_util import load_bucr_gtfs
    from stoptimes.train.sources.bucr_gtfs import build_shape_polyline
    if "g" not in cache:
        cache["g"] = load_bucr_gtfs(tag)
    out = {}
    for s in shape_ids:
        if s not in cache:
            poly = build_shape_polyline(cache["g"], s)
            cache[s] = (poly, np.array([p.cum_m for p in poly]))
        out[s] = cache[s][0]
    return out


def trip_arrays(vp_by_trip, gt_trip, poly):
    """Matched pings of the GT-chosen vehicle for one trip. Returns dict(ts, valid, prog, status, vspeed) or None."""
    veh = gt_trip.vehicle_id.iloc[0]
    tv = vp_by_trip
    pv = tv[tv.vehicle_id == veh].sort_values("ts", kind="stable").drop_duplicates("ts")
    if len(pv) < 2:
        return None
    ts = pv.ts.to_numpy(float)
    valid, prog, xt = match_pings(ts, pv.lat.to_numpy(float), pv.lon.to_numpy(float), poly)
    return dict(ts=ts, valid=valid, prog=prog, xt=xt, status=pv.current_status.map(STATUS_CODE).to_numpy(float),
                vspeed=pd.to_numeric(pv.speed, errors="coerce").to_numpy(float))


def arrivals_with_detect(arr, sprog):
    """arr = trip_arrays dict, sprog = stop progress (m). Returns (a, detect, gap) per stop (NaN where GT rule rejects)."""
    v = arr["valid"]
    vts, vprog = arr["ts"][v], arr["prog"][v]
    a, gap, reason = interp_arrivals(vts, vprog, sprog)
    idx = np.searchsorted(vprog, sprog, side="left")
    ok = np.isfinite(a)
    detect = np.full(len(sprog), np.nan)
    detect[ok] = vts[np.clip(idx[ok], 0, len(vts) - 1)]
    return a, detect, gap


def kinematics(arr, ping_idx):
    """Causal kinematics at ping indices `ping_idx` (indices into arr['ts']); every quantity uses valid pings <= that ping."""
    v = arr["valid"]; vts, vprog = arr["ts"][v], arr["prog"][v]
    vspeed = arr["vspeed"][v]; vstat = arr["status"][v]
    cv = np.cumsum(v) - 1
    vk = cv[ping_idx]
    out = {}
    for W in (30, 60, 120):
        j = np.maximum(np.searchsorted(vts, vts[vk] - W, side="right") - 1, 0)
        dtm = vts[vk] - vts[j]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"spd{W}"] = np.where(dtm >= 10, (vprog[vk] - vprog[j]) / dtm, np.nan)
    m = np.searchsorted(vprog, vprog[vk] - 25.0, side="left")
    out["stopped_s"] = np.minimum(vts[vk] - vts[m], 3600.0)
    j5 = np.maximum(np.searchsorted(vts, vts[vk] - 300, side="right") - 1, 0)
    out["dist300"] = vprog[vk] - vprog[j5]
    out["vp_speed"] = vspeed[vk]; out["status_code"] = vstat[vk]
    out["trip_age_s"] = vts[vk] - arr["ts"][0]
    return out


def build_trip_tables(vp, gt, insts, shapes_by_id):
    """vp: raw pings (trip_id, vehicle_id, ts, lat, lon, current_status, speed). gt: GT rows of the day. insts: instants of the day.
    Returns (events DataFrame one row per (trip, stop), kin DataFrame one row per instant, check dict)."""
    vp_by = {k: v for k, v in vp.groupby("trip_id", sort=False)}
    ev_parts, kin_parts = [], []
    chk = dict(n_trips=0, n_stop_cmp=0, n_mismatch=0, max_abs_diff=0.0)
    inst_by = {k: v for k, v in insts.groupby("trip_id", sort=False)}
    for trip_id, G in gt.groupby("trip_id", sort=False):
        if trip_id not in vp_by:
            continue
        G = G.sort_values("stop_sequence")
        shape_id = G.shape_id.iloc[0]
        if shape_id not in shapes_by_id:
            continue
        arr = trip_arrays(vp_by[trip_id], G, shapes_by_id[shape_id])
        if arr is None or arr["valid"].sum() < 2:
            continue
        sprog = G.stop_progress_m.to_numpy(float)
        a, detect, gap = arrivals_with_detect(arr, sprog)
        g_ref = G.arr_interp.to_numpy(float)
        both = np.isfinite(a) & np.isfinite(g_ref)
        chk["n_trips"] += 1; chk["n_stop_cmp"] += int(both.sum())
        if both.any():
            d = np.abs(a[both] - g_ref[both]); chk["n_mismatch"] += int((d > 1e-3).sum()); chk["max_abs_diff"] = max(chk["max_abs_diff"], float(d.max()))
        n = len(G)
        ev_parts.append(pd.DataFrame(dict(
            trip_id=trip_id, route_id=G.route_id.iloc[0], direction_id=int(G.direction_id.iloc[0]), shape_id=shape_id,
            stop_idx=np.arange(n), stop_id=G.stop_id.to_numpy(), stop_sequence=G.stop_sequence.to_numpy(), sprog=sprog,
            sched=G.sched_arr.to_numpy(float), a=a, detect=detect)))
        I = inst_by.get(trip_id)
        if I is not None:
            pi = np.searchsorted(arr["ts"], I.ping_ts.to_numpy(float))
            assert np.all(arr["ts"][pi] == I.ping_ts.to_numpy(float)), "instant ping not found in re-matched pings"
            k = kinematics(arr, pi)
            kin_parts.append(pd.DataFrame(dict(instant_id=I.instant_id.to_numpy(), **k)))
    ev = pd.concat(ev_parts, ignore_index=True)
    kin = pd.concat(kin_parts, ignore_index=True)
    return ev, kin, chk
