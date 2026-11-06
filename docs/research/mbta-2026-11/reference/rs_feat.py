"""Phase 2c causal feature builder for (instant t, trip instance, downstream stop j) rows.  See models/research/NOTES.md for the table.

CAUSALITY CONTRACT (each point has a test in 2c_test_causality.py):
  * vehicle state            = latest valid ping <= t (ping_ts <= t); kinematics from pings <= ping_ts
  * own passed stops (< j0)  = arrivals interpolated from pings <= ping_ts (detect <= ping_ts, asserted)
  * other trips' events      = used only if detect (ts of the first ping after the crossing) <= t
  * statistics               = HistStats fitted on train-pool days only (out-of-fold for train rows)
  * schedule                 = static GTFS of the day;  MBTA TripUpdates are never read.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from rs_stats import local_offset_s, local_hour, daytype_of

WINDOWS = (15, 30, 60)
TEMPORAL = ["hour", "day_of_week", "is_weekend", "is_holiday", "is_peak_hour", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
GROUPS = {
    "serve": ["distance_to_stop", "stops_ahead", "minute_of_day"] + TEMPORAL,      # what the package path supplies (distance, stops ahead, clock)
    "ident": ["route_i", "direction_id", "stop_sequence", "n_stops"],              # static identity of the target stop (route/stop/direction)
    "pos": ["dist_match", "prog_frac", "dist_in_seg", "seg_len_cur", "stop_progress_m"],
    "sched": ["delay_now", "sched_rem", "sched_rem_t", "delay_last", "delay_prev", "delay_start"],
    "prior": ["prior_rem", "hist_cur"],
    "own": ["since_last", "nstops_since", "own_r1", "own_r3", "own_r8"],
    "kin": ["spd30", "spd60", "spd120", "stopped_s", "dist300", "status_code", "trip_age_s", "ping_age"],
    "xv_seg": [f"{p}{w}" for p in ("rr", "rn") for w in WINDOWS],
    "xv_route": [f"{p}{w}" for p in ("rt", "rtn", "rdel") for w in WINDOWS],
    "lead": ["lead_gap_j", "lead_delay_j", "lead_sched_hw_j", "lead_obs_hw_j", "lead_gap_next", "lead_gap_last", "lead_delay_next", "lead_delay_last"],
}
ALL_FEATURES = [f for g in GROUPS.values() for f in g]
# servable-plus: only what a small package change could supply (no direction: the val path passes direction_id=0; no cross-vehicle state)
SERVABLE_PLUS = GROUPS["serve"] + ["route_i", "stop_sequence", "delay_now", "sched_rem", "sched_rem_t", "delay_last", "prior_rem"]
GAP_CAP = 7200.0


class _CumArrays:
    """Sorted-by-detect event arrays with prefix sums; window queries are searchsorted."""
    def __init__(self, det, *vals):
        o = np.argsort(det, kind="stable")
        self.det = np.asarray(det, float)[o]
        self.cs = [np.concatenate([[0.0], np.cumsum(np.asarray(v, float)[o])]) for v in vals]

    def window(self, t, W):
        i1 = np.searchsorted(self.det, t, side="right"); i0 = np.searchsorted(self.det, t - W * 60.0, side="right")
        return (i1 - i0, *[c[i1] - c[i0] for c in self.cs])


def _prepare_day(ev, stats, day):
    """Day-level event structures (all derived from ev; queries filter by detect <= t)."""
    off = local_offset_s(day); dtp = daytype_of(day)
    ev = ev.sort_values(["trip_id", "stop_idx"], kind="stable").reset_index(drop=True)
    nxt = ev.shift(-1)
    same = (nxt.trip_id == ev.trip_id) & (nxt.stop_idx == ev.stop_idx + 1)
    ev["stop_id_next"] = np.where(same, nxt.stop_id, None)
    ev["seglen_next"] = np.where(same, nxt.sprog - ev.sprog, np.nan)
    ev["sid_next"] = -1
    if same.any():
        ev.loc[same, "sid_next"] = stats.sids(ev.route_id[same].astype(str), ev.stop_id[same].astype(str), nxt.stop_id[same].astype(str))
    segkey = np.where(same, ev.route_id.astype(str) + "|" + ev.stop_id.astype(str) + "|" + ev.stop_id_next.astype(str), None)
    # completed segment events
    ok = same & ev.a.notna() & nxt.a.notna() & nxt.detect.notna()
    tt = (nxt.a - ev.a).where(ok)
    ok &= (tt > 0) & (tt <= 1800)
    S = pd.DataFrame(dict(segkey=segkey[ok], route=ev.route_id[ok].astype(str).to_numpy(), dire=ev.direction_id[ok].to_numpy(),
                          det=nxt.detect[ok].to_numpy(), tt=tt[ok].to_numpy(), sid=ev.sid_next[ok].to_numpy(), a_from=ev.a[ok].to_numpy(),
                          sl=ev.seglen_next[ok].to_numpy()))
    S["hst"] = stats.lookup(S.sid.to_numpy(), dtp, local_hour(S.a_from.to_numpy(), off), S.sl.to_numpy(), S.route.to_numpy())
    seg_arr = {k: _CumArrays(g.det.to_numpy(), g.tt.to_numpy(), g.hst.to_numpy()) for k, g in S.groupby("segkey", sort=False)}
    route_arr = {k: _CumArrays(g.det.to_numpy(), g.tt.to_numpy(), g.hst.to_numpy()) for k, g in S.groupby(["route", "dire"], sort=False)}
    A = ev[ev.a.notna() & ev.detect.notna()]
    dly = np.clip(A.a.to_numpy() - A.sched.to_numpy(), -1800, 3600)
    del_arr = {k: _CumArrays(A.detect.to_numpy()[ix], dly[ix], np.ones(len(ix))) for k, ix in A.groupby(["route_id", "direction_id"], sort=False).indices.items()}
    lead_arr = {}
    for k, ix in A.groupby(["route_id", "direction_id", "stop_id"], sort=False).indices.items():
        o = np.argsort(A.detect.to_numpy()[ix], kind="stable"); ix = ix[o]
        lead_arr[k] = (A.detect.to_numpy()[ix], A.a.to_numpy()[ix], A.sched.to_numpy()[ix], A.trip_id.to_numpy()[ix])
    return dict(ev=ev, off=off, dtp=dtp, seg_arr=seg_arr, route_arr=route_arr, del_arr=del_arr, lead_arr=lead_arr)


def _last_valid_before(valid):
    n = len(valid)
    idx = np.where(valid, np.arange(n), -1)
    lv = np.maximum.accumulate(idx)
    return np.concatenate([[-1], lv])        # lv_before[j] = last valid k < j  (length n+1)


def _trip_block(E, I, D, stats, route):
    """Instant x stop matrices for one trip.  E: events of the trip (n rows); I: instants of the trip (T rows, with kin columns)."""
    n = len(E); T = len(I)
    off, dtp = D["off"], D["dtp"]
    sprog = E.sprog.to_numpy(float); sched = E.sched.to_numpy(float); a = E.a.to_numpy(float); det = E.detect.to_numpy(float)
    sidk = E.sid_next.to_numpy(np.int64)[:-1]; slk = E.seglen_next.to_numpy(float)[:-1]
    t = I.t.to_numpy(float); pts = I.ping_ts.to_numpy(float); prog = I.prog.to_numpy(float)
    j0 = np.searchsorted(sprog, prog, side="right")
    assert np.all(j0 <= n - 1) and np.all(j0 <= n)
    out = {}
    # ---- own history (stops < j0): arrivals detectable at the ping
    av = np.isfinite(a)
    kk = np.arange(n)
    own_ok = kk[None, :] < j0[:, None]
    assert np.all(~(own_ok & av[None, :] & (det[None, :] > pts[:, None] + 1e-6))), "own arrival used before its detection"
    lvb = _last_valid_before(av)
    k1 = lvb[j0]; k2 = np.where(k1 >= 0, lvb[np.maximum(k1, 0)], -1)
    dl = a - sched
    fv = np.argmax(av) if av.any() else -1
    out["delay_last"] = np.where(k1 >= 0, dl[np.maximum(k1, 0)], np.nan)
    out["delay_prev"] = np.where(k2 >= 0, dl[np.maximum(k2, 0)], np.nan)
    out["delay_start"] = np.where((fv >= 0) & (fv < j0), dl[max(fv, 0)], np.nan)
    out["since_last"] = np.where(k1 >= 0, t - a[np.maximum(k1, 0)], np.nan)
    out["nstops_since"] = np.where(k1 >= 0, j0 - 1 - k1, np.nan)
    # own segment ratios r_k = tt_k / hist_k  (segment k-1 -> k), k >= 1
    ratio = np.full(n, np.nan)
    if n > 1:
        tt = a[1:] - a[:-1]
        hist = stats.lookup(sidk, dtp, local_hour(np.where(av[:-1], a[:-1], 0.0), off), slk, np.full(n - 1, route))
        okr = av[1:] & av[:-1] & (tt > 0) & (tt <= 1800)
        ratio[1:] = np.where(okr, tt / hist, np.nan)
    rv = np.isfinite(ratio)
    csr = np.concatenate([[0.0], np.cumsum(np.where(rv, ratio, 0.0))]); csc = np.concatenate([[0.0], np.cumsum(rv)])   # prefix over k (index k+1)
    for M, nm in ((1, "own_r1"), (3, "own_r3"), (8, "own_r8")):
        lo = np.maximum(1, j0 - M); hi = j0                  # k in [lo, j0-1]
        cnt = csc[hi] - csc[np.minimum(lo, hi)]; sm = csr[hi] - csr[np.minimum(lo, hi)]
        out[nm] = np.where(cnt > 0, sm / np.maximum(cnt, 1), np.nan)
    # ---- schedule-relative state
    sm_ = np.maximum.accumulate(sprog)
    sched_now = np.interp(prog, sm_, sched)
    out["delay_now"] = pts - sched_now
    # ---- path prior (dynamic hour); eta[i,j] seconds from t to arrival at j
    s = j0 - 1
    sc = np.maximum(s, 0)
    if n > 1:
        span = np.maximum(sprog[np.minimum(sc + 1, n - 1)] - sprog[sc], 1.0)
        frac = np.where(s >= 0, np.clip((prog - sprog[sc]) / span, 0, 1), 0.0)
        Hcur = stats.lookup(sidk[np.minimum(sc, n - 2)], dtp, local_hour(t, off), slk[np.minimum(sc, n - 2)], np.full(T, route))
    else:
        frac = np.zeros(T); Hcur = np.zeros(T)
    cur = np.where(s >= 0, (1 - frac) * Hcur, 0.0)
    eta = np.full((T, n), np.nan)
    for j in range(n):
        m = j0 == j
        eta[m, j] = cur[m]
        if j > 0:
            g = j0 < j
            if g.any():
                prev = eta[g, j - 1]
                eta[g, j] = prev + stats.lookup(sidk[j - 1], dtp, local_hour(t[g] + prev, off), slk[j - 1], np.full(g.sum(), route))
    out["hist_cur"] = Hcur
    out["prog_frac"] = prog / max(sprog[-1], 1.0)
    out["dist_in_seg"] = np.where(s >= 0, prog - sprog[sc], prog)
    out["seg_len_cur"] = np.where(s >= 0, sprog[np.minimum(sc + 1, n - 1)] - sprog[sc], sprog[min(1, n - 1)] - sprog[0])
    # ---- cross-vehicle recent conditions
    sids_ = E.stop_id.astype(str).to_numpy()
    keys = [f"{route}|{sids_[k]}|{sids_[k + 1]}" for k in range(n - 1)]
    lo_seg = np.maximum(j0 - 1, 0)
    for W in WINDOWS:
        CT = np.zeros((T, n)); CH = np.zeros((T, n)); CN = np.zeros((T, n))      # prefix: column k+1 = sum over segments <= k
        for k in range(n - 1):
            arr_ = D["seg_arr"].get(keys[k])
            if arr_ is None:
                continue
            c, stt, sh = arr_.window(t, W)
            CN[:, k + 1] = c; CT[:, k + 1] = stt; CH[:, k + 1] = sh
        CT, CH, CN = np.cumsum(CT, 1), np.cumsum(CH, 1), np.cumsum(CN, 1)
        out[f"_CT{W}"], out[f"_CH{W}"], out[f"_CN{W}"] = CT, CH, CN
    out["_lo_seg"] = lo_seg
    rk = (str(route), int(E.direction_id.iloc[0]))
    for W in WINDOWS:
        ra = D["route_arr"].get(rk)
        if ra is not None:
            c, stt, sh = ra.window(t, W)
            out[f"rt{W}"] = np.where(sh > 0, stt / np.maximum(sh, 1e-9), np.nan); out[f"rtn{W}"] = c.astype(float)
        else:
            out[f"rt{W}"] = np.full(T, np.nan); out[f"rtn{W}"] = np.zeros(T)
        da = D["del_arr"].get((rk[0], rk[1]))
        if da is not None:
            c, sd, sn = da.window(t, W)
            out[f"rdel{W}"] = np.where(sn > 0, sd / np.maximum(sn, 1), np.nan)
        else:
            out[f"rdel{W}"] = np.full(T, np.nan)
    # ---- leaders (matrices over target stops)
    LG = np.full((T, n), np.nan); LD = np.full((T, n), np.nan); LH = np.full((T, n), np.nan); LO = np.full((T, n), np.nan)
    own_trip = E.trip_id.iloc[0]
    for j in range(n):
        la = D["lead_arr"].get((E.route_id.iloc[0], E.direction_id.iloc[0], E.stop_id.iloc[j]))
        if la is None:
            continue
        ldet, la_, ls, ltr = la
        idx = np.searchsorted(ldet, t, side="right") - 1
        idx = np.where((idx >= 0) & (ltr[np.maximum(idx, 0)] == own_trip), idx - 1, idx)
        ok = idx >= 0
        ii = np.maximum(idx, 0)
        LG[:, j] = np.where(ok, np.minimum(t - la_[ii], GAP_CAP), np.nan)
        LD[:, j] = np.where(ok, np.clip(la_[ii] - ls[ii], -1800, 3600), np.nan)
        LH[:, j] = np.where(ok, np.clip(sched[j] - ls[ii], -7200, 7200), np.nan)
        i2 = np.maximum(idx - 1, 0)
        ok2 = ok & (idx >= 1) & (ltr[i2] != own_trip)
        LO[:, j] = np.where(ok2, np.minimum(la_[ii] - la_[i2], GAP_CAP), np.nan)
    out["_LG"], out["_LD"], out["_LH"], out["_LO"] = LG, LD, LH, LO
    out["_eta"], out["_j0"], out["_sched"], out["_sprog"], out["_sched_now"] = eta, j0, sched, sprog, sched_now
    return out


def build_features(day, inst, rows, ev, kin, stats, label=True):
    """inst/rows: harness tables (rows already subsampled/selected by the caller). Returns a float32 feature frame, one row per `rows` row."""
    D = _prepare_day(ev, stats, day)
    inst = inst.merge(kin, on="instant_id", how="left")
    R = rows.merge(inst, on="instant_id", how="left", suffixes=("", "_i"))
    ev_by = {k: g for k, g in D["ev"].groupby("trip_id", sort=False)}
    parts = []
    for trip_id, Rt in R.groupby("trip_id", sort=False):
        E = ev_by.get(trip_id)
        if E is None:
            continue
        uid, inv = np.unique(Rt.instant_id.to_numpy(), return_inverse=True)
        It = inst[inst.instant_id.isin(uid)].set_index("instant_id").loc[uid].reset_index()
        B = _trip_block(E.reset_index(drop=True), It, D, stats, str(Rt.route_id.iloc[0]))
        j = Rt.stop_idx.to_numpy(); i = inv
        f = {}
        for nm in ("delay_now", "delay_last", "delay_prev", "delay_start", "since_last", "nstops_since", "own_r1", "own_r3", "own_r8", "hist_cur", "prog_frac",
                   "dist_in_seg", "seg_len_cur"):
            f[nm] = B[nm][i]
        for W in WINDOWS:
            for nm in (f"rt{W}", f"rtn{W}", f"rdel{W}"):
                f[nm] = B[nm][i]
            lo = B["_lo_seg"][i]
            ct = B[f"_CT{W}"][i, j] - B[f"_CT{W}"][i, lo]; ch = B[f"_CH{W}"][i, j] - B[f"_CH{W}"][i, lo]; cn = B[f"_CN{W}"][i, j] - B[f"_CN{W}"][i, lo]
            f[f"rr{W}"] = np.where(ch > 0, ct / np.maximum(ch, 1e-9), np.nan); f[f"rn{W}"] = cn
        j0 = B["_j0"][i]
        f["prior_rem"] = B["_eta"][i, j]
        f["sched_rem"] = B["_sched"][j] - B["_sched_now"][i]
        f["sched_rem_t"] = B["_sched"][j] - Rt.t.to_numpy(float)
        f["dist_match"] = B["_sprog"][j] - It.prog.to_numpy(float)[i]
        for nm, M in (("lead_gap_j", "_LG"), ("lead_delay_j", "_LD"), ("lead_sched_hw_j", "_LH"), ("lead_obs_hw_j", "_LO")):
            f[nm] = B[M][i, j]
        f["lead_gap_next"] = B["_LG"][i, j0]; f["lead_delay_next"] = B["_LD"][i, j0]
        jl = np.maximum(j0 - 1, 0)
        f["lead_gap_last"] = np.where(j0 >= 1, B["_LG"][i, jl], np.nan); f["lead_delay_last"] = np.where(j0 >= 1, B["_LD"][i, jl], np.nan)
        df = pd.DataFrame(f)
        df["row_id"] = Rt.row_id.to_numpy(); df["instant_id"] = Rt.instant_id.to_numpy(); df["trip_id"] = trip_id
        df["t"] = Rt.t.to_numpy(float); df["ping_ts"] = Rt.ping_ts.to_numpy(float); df["stop_idx"] = j
        df["stop_sequence"] = Rt.stop_sequence.to_numpy(); df["stop_progress_m"] = Rt.stop_progress_m.to_numpy(float)
        df["n_stops"] = Rt.n_stops.to_numpy(); df["direction_id"] = Rt.direction_id.to_numpy()
        df["route_i"] = int(Rt.route_id.iloc[0]); df["route_id"] = str(Rt.route_id.iloc[0])
        df["distance_to_stop"] = Rt.stop_progress_m.to_numpy(float) - Rt.pkg_prog.to_numpy(float)
        df["stops_ahead"] = (j - Rt.pkg_j0.to_numpy()).astype(float)
        df["ping_age"] = Rt.t.to_numpy(float) - Rt.ping_ts.to_numpy(float)
        for nm in ("spd30", "spd60", "spd120", "stopped_s", "dist300", "status_code", "trip_age_s"):
            df[nm] = Rt[nm].to_numpy(float)
        df["gt"] = Rt.gt_interp.to_numpy(float) if "gt_interp" in Rt else np.nan
        parts.append(df)
    F = pd.concat(parts, ignore_index=True)
    F["day"] = str(day)
    F = add_clock(F)
    F["y"] = (F["gt"] - F["t"]).astype(np.float32)
    return F


def add_clock(F):
    from evalpred_model import temporal_frame
    T = temporal_frame(F.t.to_numpy(float), "America/New_York", "US_MA")
    for c in TEMPORAL:
        F[c] = T[c].to_numpy().astype(np.float32)
    off = local_offset_s(F.day.iloc[0])
    F["minute_of_day"] = (((F.t.to_numpy(float) + off) % 86400.0) / 60.0).astype(np.float32)
    return F
