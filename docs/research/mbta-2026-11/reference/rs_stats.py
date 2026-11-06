"""Historical segment statistics (train-pool days ONLY) for the path prior.

Segment = (route, from_stop_id, to_stop_id); value = arrival-to-arrival time (traversal + dwell at the from-stop) from GT-equivalent
arrivals.  Dense table H[sid, daytype(wkday/sat/sun), local hour] (median, seconds) with a hierarchical fallback:
  (seg,daytype,hour) n>=5 -> (seg,daytype,hour+-1) n>=8 -> (seg,hour+-1) n>=8 -> (seg,daytype) n>=3 -> (seg) -> length / route median speed.
Out-of-fold use: a training day's features use the table fitted on the OTHER folds of the train pool (fold = day-index mod K), so the
path-prior feature has the same (out-of-sample) quality on train rows as on val rows (otherwise the tree over-trusts it).
"""
import os, sys, datetime as dt
import numpy as np, pandas as pd

TT_MAX = 1800.0
MIN_N = dict(l0=5, l1=8, l1b=8, l2=3)
K_FOLDS = 6


def daytype_of(day):
    w = dt.date.fromisoformat(str(day)).weekday()
    return 0 if w < 5 else (1 if w == 5 else 2)


def local_offset_s(day, tz="America/New_York"):
    """UTC offset (s) at local noon of the service date (constant within a service day except on DST-change days)."""
    d = dt.date.fromisoformat(str(day))
    return float(pd.Timestamp(dt.datetime(d.year, d.month, d.day, 12), tz=tz).utcoffset().total_seconds())


def local_hour(epoch, off):
    return (np.floor((np.asarray(epoch, float) + off) / 3600.0).astype(np.int64)) % 24


def segment_rows(ev, day):
    """ev: events of one day (columns trip_id, route_id, stop_idx, stop_id, sprog, a). Returns consecutive-stop segment rows."""
    e = ev.sort_values(["trip_id", "stop_idx"], kind="stable")
    nxt = e.shift(-1)
    ok = (nxt.trip_id == e.trip_id) & (nxt.stop_idx == e.stop_idx + 1) & e.a.notna() & nxt.a.notna()
    tt = (nxt.a - e.a)[ok]
    s = pd.DataFrame(dict(route=e.route_id[ok].astype(str), frm=e.stop_id[ok].astype(str), to=nxt.stop_id[ok].astype(str),
                          tt=tt.to_numpy(float), seglen=(nxt.sprog - e.sprog)[ok].to_numpy(float), a_from=e.a[ok].to_numpy(float)))
    s = s[(s.tt > 0) & (s.tt <= TT_MAX)]
    off = local_offset_s(day)
    s["hour"] = local_hour(s.a_from.to_numpy(), off); s["dtype"] = daytype_of(day); s["day"] = str(day)
    return s.drop(columns="a_from")


class HistStats:
    def __init__(self, sid_map, H, route_speed, speed_default, n_obs):
        self.sid_map, self.H, self.route_speed, self.speed_default, self.n_obs = sid_map, H, route_speed, speed_default, n_obs

    def sids(self, route, frm, to):
        keys = pd.Series(np.char.add(np.char.add(np.asarray(route, str), "|"), np.char.add(np.asarray(frm, str), np.char.add("|", np.asarray(to, str)))))
        return keys.map(self.sid_map).fillna(-1).astype(np.int64).to_numpy()

    def lookup(self, sid, dtype, hour, seglen, route):
        """Median arrival-to-arrival seconds. sid/hour/seglen/route arrays (any broadcastable shapes); dtype int scalar or array."""
        sid = np.asarray(sid); hour = np.asarray(hour)
        ok = sid >= 0
        v = np.where(ok, self.H[np.where(ok, sid, 0), dtype, hour], np.nan)
        sp = pd.Series(np.asarray(route).ravel().astype(str)).map(self.route_speed).fillna(self.speed_default).to_numpy().reshape(np.shape(route))
        fb = np.maximum(np.asarray(seglen, float), 30.0) / sp + 15.0
        return np.where(np.isfinite(v), v, fb)


def fit(seg):
    """seg: concatenated segment_rows over the fitting days."""
    keys = seg.route + "|" + seg.frm + "|" + seg.to
    sid, uniq = pd.factorize(keys)
    S = len(uniq)
    d = pd.DataFrame(dict(sid=sid, dtype=seg.dtype.to_numpy(), hour=seg.hour.to_numpy(), tt=seg.tt.to_numpy()))
    H = np.full((S, 3, 24), np.nan, np.float32)

    # level 0
    g = d.groupby(["sid", "dtype", "hour"]).tt.agg(["median", "count"]).reset_index()
    g = g[g["count"] >= MIN_N["l0"]]
    H[g.sid.to_numpy(), g.dtype.to_numpy(), g.hour.to_numpy()] = g["median"].to_numpy(np.float32)
    # level 1: hour window +-1 (rows copied to h-1,h,h+1)
    dd = pd.concat([d.assign(hour=(d.hour + s) % 24) for s in (-1, 0, 1)], ignore_index=True)
    g = dd.groupby(["sid", "dtype", "hour"]).tt.agg(["median", "count"]).reset_index(); g = g[g["count"] >= MIN_N["l1"]]
    cur = H[g.sid.to_numpy(), g.dtype.to_numpy(), g.hour.to_numpy()]
    H[g.sid.to_numpy(), g.dtype.to_numpy(), g.hour.to_numpy()] = np.where(np.isnan(cur), g["median"].to_numpy(np.float32), cur)
    # level 1b: ignore daytype
    g = dd.groupby(["sid", "hour"]).tt.agg(["median", "count"]).reset_index(); g = g[g["count"] >= MIN_N["l1b"]]
    for dtp in range(3):
        cur = H[g.sid.to_numpy(), dtp, g.hour.to_numpy()]
        H[g.sid.to_numpy(), dtp, g.hour.to_numpy()] = np.where(np.isnan(cur), g["median"].to_numpy(np.float32), cur)
    # level 2: (seg, daytype)
    g = d.groupby(["sid", "dtype"]).tt.agg(["median", "count"]).reset_index(); g = g[g["count"] >= MIN_N["l2"]]
    m2 = np.full((S, 3), np.nan, np.float32); m2[g.sid.to_numpy(), g.dtype.to_numpy()] = g["median"].to_numpy(np.float32)
    m3 = d.groupby("sid").tt.median().reindex(range(S)).to_numpy(np.float32)
    fillv = np.where(np.isnan(m2), m3[:, None], m2)                      # [S,3]
    H = np.where(np.isnan(H), fillv[:, :, None], H).astype(np.float32)
    vv = seg.seglen.to_numpy() / seg.tt.to_numpy()
    okv = (seg.tt.to_numpy() > 5) & (seg.seglen.to_numpy() > 20)
    rs = pd.Series(vv[okv]).groupby(seg.route.to_numpy()[okv]).median().to_dict()
    sp_def = float(np.median(vv[okv])) if okv.any() else 5.0
    return HistStats({k: i for i, k in enumerate(uniq)}, H, rs, sp_def, len(seg))


def fit_oof(seg_by_day, days, k=K_FOLDS):
    """Return ({fold: HistStats fitted on days not in fold}, HistStats fitted on all `days`, {day: fold}). Fold = index of the day mod k."""
    fold = {d: i % k for i, d in enumerate(days)}
    allseg = pd.concat([seg_by_day[d] for d in days], ignore_index=True)
    allseg["fold"] = allseg.day.map(fold).to_numpy()
    by_fold = {f: fit(allseg[allseg.fold != f].reset_index(drop=True)) for f in range(k)}
    return by_fold, fit(allseg), fold
