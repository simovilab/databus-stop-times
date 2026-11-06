"""Vectorised-ish ground-truth arrival extraction for one trip instance (PROTOCOL section 2).

interp: pings are map-matched to the GTFS shape with a forward-only windowed projection (no loop-back snapping),
then the arrival at stop j is the linear interpolation of time at which along-shape progress first reaches the
stop's progress S_j. Rejected if the bracketing valid-ping gap > MAX_GAP_S, or there is no bracketing pair.
"""
import math
import numpy as np

M_PER_DEG = 6371000.0 * math.pi / 180.0
XT_MAX_M = 100.0       # ping farther than this from the allowed shape window = off-route, ignored
BACK_TOL_M = 60.0      # allowed backward GPS jitter
MAX_GAP_S = 60.0       # protocol: bracketing gap
TIE_M = 20.0           # among near-equal cross-track candidates prefer earliest progress


def poly_arrays(polyline):
    lat = np.array([p.lat for p in polyline]); lon = np.array([p.lon for p in polyline]); cum = np.array([p.cum_m for p in polyline])
    return lat, lon, cum


def _project_window(plat, plon, lat, lon, cum, k0, k1):
    """Project (plat,plon) on segments k0..k1-1; returns (progress[], xt[]) arrays."""
    c = math.cos(math.radians(plat))
    ax = (lon[k0:k1] - plon) * c * M_PER_DEG; ay = (lat[k0:k1] - plat) * M_PER_DEG
    bx = (lon[k0 + 1:k1 + 1] - plon) * c * M_PER_DEG; by = (lat[k0 + 1:k1 + 1] - plat) * M_PER_DEG
    dx = bx - ax; dy = by - ay
    L2 = dx * dx + dy * dy
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(L2 > 0, -(ax * dx + ay * dy) / L2, 0.0)
    t = np.clip(t, 0.0, 1.0)
    cx = ax + t * dx; cy = ay + t * dy
    xt = np.hypot(cx, cy)
    prog = cum[k0:k1] + t * (cum[k0 + 1:k1 + 1] - cum[k0:k1])
    return prog, xt


def match_pings(ts, plat, plon, poly):
    """Forward-only map matching. Returns (valid_mask, progress[valid-aligned full array, nan if invalid], xt)."""
    lat, lon, cum = poly_arrays(poly)
    n = len(ts); nseg = len(lat) - 1
    prog = np.full(n, np.nan); xtv = np.full(n, np.nan)
    p_prev = None; t_prev = None; lost = True
    for i in range(n):
        if p_prev is None or lost:
            lo_m = 0.0 if p_prev is None else p_prev - BACK_TOL_M
            hi_m = np.inf
        else:
            dt_ = max(ts[i] - t_prev, 0.0)
            lo_m = p_prev - BACK_TOL_M
            hi_m = p_prev + min(35.0 * dt_ + 400.0, 3000.0)
        k0 = int(np.searchsorted(cum, lo_m, side="right")) - 1
        k0 = max(k0, 0)
        k1 = nseg if not np.isfinite(hi_m) else int(np.searchsorted(cum, hi_m, side="left"))
        k1 = min(max(k1, k0 + 1), nseg)
        pr, xt = _project_window(plat[i], plon[i], lat, lon, cum, k0, k1)
        j = int(np.argmin(xt))
        if xt[j] > XT_MAX_M:
            lost = True
            continue
        if lost or p_prev is None:
            near = np.flatnonzero(xt <= xt[j] + TIE_M)
            j = int(near[0])
        p = max(pr[j], p_prev if p_prev is not None else 0.0)
        if p_prev is not None and pr[j] < p_prev - BACK_TOL_M:
            lost = True; continue
        prog[i] = p; xtv[i] = xt[j]; p_prev = p; t_prev = ts[i]; lost = False
    return ~np.isnan(prog), prog, xtv


def interp_arrivals(ts, prog, stop_prog):
    """ts/prog: valid pings (sorted, prog nondecreasing). Returns (arr[], gap[], reason[])."""
    n = len(stop_prog)
    arr = np.full(n, np.nan); gap = np.full(n, np.nan); reason = np.array(["ok"] * n, dtype=object)
    if len(ts) < 2:
        reason[:] = "too_few_pings"; return arr, gap, reason
    idx = np.searchsorted(prog, stop_prog, side="left")
    for j in range(n):
        k = idx[j]
        if k == 0:
            reason[j] = "before_first_ping"; continue
        if k >= len(prog):
            reason[j] = "never_reached"; continue
        g = ts[k] - ts[k - 1]; gap[j] = g
        if g > MAX_GAP_S:
            reason[j] = "gap_gt_60s"; continue
        dp = prog[k] - prog[k - 1]
        if dp <= 0:
            reason[j] = "non_monotonic"; continue
        arr[j] = ts[k - 1] + (stop_prog[j] - prog[k - 1]) / dp * g
    return arr, gap, reason
