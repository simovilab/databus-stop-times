"""End-to-end smoke: clean -> adapt -> infer -> segment-build on the real sample."""
import glob
import pandas as pd

from feature_engineering.bucr_gtfs import load_gtfs, route_directions, route_direction_candidates
from feature_engineering.navsat_cleaning import clean_navsat_trace
from feature_engineering.navsat_normalize import normalize_raw_navsat
from feature_engineering.navsat_adapter import navsat_to_vehicle_positions
from feature_engineering.bucr_trip_inference import infer_bucr_trips
from feature_engineering.segment_dataset_builder import build_bucr_segment_dataset

BASE = "/Users/dotj/Desktop/SIMOVI/git.no_sync/gtfs-eta/scratchpad"

gtfs = load_gtfs(f"{BASE}/gtfs.zip")
candidates = []
for route_id, direction_id in route_directions(gtfs):
    candidates.extend(route_direction_candidates(gtfs, route_id, direction_id))
print(f"candidates: {len(candidates)}")

raw = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{BASE}/navsat_sample/*.parquet"))], ignore_index=True)
print(f"raw rows: {len(raw):,}")

raw = normalize_raw_navsat(raw)
cleaned, _crep = clean_navsat_trace(raw)
vp = navsat_to_vehicle_positions(cleaned)
inferred, stats = infer_bucr_trips(vp, candidates)
print(f"inferred assigned points: {stats.points_assigned:,}  trips: {stats.trips_inferred}")

seg = build_bucr_segment_dataset(inferred, candidates)
print(f"\n=== SEGMENT DATASET ===")
print(f"rows: {len(seg):,}")
print(f"trips (instances): {seg['trip_id'].nunique():,}")
print(f"routes: {seg['route_id'].nunique()}  shapes-via-trip: n/a")
print(f"distinct segments (from->to): {seg.groupby(['from_stop_id','stop_id']).ngroups}")
tta = seg["time_to_arrival_seconds"]
print(f"\ntarget time_to_arrival_seconds:")
print(f"  count {tta.count():,}  min {tta.min():.0f}s  median {tta.median():.0f}s  mean {tta.mean():.0f}s  p95 {tta.quantile(.95):.0f}s  max {tta.max():.0f}s")
print(f"\ndistance_to_stop (segment length m): median {seg['distance_to_stop'].median():.0f}  min {seg['distance_to_stop'].min():.0f}  max {seg['distance_to_stop'].max():.0f}")
print(f"cross_track_error m: median {seg['cross_track_error'].median():.1f}  p95 {seg['cross_track_error'].quantile(.95):.1f}")
print(f"stops_ahead range: {seg['stops_ahead'].min()}..{seg['stops_ahead'].max()}")
print(f"is_at_stop frac: {seg['is_at_stop'].mean():.2f}")
print(f"hour range (CR local): {seg['hour'].min()}..{seg['hour'].max()}")
print(f"\nnull check (feature cols):")
feat = ["distance_to_stop","distance_to_next_stop","shape_distance_to_stop","shape_progress","cross_track_error","progress_ratio","stops_ahead","current_speed_kmh","bearing_to_stop","bearing_diff","is_at_stop","hour","hour_sin","dow_sin","time_to_arrival_seconds"]
print(seg[feat].isna().mean().round(3).to_string())

# implied segment speed sanity
seg = seg.copy()
seg["kmh"] = seg["distance_to_stop"]/seg["time_to_arrival_seconds"]*3.6
print(f"\nimplied segment speed km/h: median {seg['kmh'].median():.1f}  p95 {seg['kmh'].quantile(.95):.1f}  max {seg['kmh'].max():.1f}")

seg.drop(columns="kmh").to_parquet(f"{BASE}/segments_real.parquet", index=False)
print(f"\nsaved -> {BASE}/segments_real.parquet")
