"""Build the full-corpus segment dataset from all downloaded navsat days."""
import glob
import pandas as pd

from feature_engineering.bucr_gtfs import load_gtfs, route_directions, route_direction_candidates
from feature_engineering.navsat_cleaning import clean_navsat_trace
from feature_engineering.navsat_normalize import normalize_raw_navsat
from feature_engineering.navsat_adapter import navsat_to_vehicle_positions
from feature_engineering.bucr_trip_inference import infer_bucr_trips
from feature_engineering.segment_dataset_builder import build_bucr_segment_dataset

BASE = "/Users/dotj/Desktop/SIMOVI/git.no_sync/gtfs-eta"
OUT = f"{BASE}/datasets/bucr_segments_corpus.parquet"

gtfs = load_gtfs(f"{BASE}/scratchpad/gtfs.zip")
candidates = []
for route_id, direction_id in route_directions(gtfs):
    candidates.extend(route_direction_candidates(gtfs, route_id, direction_id))

files = sorted(glob.glob(f"{BASE}/scratchpad/navsat_full/**/*.parquet", recursive=True))
print(f"navsat day files: {len(files)}")
raw = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
print(f"raw rows: {len(raw):,}")

raw = normalize_raw_navsat(raw)
cleaned, crep = clean_navsat_trace(raw)
print(f"cleaned rows: {len(cleaned):,}  (dropped {crep.drop_rate:.1%})")
vp = navsat_to_vehicle_positions(cleaned)
inferred, stats = infer_bucr_trips(vp, candidates)
print(f"inferred: assigned {stats.points_assigned:,}/{stats.points_in:,} "
      f"({stats.points_assigned/stats.points_in:.1%})  trips {stats.trips_inferred}")

seg = build_bucr_segment_dataset(inferred, candidates)
seg["cr_day"] = (seg["vp_ts"] - pd.Timedelta(hours=6)).dt.date
print(f"\nSEGMENTS: {len(seg):,} rows  trips {seg['trip_id'].nunique():,}  "
      f"days {seg['cr_day'].nunique()}  legs {seg.groupby(['from_stop_id','stop_id']).ngroups}")
print("segments per CR-day:")
print(seg.groupby("cr_day").size().to_string())
tta = seg["time_to_arrival_seconds"]
print(f"\ntarget: median {tta.median():.0f}s  mean {tta.mean():.0f}s  p95 {tta.quantile(.95):.0f}s")

seg.drop(columns="cr_day").to_parquet(OUT, index=False)
print(f"\nsaved -> {OUT}")
