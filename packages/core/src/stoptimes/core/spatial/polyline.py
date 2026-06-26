"""Spatial map-matching, implemented fresh (no external geometry dep). Phase P2.

haversine_m; build_polyline(shape_pts, shape_dist_traveled?) -> cumulative (lat,lon,cum_m);
project_point_to_polyline -> {progress_m, cross_track_m, segment_idx};
assign_stops_monotonic(stops, polyline) -> forward-monotonic (loop-back-safe) stop progress;
upcoming_stops(progress_m, stops) slice.
"""
