# feature_engineering/spatial.py - Shape-informed progress extension
from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional

import numpy as np

EARTH_RADIUS_M = 6_371_000.0

def _deg2rad(x: float) -> float:
    return x * math.pi / 180.0

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters."""
    φ1, φ2 = _deg2rad(lat1), _deg2rad(lat2)
    dφ = φ2 - φ1
    dλ = _deg2rad(lon2 - lon1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_M * c


def _haversine_vec(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distance (meters) from one point to an array of points."""
    phi1 = math.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + math.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(np.clip(1 - a, 0.0, None)))
    return EARTH_RADIUS_M * c


class ShapePolyline:
    """
    Represents a route shape as an ordered sequence of (lat, lon) points.
    Provides methods to project vehicle positions onto the polyline and compute
    accurate progress along the route.
    """
    
    def __init__(self, points: List[Tuple[float, float]]):
        """
        Args:
            points: List of (lat, lon) tuples in route order
        """
        if len(points) < 2:
            raise ValueError("Shape must have at least 2 points")
        
        self.points = points
        self._segment_lengths = self._compute_segment_lengths()
        self._cumulative_distances = self._compute_cumulative_distances()
        self.total_length = self._cumulative_distances[-1]

        # Precomputed per-segment planar-projection arrays (numpy), reused by
        # every project_point() call — was previously O(shape_points) worth
        # of Python-loop trig work recomputed from scratch on every call
        # (x2-3 calls per (VP, stop) pair in calculate_distance_features_with_shape),
        # which dominates runtime once the STEP-4 O(V^2) arrival-scan is fixed.
        pts = np.asarray(self.points, dtype=float)
        self._lat1_arr = pts[:-1, 0]
        self._lon1_arr = pts[:-1, 1]
        self._lat2_arr = pts[1:, 0]
        self._lon2_arr = pts[1:, 1]
        avg_lat = (self._lat1_arr + self._lat2_arr) / 2.0
        self._meters_per_deg_lon_arr = 111320.0 * np.cos(np.radians(avg_lat))
        self._seg_x_arr = (self._lon2_arr - self._lon1_arr) * self._meters_per_deg_lon_arr
        self._seg_y_arr = (self._lat2_arr - self._lat1_arr) * 111320.0
        self._seg_length_sq_arr = self._seg_x_arr ** 2 + self._seg_y_arr ** 2
        self._degenerate_arr = self._seg_length_sq_arr < 1e-6
        self._cumulative_arr = np.asarray(self._cumulative_distances, dtype=float)
    
    def _compute_segment_lengths(self) -> List[float]:
        """Compute distance for each segment between consecutive points."""
        lengths = []
        for i in range(len(self.points) - 1):
            lat1, lon1 = self.points[i]
            lat2, lon2 = self.points[i + 1]
            lengths.append(_haversine_m(lat1, lon1, lat2, lon2))
        return lengths
    
    def _compute_cumulative_distances(self) -> List[float]:
        """Compute cumulative distance from start to each point."""
        cumulative = [0.0]
        for length in self._segment_lengths:
            cumulative.append(cumulative[-1] + length)
        return cumulative
    
    def project_point(self, lat: float, lon: float) -> Dict:
        """
        Project a point onto the polyline, finding the closest position.
        
        Args:
            lat, lon: Point to project
            
        Returns:
            {
                'distance_along_shape': meters from shape start,
                'cross_track_distance': perpendicular distance from shape (meters),
                'closest_segment_idx': index of nearest segment,
                'progress': normalized progress [0, 1]
            }
        """
        # Vectorized over all segments at once (was: a Python loop calling
        # _project_onto_segment once per segment — see __init__ for the
        # precomputed arrays this reuses).
        with np.errstate(invalid='ignore', divide='ignore'):
            dx = (lon - self._lon1_arr) * self._meters_per_deg_lon_arr
            dy = (lat - self._lat1_arr) * 111320.0
            t = (dx * self._seg_x_arr + dy * self._seg_y_arr) / self._seg_length_sq_arr
        t = np.clip(t, 0.0, 1.0)

        proj_lon = self._lon1_arr + t * (self._lon2_arr - self._lon1_arr)
        proj_lat = self._lat1_arr + t * (self._lat2_arr - self._lat1_arr)
        dist = _haversine_vec(lat, lon, proj_lat, proj_lon)

        if self._degenerate_arr.any():
            dist_degenerate = _haversine_vec(lat, lon, self._lat1_arr, self._lon1_arr)
            dist = np.where(self._degenerate_arr, dist_degenerate, dist)
            t = np.where(self._degenerate_arr, 0.0, t)

        best_segment_idx = int(np.argmin(dist))
        min_dist = float(dist[best_segment_idx])
        seg_length = math.sqrt(float(self._seg_length_sq_arr[best_segment_idx]))
        best_projection_dist = float(t[best_segment_idx]) * seg_length

        # Calculate total distance along shape to projection point
        distance_along_shape = (
            self._cumulative_distances[best_segment_idx] + best_projection_dist
        )

        # Normalized progress
        progress = distance_along_shape / self.total_length if self.total_length > 0 else 0.0

        return {
            'distance_along_shape': distance_along_shape,
            'cross_track_distance': min_dist,
            'closest_segment_idx': best_segment_idx,
            'progress': min(1.0, max(0.0, progress))
        }
    
    def get_distance_between_stops(
        self, 
        stop1_lat: float, 
        stop1_lon: float,
        stop2_lat: float,
        stop2_lon: float
    ) -> float:
        """
        Get shape distance between two stops (more accurate than haversine).
        """
        proj1 = self.project_point(stop1_lat, stop1_lon)
        proj2 = self.project_point(stop2_lat, stop2_lon)
        return abs(proj2['distance_along_shape'] - proj1['distance_along_shape'])


def calculate_distance_features_with_shape(
    vehicle_position: Dict,
    stop: Dict,
    next_stop: Optional[Dict],
    shape: Optional[ShapePolyline] = None,
    vehicle_stop_order: Optional[int] = None,
    total_segments: Optional[int] = None,
) -> Dict:
    """
    Enhanced version that uses shape data when available.
    
    Args:
        vehicle_position: {'lat': float, 'lon': float, 'bearing': Optional[float]}
        stop: {'stop_id': str, 'lat': float, 'lon': float, 'stop_order': Optional[int]}
        next_stop: {'stop_id': str, 'lat': float, 'lon': float} or None
        shape: ShapePolyline instance or None
        vehicle_stop_order: 0-based index of the closest upstream stop (optional)
        total_segments: Total number of stop-to-stop segments in trip (optional)
        
    Returns:
        Same as calculate_distance_features() but with additional shape-based fields:
        - 'shape_progress': accurate progress along route shape [0, 1]
        - 'shape_distance_to_stop': along-shape distance to stop (meters)
        - 'cross_track_error': perpendicular distance from route (meters)
        - 'progress_ratio': coarse fallback progress when shape is missing
    """
    vlat, vlon = float(vehicle_position["lat"]), float(vehicle_position["lon"])
    slat, slon = float(stop["lat"]), float(stop["lon"])
    
    # Base features (always computed)
    result = {
        'distance_to_stop': _haversine_m(vlat, vlon, slat, slon),
        'distance_to_next_stop': None,
        'progress_on_segment': None,
        'progress_ratio': None,
        'shape_progress': None,
        'shape_distance_to_stop': None,
        'cross_track_error': None,
    }
    
    if next_stop is not None:
        nlat, nlon = float(next_stop["lat"]), float(next_stop["lon"])
        seg_len = _haversine_m(slat, slon, nlat, nlon)
        if seg_len == 0.0:
            result['distance_to_next_stop'] = 0.0
        else:
            result['distance_to_next_stop'] = _haversine_m(vlat, vlon, nlat, nlon)
    
    # Simple progress proxy if we have next stop but no shape
    if result['progress_on_segment'] is None and next_stop is not None and result['distance_to_next_stop'] is not None:
        nlat, nlon = float(next_stop["lat"]), float(next_stop["lon"])
        seg_len = _haversine_m(slat, slon, nlat, nlon)
        if seg_len > 0:
            progress = 1.0 - (result['distance_to_next_stop'] / seg_len)
            result['progress_on_segment'] = max(0.0, min(1.0, progress))
        else:
            result['progress_on_segment'] = 0.0

    # Shape-based features (if shape available)
    if shape is not None:
        vehicle_proj = shape.project_point(vlat, vlon)
        stop_proj = shape.project_point(slat, slon)
        
        # Distance along shape from vehicle to stop
        shape_dist_to_stop = stop_proj['distance_along_shape'] - vehicle_proj['distance_along_shape']
        
        result.update({
            'shape_progress': vehicle_proj['progress'],
            'shape_distance_to_stop': max(0, shape_dist_to_stop),  # Don't allow negative
            'cross_track_error': vehicle_proj['cross_track_distance'],
            'progress_ratio': vehicle_proj['progress'],
        })
        
        # If we have next_stop, compute segment progress along shape
        if next_stop is not None:
            next_proj = shape.project_point(nlat, nlon)
            segment_length = next_proj['distance_along_shape'] - stop_proj['distance_along_shape']
            
            if segment_length > 0:
                # How far past current stop along shape
                past_stop = vehicle_proj['distance_along_shape'] - stop_proj['distance_along_shape']
                result['progress_on_segment'] = max(0.0, min(1.0, past_stop / segment_length))
            else:
                result['progress_on_segment'] = 0.0
    # Fallback progress_ratio using stop order metadata
    if result['progress_ratio'] is None:
        order = vehicle_stop_order
        if order is None:
            order = stop.get("vehicle_stop_order") or stop.get("stop_order")
        segments = total_segments
        if segments is None:
            segments = stop.get("total_segments")
        if order is not None and segments:
            completed_segments = max(float(order), 0.0)
            progress_within = result['progress_on_segment'] or 0.0
            denom = max(float(segments), 1.0)
            ratio = (completed_segments + progress_within) / denom
            result['progress_ratio'] = max(0.0, min(1.0, ratio))

    return result


# ==================== Helper: Load shapes from GTFS ====================

def load_shape_from_gtfs(shape_id: str, conn) -> ShapePolyline:
    """
    Load a shape polyline from GTFS shapes table.
    
    Args:
        shape_id: GTFS shape_id
        conn: Database connection (psycopg2/asyncpg)
        
    Returns:
        ShapePolyline instance
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT shape_pt_lat, shape_pt_lon
            FROM sch_pipeline_shape
            WHERE shape_id = %s
            ORDER BY shape_pt_sequence
            """,
            (shape_id,)
        )
        rows = cur.fetchall()
    
    if not rows:
        raise ValueError(f"No shape found for shape_id: {shape_id}")
    
    points = [(float(row[0]), float(row[1])) for row in rows]
    return ShapePolyline(points)


def load_shape_for_trip(trip_id: str, conn) -> Optional[ShapePolyline]:
    """
    Load shape for a trip, returns None if trip has no shape.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT shape_id
            FROM sch_pipeline_trip
            WHERE trip_id = %s
            """,
            (trip_id,)
        )
        row = cur.fetchone()
    
    if not row or not row[0]:
        return None
    
    return load_shape_from_gtfs(row[0], conn)


# ==================== Usage Example ====================

def example_usage():
    """
    Example showing how to use shape-informed features in practice.
    """
    # 1. Load shape once per trip (cache this!)
    from psycopg2 import connect
    conn = connect("postgresql://user:pass@localhost/gtfs")
    
    shape = load_shape_for_trip("trip_123", conn)
    
    # 2. For each vehicle position update
    vehicle_position = {
        'lat': 42.3601,
        'lon': -71.0589,
        'bearing': 180.0
    }
    
    current_stop = {
        'stop_id': 'stop_A',
        'lat': 42.3598,
        'lon': -71.0592
    }
    
    next_stop = {
        'stop_id': 'stop_B', 
        'lat': 42.3620,
        'lon': -71.0580
    }
    
    # 3. Get shape-informed features
    features = calculate_distance_features_with_shape(
        vehicle_position,
        current_stop,
        next_stop,
        shape=shape
    )
    
    print(f"Shape progress: {features['shape_progress']:.2%}")
    print(f"Distance to stop (along shape): {features['shape_distance_to_stop']:.0f}m")
    print(f"Cross-track error: {features['cross_track_error']:.1f}m")
    print(f"Segment progress: {features['progress_on_segment']:.2%}")

if __name__ == "__main__":
    example_usage()
