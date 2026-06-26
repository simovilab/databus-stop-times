"""GTFS static .zip loader (stdlib zipfile + pandas). Phase P1.

Exposes stops, trips, stop_times, shapes, routes; provides
trip_geometry(trip_id) -> (shape points, ordered stops).
"""
