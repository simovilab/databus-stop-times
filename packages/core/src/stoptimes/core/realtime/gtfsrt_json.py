"""GTFS-RT JSON feed adapters (httpx, async). Phase P1.

VehiclePositionsPoller + TripUpdatesPoller. Parse the standard GTFS-RT
FeedMessage (header.timestamp, entity[]). VP entity -> VehicleState;
TU entity -> stop_time_update[] arrival predictions.
"""
