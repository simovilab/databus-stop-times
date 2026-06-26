"""Pluggable interfaces (typing.Protocol). Phase P0.

- Predictor.predict(trip, upcoming_stops, vehicle_state, trip_updates) -> list[StopPrediction]
- GroundTruthDetector.detect(trip, position_history, stops) -> list[ActualArrival]
- FeedAdapter.poll() -> parsed entities
- Storage.save_samples / save_arrivals / save_results / load_results
"""
