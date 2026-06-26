"""Engine runtime (asyncio tick loop). Phase P5.

Every poll_interval_s: poll VP+TU, update tracker + geometry, run all predictors
and detectors per active trip, run the scorer for newly-resolved stops, persist
samples/arrivals/results. Runs for a fixed --duration. Supports replay mode for tests.
"""
