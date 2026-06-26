"""SQLite storage. Phase P0.

Three tables: prediction_samples, actual_arrivals, validation_results;
indexed by (trip_key, predictor_id). Implements the Storage protocol.
"""
