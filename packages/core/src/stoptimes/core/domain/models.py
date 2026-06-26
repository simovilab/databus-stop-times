"""Domain model (Pydantic v2). Phase P0.

Defines the immutable data carried through the pipeline:
- SourceConfig    : agency feed URLs + poll interval + detector params (loaded from YAML)
- TripInstance    : the unit of validation, identified from VehiclePositions
- Stop            : (stop_id, stop_sequence, lat, lon, progress_m)
- VehicleState    : a single position-history sample for a TripInstance
- PredictionSample: one predictor x stop x tick arrival prediction
- ActualArrival   : a detected ground-truth arrival at a stop
- ValidationResult: derived once actual is known (error_s, horizon_s, ...)
"""
