"""ConstantSpeedPredictor (predictor_id="baseline:constant_speed"). Phase P3.

etaval's own white-box baseline (zero external deps). speed = dprogress_m/dt from
position history (fallback to reported speed); predicted arrival = now +
dist_remaining_m / speed per upcoming stop.
"""
