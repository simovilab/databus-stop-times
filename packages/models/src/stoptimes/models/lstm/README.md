# LSTM/GRU segment-sequence model (roadmap 5.2)

## What this is

An alternative to the row-independent XGBoost model (`models/xgb`): instead
of treating each (trip instance × segment) row as an independent sample,
this model treats **one trip instance as a sequence** of its segments,
ordered by position along the trip, and predicts `time_to_arrival_seconds`
at every segment with a single causal, unidirectional recurrent network
(`SegmentSequenceModel` in `model.py`, LSTM or GRU, selectable). The
hypothesis: a trip's own recent segment history (speed, dwell, progress)
carries signal an i.i.d. row model can't see.

- `data.py` — groups rows into sequences keyed `(trip_id, service_date)`
  (`service_date` folds a `vp_ts` back onto the service day it belongs to, so
  cross-midnight trips and trip_ids reused across days resolve correctly),
  orders rows within a sequence by `segment_index`, fits/applies a
  train-only `StandardScaler` + median imputation, and pads sequences into
  batch tensors.
- `model.py` — the network itself. **Unidirectional only** — this is a
  correctness requirement, not a style choice: at serving time the model
  only ever has segments 0..t of the trip so far, never the future ones a
  bidirectional RNN would peek at.
- `train.py` — `train_lstm(...)`, mirroring `xgb.train.train_xgboost`'s
  contract (same `pre_split`/`save_model`/metric-key conventions), plus
  `LSTMSegmentModel`, a picklable wrapper (state_dict + config + scaler,
  not a live `nn.Module`) with a `predict(df)` method for serving/validation.

## Why `.venv-dl`

Training needs `torch`, which is **not** in the project's default
environment — a ~200 MB wheel that only this trainer and its tests need.
It's pinned to `torch>=2.2,<2.3` because torch dropped macOS x86_64 wheels
after 2.2.2, and 2.2.2 requires CPython ≤3.12. `.venv-dl` is a dedicated
Python 3.12 venv with that pin plus the rest of the training stack
(pandas, numpy<2, sklearn, xgboost, scipy, optuna, pytest). Always invoke it
directly — never `uv run` (which would mutate the shared default venv).

## Training

```bash
PYTHONPATH=.:models /path/to/.venv-dl/bin/python -c "
from lstm.train import train_lstm
train_lstm(dataset_name='mbta_segments_corpus', hidden_size=64, num_layers=1, max_epochs=30)
"
```

or, from a script/REPL with a pre-built `(train_df, val_df, test_df)` split
(e.g. the day-grouped split in `scratchpad/hpo_xgboost.py::day_grouped_split`):

```python
from lstm.train import train_lstm
out = train_lstm(pre_split=(train_df, val_df, test_df), hidden_size=64,
                  num_layers=1, max_epochs=30, service_offset_h=7, save_model=False)
print(out["metrics"]["test_mae_seconds"], out["metrics"]["test_r2"])
```

## Tests

```bash
PYTHONPATH=.:models .venv-dl/bin/python -m pytest models/tests/test_lstm.py -q
```

Synthetic data only, no real corpus needed — runs in well under a minute.
