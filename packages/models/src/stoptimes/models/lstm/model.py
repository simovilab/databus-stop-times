"""
Segment-sequence recurrent model (roadmap 5.2).

`SegmentSequenceModel` is a **unidirectional, causal** LSTM/GRU with a
many-to-many linear head: it emits one ETA (time_to_arrival_seconds)
prediction per segment/timestep. "Causal" here is a correctness
requirement, not just a style choice -- at serving time the model only ever
has observed segments 0..t of the *current* trip, never segments beyond the
vehicle's current position. A bidirectional RNN would let the prediction at
step t depend on future segments (t+1, t+2, ...), which do not exist yet at
inference time and would make training/serving skew silently. Accordingly:

- `bidirectional=False` always, for both `nn.LSTM` and `nn.GRU`.
- Padding is handled via `pack_padded_sequence`/`pad_packed_sequence` so the
  recurrent state never mixes information across padded positions or across
  sequences in a batch.

See `models/tests/test_lstm.py::test_causality_*` for the enforcement test:
perturbing the input at step t must change predictions at steps >= t only,
never at steps < t.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SegmentSequenceModel(nn.Module):
    """Unidirectional LSTM/GRU over segment feature sequences, many-to-many.

    Args:
        n_features: number of input features per timestep.
        hidden_size: RNN hidden state size.
        num_layers: number of stacked recurrent layers.
        dropout: inter-layer dropout (only applied when num_layers > 1, per
            torch's RNN semantics).
        rnn_type: "lstm" or "gru".
        use_layer_norm: apply LayerNorm to the input features before the RNN
            (helps with the mixed-scale feature set even after StandardScaler).
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        rnn_type: str = "lstm",
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if rnn_type not in ("lstm", "gru"):
            raise ValueError(f"rnn_type must be 'lstm' or 'gru', got {rnn_type!r}")

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.use_layer_norm = use_layer_norm

        self.input_norm = nn.LayerNorm(n_features) if use_layer_norm else nn.Identity()

        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,  # CAUSAL -- do not change. See module docstring.
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, max_len, n_features) float tensor, zero-padded.
            lengths: (batch,) int64 tensor of true (unpadded) sequence lengths.

        Returns:
            (batch, max_len) tensor of per-timestep predictions. Values at
            padded positions are defined (the RNN still emits something
            there) but are meaningless and must be masked out by the caller
            -- they carry zero weight in the loss/metrics by construction.
        """
        x = self.input_norm(x)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.size(1)
        )
        preds = self.head(out).squeeze(-1)
        return preds


def masked_loss(
    preds: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss: str = "huber",
) -> torch.Tensor:
    """Masked Huber (SmoothL1) or MSE loss. Padded positions (mask=False)
    contribute exactly zero -- both to the per-element loss (zeroed out) and
    to the denominator (mask.sum(), not numel())."""
    if loss == "huber":
        criterion = nn.SmoothL1Loss(reduction="none")
    elif loss == "mse":
        criterion = nn.MSELoss(reduction="none")
    else:
        raise ValueError(f"loss must be 'huber' or 'mse', got {loss!r}")

    mask_f = mask.float()
    per_elem = criterion(preds, target) * mask_f
    denom = mask_f.sum().clamp(min=1.0)
    return per_elem.sum() / denom
