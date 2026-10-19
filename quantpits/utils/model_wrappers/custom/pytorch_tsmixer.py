"""A small Qlib-compatible TSMixer backbone.

The class in this module is intentionally only the PyTorch model.  It is
constructed by Qlib's ``GeneralPTNN`` through ``pt_model_uri`` and therefore
expects a time-series tensor with shape ``[batch, seq_len, d_feat]``.
"""

from __future__ import annotations

import numbers

import torch
from torch import nn


def _validate_positive_int(name: str, value: object) -> int:
    """Validate and return a positive integer constructor argument."""

    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _validate_dropout(value: object) -> float:
    """Validate and return a PyTorch dropout probability."""

    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"dropout must be a number in [0, 1), got {value!r}")
    value = float(value)
    if not 0.0 <= value < 1.0:
        raise ValueError(f"dropout must be a number in [0, 1), got {value!r}")
    return value


class _TSMixerBlock(nn.Module):
    """One pre-normalized temporal and feature mixing residual block."""

    def __init__(
        self,
        seq_len: int,
        d_feat: int,
        hidden_time: int,
        hidden_feature: int,
        dropout: float,
    ) -> None:
        super().__init__()

        # LayerNorm over the feature axis is valid for both mixing stages.  The
        # temporal MLP is applied after transposing to [batch, feature, time],
        # so each feature gets an independent projection over the fixed window.
        self.norm_time = nn.LayerNorm(d_feat)
        self.time_mixing = nn.Sequential(
            nn.Linear(seq_len, hidden_time),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_time, seq_len),
            nn.Dropout(dropout),
        )

        self.norm_feature = nn.LayerNorm(d_feat)
        self.feature_mixing = nn.Sequential(
            nn.Linear(d_feat, hidden_feature),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_feature, d_feat),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal mixing followed by feature mixing."""

        residual = self.norm_time(x).transpose(1, 2)
        x = x + self.time_mixing(residual).transpose(1, 2)
        x = x + self.feature_mixing(self.norm_feature(x))
        return x


class TSMixer(nn.Module):
    """TSMixer backbone for Qlib's ``GeneralPTNN``.

    Parameters
    ----------
    d_feat : int, default 158
        Number of Alpha158 features at every timestep.
    seq_len : int, default 20
        Number of timesteps in each ``TSDatasetH`` sample.  Temporal mixing
        uses this fixed length and inputs with a different length are rejected.
    hidden_time : int, default 64
        Hidden width of the temporal mixing MLP.
    hidden_feature : int, default 256
        Hidden width of the feature mixing MLP.
    n_blocks : int, default 2
        Number of temporal/feature residual mixing blocks.
    dropout : float, default 0.1
        Dropout probability used inside both MLPs.

    Returns
    -------
    torch.Tensor
        A score tensor with shape ``[batch, 1]``.  The last timestep is used
        as the forecasting representation.
    """

    def __init__(
        self,
        d_feat: int = 158,
        seq_len: int = 20,
        hidden_time: int = 64,
        hidden_feature: int = 256,
        n_blocks: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.d_feat = _validate_positive_int("d_feat", d_feat)
        self.seq_len = _validate_positive_int("seq_len", seq_len)
        self.hidden_time = _validate_positive_int("hidden_time", hidden_time)
        self.hidden_feature = _validate_positive_int(
            "hidden_feature", hidden_feature
        )
        self.n_blocks = _validate_positive_int("n_blocks", n_blocks)
        self.dropout = _validate_dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                _TSMixerBlock(
                    seq_len=self.seq_len,
                    d_feat=self.d_feat,
                    hidden_time=self.hidden_time,
                    hidden_feature=self.hidden_feature,
                    dropout=self.dropout,
                )
                for _ in range(self.n_blocks)
            ]
        )
        self.head = nn.Linear(self.d_feat, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one prediction per sample for ``x``.

        ``GeneralPTNN`` supplies a three-dimensional TSDatasetH batch.  The
        explicit checks here turn shape mistakes into actionable errors rather
        than obscure Linear/LayerNorm exceptions.
        """

        if not isinstance(x, torch.Tensor):
            raise ValueError(
                "x must be a torch.Tensor with shape "
                "[batch, seq_len, d_feat]"
            )
        if x.ndim != 3:
            raise ValueError(
                "x must have shape [batch, seq_len, d_feat], "
                f"got {tuple(x.shape)}"
            )
        if x.shape[0] <= 0:
            raise ValueError(
                "x batch dimension must be positive, "
                f"got {tuple(x.shape)}"
            )
        if x.shape[1] != self.seq_len:
            raise ValueError(
                f"x seq_len must be {self.seq_len}, got {x.shape[1]}"
            )
        if x.shape[2] != self.d_feat:
            raise ValueError(
                f"x d_feat must be {self.d_feat}, got {x.shape[2]}"
            )

        for block in self.blocks:
            x = block(x)

        # Indexing the timestep (rather than squeeze()) preserves [batch, 1]
        # when GeneralPTNN evaluates a single-sample batch.
        return self.head(x[:, -1, :])


__all__ = ["TSMixer"]
