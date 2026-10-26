"""All-sensor 1D CNN + GRU baseline for gesture classification.

Inputs are the preprocessed ``ready_full`` arrays:

* ``x`` has shape ``(batch, time, 346)``.
* ``mask`` has shape ``(batch, time)`` with left padding marked as zero.

The model predicts one of the 18 gesture labels. It deliberately treats each
ToF pixel as a temporal feature; a spatial ToF encoder is a later experiment.
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class TemporalConvStem(nn.Module):
    """Extract local temporal patterns from one sensor group."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # Zero padded positions after each temporal convolution so they cannot
        # leak into nearby real time steps through the next convolution.
        hidden = self.activation(self.conv1(x)) * valid
        hidden = self.conv2(hidden) + self.skip(x)
        return self.dropout(self.activation(hidden)) * valid


class FullSensorCNNGRU(nn.Module):
    """Classify full-sensor sequences with three CNN stems and a bidirectional GRU.

    Feature layout, from ``stats_all.npz``:
        0:16    motion and engineered IMU features
        16:21   thermopile readings
        21:341  five flattened 8x8 ToF grids
        341:346 missing-data indicators
    """

    def __init__(
        self,
        num_classes: int = 18,
        gru_hidden_size: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.motion_stem = TemporalConvStem(16, 64)
        self.thermal_stem = TemporalConvStem(5, 32)
        self.tof_stem = TemporalConvStem(320, 96)

        self.fusion = nn.Sequential(
            nn.Conv1d(64 + 32 + 96 + 5, 192, kernel_size=1),
            nn.SiLU(),
        )
        self.gru = nn.GRU(
            input_size=192,
            hidden_size=gru_hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * gru_hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return gesture logits with shape ``(batch, num_classes)``.

        ``pack_padded_sequence`` expects right padding. The preprocessed data
        is left padded, so valid time steps are shifted to the front first.
        """
        if x.ndim != 3 or x.shape[-1] != 346:
            raise ValueError(f"x must have shape (batch, time, 346), got {tuple(x.shape)}")
        if mask.shape != x.shape[:2]:
            raise ValueError(f"mask must have shape {tuple(x.shape[:2])}, got {tuple(mask.shape)}")

        batch_size, time_steps, feature_count = x.shape
        lengths = mask.sum(dim=1).to(dtype=torch.long)

        # For a sequence of length L, gather positions [T-L, ..., T-1] first.
        # Remaining positions are explicitly zeroed to form right padding.
        time = torch.arange(time_steps, device=x.device).unsqueeze(0)
        source = (time_steps - lengths.unsqueeze(1) + time).clamp(max=time_steps - 1)
        x = x.gather(
            dim=1,
            index=source.unsqueeze(-1).expand(batch_size, time_steps, feature_count),
        )
        valid = time < lengths.unsqueeze(1)
        x = x * valid.unsqueeze(-1).to(dtype=x.dtype)
        valid_channels = valid.unsqueeze(1).to(dtype=x.dtype)

        motion = self.motion_stem(x[:, :, :16].transpose(1, 2), valid_channels)
        thermal = self.thermal_stem(x[:, :, 16:21].transpose(1, 2), valid_channels)
        tof = self.tof_stem(x[:, :, 21:341].transpose(1, 2), valid_channels)
        flags = x[:, :, 341:346].transpose(1, 2)

        features = self.fusion(torch.cat((motion, thermal, tof, flags), dim=1))
        features = features.transpose(1, 2)
        packed = pack_padded_sequence(
            features,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)

        sequence_features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(sequence_features)
