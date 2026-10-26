"""Full-sensor CNN + GRU with a spatial encoder for the ToF grids.

Uses the same ``ready_full`` input and 18-class output as the baseline. The
only model branch changed is ToF: each time step's 320 ToF values become five
8x8 grids before temporal processing.
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from model.baseline import TemporalConvStem


class ToFSpatialEncoder(nn.Module):
    """Encode five 8x8 ToF grids at one time step into 96 features."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
        )

    def forward(self, grids: torch.Tensor) -> torch.Tensor:
        """Accept ``(batch * time, 5, 8, 8)`` and return ``(batch * time, 96)``."""
        return self.layers(grids)


class FullSensorToFCNNGRU(nn.Module):
    """Classify sequences using spatial ToF features and a bidirectional GRU.

    Feature layout matches ``FullSensorCNNGRU`` from ``model.baseline``:
    0:16 motion, 16:21 thermopile, 21:341 ToF, 341:346 missingness flags.
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
        self.tof_spatial = ToFSpatialEncoder()
        self.tof_temporal = TemporalConvStem(96, 96)

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
        """Return gesture logits with shape ``(batch, num_classes)``."""
        if x.ndim != 3 or x.shape[-1] != 346:
            raise ValueError(f"x must have shape (batch, time, 346), got {tuple(x.shape)}")
        if mask.shape != x.shape[:2]:
            raise ValueError(f"mask must have shape {tuple(x.shape[:2])}, got {tuple(mask.shape)}")

        batch_size, time_steps, feature_count = x.shape
        lengths = mask.sum(dim=1).to(dtype=torch.long)

        # Match the baseline: move valid steps from left padding to the front,
        # then give the GRU right-padded sequences.
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

        # The ToF features are ordered by sensor, then row-major pixel index.
        grids = x[:, :, 21:341].reshape(batch_size * time_steps, 5, 8, 8)
        tof = self.tof_spatial(grids)
        tof = tof.reshape(batch_size, time_steps, 96).transpose(1, 2)
        tof = self.tof_temporal(tof * valid_channels, valid_channels)

        flags = x[:, :, 341:346].transpose(1, 2)
        features = self.fusion(torch.cat((motion, thermal, tof, flags), dim=1))
        packed = pack_padded_sequence(
            features.transpose(1, 2),
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        sequence_features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(sequence_features)
