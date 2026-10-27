"""Full-sensor model with a temporal encoder for each feature group.

Accepts the existing ``ready_full`` input: left-padded ``(batch, time, 346)``
features and a ``(batch, time)`` validity mask. The output is 18 gesture
logits, so the existing single-head training and evaluation helpers can be
used without changing their interface.
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model.tof_cnn_gru import ToFSpatialEncoder


class SqueezeExcitation1d(nn.Module):
    """Weight temporal feature channels using a validity-aware mean."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.gate = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        counts = valid.sum(dim=-1).clamp_min(1)
        summary = (x * valid).sum(dim=-1) / counts
        return x * self.gate(summary).unsqueeze(-1)


class ResidualTemporalBlock(nn.Module):
    """Two temporal convolutions, channel attention, a skip, and pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd to preserve time length")

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=kernel_size // 2, bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.channel_attention = SqueezeExcitation1d(out_channels)
        self.shortcut = (
            nn.Identity() if in_channels == out_channels
            else nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2, ceil_mode=True)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x * valid
        hidden = self.activation(self.bn1(self.conv1(x))) * valid
        hidden = self.bn2(self.conv2(hidden)) * valid
        hidden = self.channel_attention(hidden, valid)
        hidden = self.activation(hidden + self.shortcut(x)) * valid

        pooled_valid = self.pool(valid) > 0
        hidden = self.pool(hidden)
        hidden = self.dropout(hidden) * pooled_valid
        return hidden, pooled_valid


class TemporalCNNGRUAttention(nn.Module):
    """Summarize one feature group's time series into one vector per trial."""

    def __init__(
        self,
        in_features: int,
        widths: tuple[int, int] = (64, 96),
        embedding_size: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.block1 = ResidualTemporalBlock(
            in_features, widths[0], dropout=dropout
        )
        self.block2 = ResidualTemporalBlock(
            widths[0], widths[1], dropout=dropout
        )
        self.gru = nn.GRU(
            input_size=widths[1], hidden_size=widths[1],
            bidirectional=True, batch_first=True,
        )
        self.temporal_dropout = nn.Dropout(dropout)
        self.attention = nn.Linear(2 * widths[1], 1)
        self.projection = nn.Linear(2 * widths[1], embedding_size)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, features); valid: (batch, time).
        valid = valid.unsqueeze(1).to(dtype=x.dtype)
        x, valid = self.block1(x.transpose(1, 2), valid)
        x, valid = self.block2(x, valid.to(dtype=x.dtype))
        valid = valid.squeeze(1).bool()
        lengths = valid.sum(dim=1).long()

        packed = pack_padded_sequence(
            x.transpose(1, 2), lengths.cpu(), batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.gru(packed)
        sequence, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=x.shape[-1]
        )
        sequence = self.temporal_dropout(sequence)

        attention_logits = self.attention(sequence).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~valid, -torch.inf)
        weights = torch.softmax(attention_logits, dim=1)
        pooled = (sequence * weights.unsqueeze(-1)).sum(dim=1)
        return self.projection(pooled)


class FullSensorGroupedCNNGRUAttention(nn.Module):
    """Encode motion groups, thermopiles, and ToF before late fusion.

    Feature indices match ``stats_all.npz`` and the existing ToF model:
    0:16 motion, 16:21 thermopiles, 21:341 ToF, 341:346 missingness flags.
    The motion groups are acceleration + magnitude (0:3, 10), quaternion
    (3:7), linear acceleration + magnitude (7:10, 11), and angular velocity
    + magnitude (12:16).
    """

    def __init__(
        self,
        num_classes: int = 18,
        widths: tuple[int, int] = (64, 96),
        embedding_size: int = 64,
        fusion_size: int = 192,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.motion_indices = {
            "acceleration": (0, 1, 2, 10),
            "rotation": (3, 4, 5, 6),
            "linear_acceleration": (7, 8, 9, 11),
            "angular_velocity": (12, 13, 14, 15),
        }
        self.motion_encoders = nn.ModuleDict({
            name: TemporalCNNGRUAttention(
                4, widths=widths, embedding_size=embedding_size,
                dropout=dropout,
            )
            for name in self.motion_indices
        })
        self.thermal_encoder = TemporalCNNGRUAttention(
            5, widths=widths, embedding_size=embedding_size,
            dropout=dropout,
        )
        self.tof_spatial = ToFSpatialEncoder()
        self.tof_temporal = TemporalCNNGRUAttention(
            96, widths=widths, embedding_size=embedding_size,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(6 * embedding_size + 5, fusion_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_size, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != 346:
            raise ValueError(f"x must have shape (batch, time, 346), got {tuple(x.shape)}")
        if mask.shape != x.shape[:2]:
            raise ValueError(f"mask must have shape {tuple(x.shape[:2])}, got {tuple(mask.shape)}")

        batch_size, time_steps, feature_count = x.shape
        lengths = mask.to(dtype=torch.bool).sum(dim=1).long()
        if torch.any(lengths == 0):
            raise ValueError("every sequence must contain at least one valid time step")

        # ready_full is left padded; packed GRUs need valid steps at the front.
        time = torch.arange(time_steps, device=x.device).unsqueeze(0)
        source = (time_steps - lengths.unsqueeze(1) + time).clamp(max=time_steps - 1)
        x = x.gather(
            1, source.unsqueeze(-1).expand(batch_size, time_steps, feature_count)
        )
        valid = time < lengths.unsqueeze(1)
        x = x * valid.unsqueeze(-1).to(dtype=x.dtype)

        embeddings = [
            self.motion_encoders[name](x[:, :, indices], valid)
            for name, indices in self.motion_indices.items()
        ]
        embeddings.append(self.thermal_encoder(x[:, :, 16:21], valid))

        grids = x[:, :, 21:341].reshape(batch_size * time_steps, 5, 8, 8)
        tof_features = self.tof_spatial(grids).reshape(batch_size, time_steps, 96)
        embeddings.append(self.tof_temporal(tof_features, valid))

        flags = x[:, :, 341:346].sum(dim=1) / lengths.unsqueeze(-1)
        return self.classifier(torch.cat((*embeddings, flags), dim=1))
