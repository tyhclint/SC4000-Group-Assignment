"""Two-head extension of the spatial ToF CNN + GRU model.

The shared encoder is identical to ``FullSensorToFCNNGRU``. One head predicts
whether a sequence is BFRB-like; the other scores the eight BFRB gestures.
Training must apply the gesture loss only to BFRB examples.
"""

import torch
from torch import nn

from model.tof_cnn_gru import FullSensorToFCNNGRU


class FullSensorToFTwoHeadCNNGRU(nn.Module):
    """Return binary logits and eight BFRB-gesture logits for each sequence.

    The input is the same ``(batch, time, 346)`` tensor and left-padding mask
    used by the single-head ToF model. The binary logit is positive for a
    BFRB-like gesture. Local label IDs 0-7 are BFRB and 8-17 are non-BFRB.
    """

    def __init__(self, gru_hidden_size: int = 128, dropout: float = 0.2):
        super().__init__()
        self.backbone = FullSensorToFCNNGRU(
            num_classes=18,
            gru_hidden_size=gru_hidden_size,
            dropout=dropout,
        )
        # The existing forward method ends with self.classifier(features).
        # Identity exposes those shared GRU features without copying its
        # padding, sensor-encoding, and fusion logic into this file.
        self.backbone.classifier = nn.Identity()

        feature_size = 2 * gru_hidden_size
        self.binary_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_size, 1),
        )
        self.gesture_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_size, 8),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return binary logits ``(batch,)`` and gesture logits ``(batch, 8)``.

        These are raw logits: use ``BCEWithLogitsLoss`` for the binary head and
        ``CrossEntropyLoss`` on BFRB examples for the gesture head.
        """
        features = self.backbone(x, mask)
        binary_logits = self.binary_head(features).squeeze(-1)
        gesture_logits = self.gesture_head(features)
        return binary_logits, gesture_logits

    @staticmethod
    def decode_logits(
        binary_logits: torch.Tensor,
        gesture_logits: torch.Tensor,
        threshold: float = 0.5,
        non_target_class: int = 8,
    ) -> torch.Tensor:
        """Convert the two heads into local gesture IDs for scoring.

        All non-BFRB IDs collapse to one class in the competition metric, so
        ``non_target_class`` may be any valid non-BFRB ID from 8 through 17.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if not 8 <= non_target_class <= 17:
            raise ValueError("non_target_class must be a valid ID from 8 to 17")
        if binary_logits.ndim != 1 or gesture_logits.shape != (binary_logits.shape[0], 8):
            raise ValueError("expected binary logits (batch,) and gesture logits (batch, 8)")

        is_bfrb = torch.sigmoid(binary_logits) >= threshold
        bfrb_class = gesture_logits.argmax(dim=1)
        non_bfrb_class = torch.full_like(bfrb_class, non_target_class)
        return torch.where(is_bfrb, bfrb_class, non_bfrb_class)
