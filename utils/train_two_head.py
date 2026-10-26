"""Training and validation helpers for the two-head ToF CNN + GRU model."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from utils.evaluation import competition_score


def two_head_loss(binary_logits, gesture_logits, y, gesture_weight=1.0):
    """Combine binary loss with eight-class loss on BFRB examples only.

    Local gesture IDs 0-7 are BFRB and 8-17 are non-BFRB. The binary head
    learns from every example; the gesture head has no target for non-BFRB
    examples. ``gesture_weight`` controls the gesture loss contribution.
    """
    is_bfrb = y < 8
    binary_loss = F.binary_cross_entropy_with_logits(
        binary_logits, is_bfrb.to(dtype=binary_logits.dtype)
    )
    if is_bfrb.any():
        gesture_loss = F.cross_entropy(gesture_logits[is_bfrb], y[is_bfrb])
    else:
        # Keep a differentiable zero when a batch contains only non-BFRBs.
        gesture_loss = gesture_logits.sum() * 0.0
    return binary_loss + gesture_weight * gesture_loss


def train_two_head_one_epoch(
    model, loader, optimizer, criterion, device, grad_clip_norm=1.0
):
    """Train for one epoch and return mean combined loss per sequence."""
    model.train()
    total_loss = 0.0

    for x, mask, y in loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        binary_logits, gesture_logits = model(x, mask)
        loss = criterion(binary_logits, gesture_logits, y)
        loss.backward()
        if grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        total_loss += loss.item() * len(y)

    return total_loss / len(loader.dataset)


@torch.inference_mode()
def validate_two_head(
    model, loader, criterion, device, threshold=0.5, non_target_class=8
):
    """Return loss, metrics, labels, predictions, and both heads' raw logits.

    The first four return values match ``utils.train.validate``. The final
    two are CPU tensors of shapes ``(n,)`` and ``(n, 8)`` so thresholds can
    be examined later without rerunning inference.
    """
    model.eval()
    total_loss = 0.0
    true_parts, binary_parts, gesture_parts = [], [], []

    for x, mask, y in loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        binary_logits, gesture_logits = model(x, mask)
        total_loss += criterion(binary_logits, gesture_logits, y).item() * len(y)
        true_parts.append(y.cpu().numpy())
        binary_parts.append(binary_logits.cpu())
        gesture_parts.append(gesture_logits.cpu())

    y_true = np.concatenate(true_parts)
    all_binary_logits = torch.cat(binary_parts)
    all_gesture_logits = torch.cat(gesture_parts)
    y_pred = model.decode_logits(
        all_binary_logits,
        all_gesture_logits,
        threshold=threshold,
        non_target_class=non_target_class,
    ).numpy()
    metrics = competition_score(y_true, y_pred)
    return (
        total_loss / len(loader.dataset),
        metrics,
        y_true,
        y_pred,
        all_binary_logits,
        all_gesture_logits,
    )
