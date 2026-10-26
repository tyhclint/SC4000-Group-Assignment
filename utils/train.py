"""Training and validation helpers for the 18-class gesture baseline."""

import numpy as np
import torch
from torch import nn

from utils.evaluation import competition_score


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip_norm=1.0):
    """Train for one epoch and return mean loss per sequence."""
    model.train()
    total_loss = 0.0

    for x, mask, y in loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x, mask)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        total_loss += loss.item() * len(y)

    return total_loss / len(loader.dataset)


@torch.inference_mode()
def validate(model, loader, criterion, device):
    """Return mean loss, competition metrics, true labels, and predictions."""
    model.eval()
    total_loss = 0.0
    true_parts, pred_parts = [], []

    for x, mask, y in loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x, mask)
        total_loss += criterion(logits, y).item() * len(y)
        true_parts.append(y.cpu().numpy())
        pred_parts.append(logits.argmax(dim=1).cpu().numpy())

    y_true = np.concatenate(true_parts)
    y_pred = np.concatenate(pred_parts)
    metrics = competition_score(y_true, y_pred)
    return total_loss / len(loader.dataset), metrics, y_true, y_pred
