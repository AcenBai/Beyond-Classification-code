from __future__ import annotations

import torch


class LesionAwareSegLoss(torch.nn.Module):
    """BCE and soft Dice on lesion images, with a small foreground penalty on normal images."""

    def __init__(self, normal_empty_weight: float = 0.1) -> None:
        super().__init__()
        self.normal_empty_weight = float(normal_empty_weight)
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.ndim))
        bce_per_sample = self.bce(logits, targets).mean(dim=dims)
        inter = (probs * targets).sum(dim=dims)
        denom = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice_per_sample = 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0))

        lesion_mask = labels != 2
        losses = []
        if lesion_mask.any():
            losses.append((bce_per_sample[lesion_mask] + dice_per_sample[lesion_mask]).mean())
        normal_mask = ~lesion_mask
        if normal_mask.any() and self.normal_empty_weight > 0:
            empty_penalty = probs[normal_mask].mean(dim=dims)
            losses.append(self.normal_empty_weight * empty_penalty.mean())
        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).sum()
