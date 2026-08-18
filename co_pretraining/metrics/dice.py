from __future__ import annotations

import torch


@torch.no_grad()
def dice_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).to(torch.float32)
    targets = targets.to(torch.float32)
    dims = tuple(range(1, preds.ndim))
    pred_sum = preds.sum(dim=dims)
    target_sum = targets.sum(dim=dims)
    inter = (preds * targets).sum(dim=dims)

    dice = torch.zeros_like(pred_sum)
    empty_gt = target_sum == 0
    empty_pred = pred_sum == 0
    dice[empty_gt & empty_pred] = 1.0
    dice[empty_gt & ~empty_pred] = 0.0
    non_empty = ~empty_gt
    dice[non_empty] = (2 * inter[non_empty]) / (pred_sum[non_empty] + target_sum[non_empty]).clamp_min(1e-6)

    labels_cpu = labels.detach().cpu()
    lesion = labels_cpu != 2
    normal = labels_cpu == 2
    out = {"dice_all": float(dice.mean().item())}
    out["dice_lesion"] = float(dice[lesion.to(dice.device)].mean().item()) if lesion.any() else float("nan")
    out["dice_normal"] = float(dice[normal.to(dice.device)].mean().item()) if normal.any() else float("nan")
    return out
