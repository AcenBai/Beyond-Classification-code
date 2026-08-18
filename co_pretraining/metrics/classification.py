from __future__ import annotations

from typing import Any

import numpy as np
import torch


def classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 3,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    class_names = class_names or ["benign", "malignant", "normal"]
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    pred = probs.argmax(axis=1)
    y = labels.detach().cpu().numpy()
    out: dict[str, Any] = {"acc": float((pred == y).mean())}
    try:
        from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score

        out["f1_macro"] = float(f1_score(y, pred, average="macro", zero_division=0))
        out["f1_weighted"] = float(f1_score(y, pred, average="weighted", zero_division=0))
        if len(set(y.tolist())) > 1:
            out["auc_macro"] = float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))
        else:
            out["auc_macro"] = float("nan")
        p, r, f, s = precision_recall_fscore_support(
            y, pred, labels=list(range(num_classes)), zero_division=0
        )
        out["per_class"] = {
            class_names[i] if i < len(class_names) else str(i): {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
            for i in range(num_classes)
        }
    except Exception:
        out["f1_macro"] = float("nan")
        out["f1_weighted"] = float("nan")
        out["auc_macro"] = float("nan")
    return out


def format_classification_metrics(metrics: dict[str, Any]) -> str:
    return (
        f"acc={metrics.get('acc', float('nan')):.4f} "
        f"f1_macro={metrics.get('f1_macro', float('nan')):.4f} "
        f"f1_weighted={metrics.get('f1_weighted', float('nan')):.4f} "
        f"auc={metrics.get('auc_macro', metrics.get('auc', float('nan'))):.4f}"
    )
