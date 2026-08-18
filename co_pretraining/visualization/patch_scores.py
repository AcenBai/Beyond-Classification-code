from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def minmax01(scores: torch.Tensor) -> torch.Tensor:
    scores = scores.float()
    return (scores - scores.min()) / (scores.max() - scores.min()).clamp_min(1e-8)


def mask_grid_coverage(mask_2d: torch.Tensor, grid_h: int, grid_w: int) -> np.ndarray:
    mask = mask_2d.float()
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={tuple(mask.shape)}")
    covered = F.interpolate(mask[None, None], size=(grid_h, grid_w), mode="area").squeeze()
    return covered.cpu().numpy().astype(np.float32)


def split_foreground_background_scores(
    scores: torch.Tensor,
    mask_2d: torch.Tensor,
    majority_ratio: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    side = int(round(scores.numel() ** 0.5))
    grid = minmax01(scores.reshape(side, side)).detach().cpu()
    coverage = mask_grid_coverage(mask_2d, side, side)
    fg = grid.numpy()[coverage > majority_ratio]
    bg = grid.numpy()[coverage <= majority_ratio]
    return fg.astype(np.float32), bg.astype(np.float32)


def save_score_distribution(
    foreground_scores: np.ndarray,
    background_scores: np.ndarray,
    out_png: str | Path,
    title: str = "Patch Score Distribution",
) -> dict[str, Any]:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    bins = np.linspace(0.0, 1.0, 80)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.hist(background_scores, bins=bins, density=True, alpha=0.55, label="background", color="#4C78A8")
    ax.hist(foreground_scores, bins=bins, density=True, alpha=0.55, label="foreground", color="#F58518")
    ax.set_xlabel("min-max normalized patch score")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return {
        "num_foreground_scores": int(foreground_scores.size),
        "num_background_scores": int(background_scores.size),
        "foreground_mean": float(foreground_scores.mean()) if foreground_scores.size else None,
        "background_mean": float(background_scores.mean()) if background_scores.size else None,
        "figure": str(out_png),
    }


def save_heatmap_overlay(
    image_path: str | Path,
    scores: torch.Tensor,
    out_png: str | Path,
    alpha: float = 0.45,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(str(image_path))
    h, w = image.shape[:2]
    side = int(round(scores.numel() ** 0.5))
    grid = minmax01(scores.reshape(side, side)).detach().cpu().numpy()
    heat = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
    heat_u8 = np.uint8(np.clip(heat, 0.0, 1.0) * 255)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 1.0 - alpha, heat_color, alpha, 0)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), overlay)
