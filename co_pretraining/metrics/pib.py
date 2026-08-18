from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def infer_grid_hw(num_patches: int) -> tuple[int, int]:
    side = int(round(math.sqrt(num_patches)))
    if side * side != num_patches:
        raise ValueError(f"Expected square patch grid, got num_patches={num_patches}")
    return side, side


def bbox_to_grid_index_set(
    bbox: list[int] | tuple[int, ...],
    image_h: int,
    image_w: int,
    grid_h: int,
    grid_w: int,
) -> set[int]:
    x0, y0, x1, y1 = bbox
    cell_h = float(image_h) / float(grid_h)
    cell_w = float(image_w) / float(grid_w)
    if cell_h <= 0 or cell_w <= 0:
        return set()
    gx0 = int(math.floor(float(x0) / cell_w))
    gx1 = int(math.floor(float(x1) / cell_w))
    gy0 = int(math.floor(float(y0) / cell_h))
    gy1 = int(math.floor(float(y1) / cell_h))
    gx0 = int(np.clip(gx0, 0, grid_w - 1))
    gx1 = int(np.clip(gx1, 0, grid_w - 1))
    gy0 = int(np.clip(gy0, 0, grid_h - 1))
    gy1 = int(np.clip(gy1, 0, grid_h - 1))
    if gx0 > gx1 or gy0 > gy1:
        return set()
    return {py * grid_w + px for py in range(gy0, gy1 + 1) for px in range(gx0, gx1 + 1)}


def pib_topk_in_bbox_grid_set(
    patch_scores: torch.Tensor,
    bbox: list[int] | tuple[int, ...],
    mask_2d: torch.Tensor,
    label: int,
    topk: int = 1,
) -> float | None:
    if label == 2:
        return None
    num_patches = int(patch_scores.shape[0])
    grid_h, grid_w = infer_grid_hw(num_patches)
    image_h, image_w = int(mask_2d.shape[0]), int(mask_2d.shape[1])
    bbox_set = bbox_to_grid_index_set(
        list(map(int, bbox)),
        image_h=image_h,
        image_w=image_w,
        grid_h=grid_h,
        grid_w=grid_w,
    )
    if not bbox_set:
        return None
    k = min(max(1, int(topk)), num_patches)
    top_flat = torch.topk(patch_scores.reshape(-1), k=k, largest=True).indices.tolist()
    return float(any(int(idx) in bbox_set for idx in top_flat))


def pib_topk_fraction_in_bbox(
    patch_scores: torch.Tensor,
    bbox: list[int] | tuple[int, ...],
    mask_2d: torch.Tensor,
    label: int,
    topk: int = 1,
) -> float | None:
    if label == 2:
        return None
    num_patches = int(patch_scores.shape[0])
    grid_h, grid_w = infer_grid_hw(num_patches)
    image_h, image_w = int(mask_2d.shape[0]), int(mask_2d.shape[1])
    bbox_set = bbox_to_grid_index_set(
        list(map(int, bbox)),
        image_h=image_h,
        image_w=image_w,
        grid_h=grid_h,
        grid_w=grid_w,
    )
    if not bbox_set:
        return None
    k = min(max(1, int(topk)), num_patches)
    top_flat = torch.topk(patch_scores.reshape(-1), k=k, largest=True).indices.tolist()
    return float(sum(int(idx) in bbox_set for idx in top_flat) / k)


def summarize_pib(hits: list[float | None], topk: int = 1) -> dict[str, Any]:
    usable = [h for h in hits if h is not None]
    rule = "top1_in_bbox_grid" if int(topk) <= 1 else f"top{int(topk)}_in_bbox_grid"
    if not usable:
        return {"pib": float("nan"), "pib_count": 0, "pib_rule": rule, "pib_topk": int(topk)}
    return {
        "pib": float(sum(usable) / len(usable)),
        "pib_count": len(usable),
        "pib_rule": rule,
        "pib_topk": int(topk),
    }
