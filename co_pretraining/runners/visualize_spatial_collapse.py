#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from co_pretraining.data import BUSIManifestDataset
from co_pretraining.models.spatial_collapse import DenseViTWithPatchScores, build_spatial_collapse_model
from co_pretraining.paths import default_manifest_dir
from co_pretraining.visualization.patch_scores import (
    save_heatmap_overlay,
    save_score_distribution,
    split_foreground_background_scores,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize patch scores for ViT and ResNet models.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--backbone", choices=["vit", "dense", "dense_inv", "resnet50"], required=True)
    p.add_argument("--manifest-dir", default=str(default_manifest_dir()))
    p.add_argument("--output-dir", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--patch-size", type=int, choices=[16, 32], default=16)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-heatmaps", type=int, default=24)
    return p.parse_args()


def make_loader(path: Path, image_size: int, batch_size: int, workers: int) -> DataLoader:
    ds = BUSIManifestDataset(
        path,
        image_size=image_size,
        mask_size=image_size,
        normalize_mean=(0.485, 0.456, 0.406),
        normalize_std=(0.229, 0.224, 0.225),
        image_backend="pil_rgb",
    )
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }
    if workers > 0:
        kwargs["multiprocessing_context"] = "fork"
    return DataLoader(ds, **kwargs)


def load_state(path: str) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(ckpt)!r}")
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        clean = key.removeprefix("module.").removeprefix("model.")
        out[clean] = value
    return out


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    heat_dir = out_dir / "heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = build_spatial_collapse_model(
        backbone=args.backbone,
        pretrained=False,
        image_size=args.image_size,
        patch_size=args.patch_size,
    )
    model.load_state_dict(load_state(args.checkpoint), strict=False)
    model.eval().to(device)

    use_votes = isinstance(model, DenseViTWithPatchScores)
    ranking_source = "dense_vote_count" if use_votes else "patch_score"
    all_fg: list[np.ndarray] = []
    all_bg: list[np.ndarray] = []
    heatmaps: list[dict[str, Any]] = []
    manifest_dir = Path(args.manifest_dir)

    for split in ("train", "val", "test"):
        loader = make_loader(manifest_dir / f"{split}.json", args.image_size, args.batch_size, args.num_workers)
        for batch in tqdm(loader, desc=f"visualize {split}", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"]
            masks = batch["mask_2d"]
            ids = batch["id"]
            img_files = batch["img_file"]
            logits, patch_scores = model(images)
            ranking_scores = model.vote_counts(images) if use_votes else patch_scores
            ranking_scores = ranking_scores.detach().cpu()
            for i in range(images.shape[0]):
                if int(labels[i].item()) == 2:
                    continue
                scores = ranking_scores[i]
                fg, bg = split_foreground_background_scores(scores, masks[i])
                if fg.size:
                    all_fg.append(fg)
                if bg.size:
                    all_bg.append(bg)
                if len(heatmaps) < args.max_heatmaps:
                    safe_id = str(ids[i]).replace("/", "_")
                    out_png = heat_dir / f"{split}_{safe_id}_{ranking_source}.png"
                    save_heatmap_overlay(img_files[i], scores, out_png)
                    heatmaps.append({"id": ids[i], "split": split, "label": int(labels[i].item()), "path": str(out_png)})

    fg_scores = np.concatenate(all_fg) if all_fg else np.array([], dtype=np.float32)
    bg_scores = np.concatenate(all_bg) if all_bg else np.array([], dtype=np.float32)
    summary = save_score_distribution(
        fg_scores,
        bg_scores,
        out_dir / f"{ranking_source}_distribution.png",
        title=f"{args.backbone} {ranking_source}",
    )
    summary.update(
        {
            "checkpoint": args.checkpoint,
            "backbone": args.backbone,
            "ranking_source": ranking_source,
            "heatmaps": heatmaps,
            "note": "Dense variants use channel-wise top-1 vote counts; ViT and ResNet use patch cosine scores.",
        }
    )
    (out_dir / "visualization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
