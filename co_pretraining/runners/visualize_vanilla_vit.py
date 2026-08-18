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
from co_pretraining.models import VanillaViTB16Joint, patch_scores
from co_pretraining.paths import default_manifest_dir
from co_pretraining.visualization.patch_scores import (
    save_heatmap_overlay,
    save_score_distribution,
    split_foreground_background_scores,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Patch-score heatmaps for the jointly trained ViT.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest-dir", default=str(default_manifest_dir()))
    p.add_argument("--output-dir", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--query", choices=["cls", "mean"], default="cls")
    p.add_argument("--max-heatmaps", type=int, default=24)
    return p.parse_args()


def make_loader(path: Path, image_size: int, batch_size: int, workers: int) -> DataLoader:
    ds = BUSIManifestDataset(path, image_size=image_size, mask_size=image_size, image_backend="cv2_bgr")
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


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    heat_dir = out_dir / "heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = VanillaViTB16Joint(**ckpt.get("model_kwargs", {"num_classes": 3, "image_size": 256}))
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    all_fg: list[np.ndarray] = []
    all_bg: list[np.ndarray] = []
    heatmaps: list[dict[str, Any]] = []
    manifest_dir = Path(args.manifest_dir)
    for split in ("train", "val", "test"):
        loader = make_loader(manifest_dir / f"{split}.json", model.image_size, args.batch_size, args.num_workers)
        for batch in tqdm(loader, desc=f"visualize {split}", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"]
            masks = batch["mask_2d"]
            ids = batch["id"]
            img_files = batch["img_file"]
            out = model(images)
            for i in range(images.shape[0]):
                if int(labels[i].item()) == 2:
                    continue
                scores = patch_scores(out["patch_tokens"][i], out["cls_token"][i], query=args.query).detach().cpu()
                fg, bg = split_foreground_background_scores(scores, masks[i])
                if fg.size:
                    all_fg.append(fg)
                if bg.size:
                    all_bg.append(bg)
                if len(heatmaps) < args.max_heatmaps:
                    safe_id = str(ids[i]).replace("/", "_")
                    out_png = heat_dir / f"{split}_{safe_id}_{args.query}.png"
                    save_heatmap_overlay(img_files[i], scores, out_png)
                    heatmaps.append({"id": ids[i], "split": split, "label": int(labels[i].item()), "path": str(out_png)})

    fg_scores = np.concatenate(all_fg) if all_fg else np.array([], dtype=np.float32)
    bg_scores = np.concatenate(all_bg) if all_bg else np.array([], dtype=np.float32)
    summary = save_score_distribution(
        fg_scores,
        bg_scores,
        out_dir / f"patch_score_distribution_{args.query}.png",
        title=f"Vanilla ViT patch scores ({args.query})",
    )
    summary.update(
        {
            "checkpoint": args.checkpoint,
            "query": args.query,
            "manifests": {split: str(manifest_dir / f"{split}.json") for split in ("train", "val", "test")},
            "heatmaps": heatmaps,
        }
    )
    (out_dir / "visualization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
