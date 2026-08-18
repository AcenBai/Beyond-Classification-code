#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from co_pretraining.data import BUSIManifestDataset
from co_pretraining.metrics import (
    pib_topk_fraction_in_bbox,
    pib_topk_in_bbox_grid_set,
    summarize_pib,
)
from co_pretraining.models import VanillaViTB16Joint, patch_scores
from co_pretraining.paths import default_manifest_dir


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
def run_pib(
    checkpoint: str,
    manifest_dir: str,
    gpu: int = 0,
    batch_size: int = 8,
    num_workers: int = 4,
    topk: int = 1,
    query: str = "cls",
) -> dict[str, Any]:
    if query not in {"cls", "mean"}:
        raise ValueError(f"Unsupported query={query!r}")
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location="cpu")
    model_kwargs = ckpt.get("model_kwargs", {"num_classes": 3, "image_size": 256})
    model = VanillaViTB16Joint(**model_kwargs)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    manifest_paths = [Path(manifest_dir) / f"{split}.json" for split in ("train", "val", "test")]
    hits: list[float | None] = []
    per_sample: list[dict[str, Any]] = []

    for manifest_path in manifest_paths:
        split = manifest_path.stem
        loader = make_loader(manifest_path, model.image_size, batch_size, num_workers)
        for batch in tqdm(loader, desc=f"pib {split} {query}", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"]
            masks = batch["mask"].squeeze(1)
            bboxes = batch["bboxes"]
            ids = batch["id"]
            out = model(images)
            patches = out["patch_tokens"]
            cls_tokens = out["cls_token"]
            grid = int(round(patches.shape[1] ** 0.5))

            for i in range(patches.shape[0]):
                scores = patch_scores(patches[i], cls_tokens[i], query=query)
                label_i = int(labels[i].item())
                bbox_i = [int(bboxes[i, j].item()) for j in range(4)]
                mask_i = masks[i].to(device)
                hit = pib_topk_in_bbox_grid_set(scores, bbox_i, mask_i, label_i, topk=topk)
                frac = pib_topk_fraction_in_bbox(scores, bbox_i, mask_i, label_i, topk=topk)
                hits.append(hit)
                per_sample.append(
                    {
                        "id": ids[i],
                        "split": split,
                        "label": label_i,
                        "pib_hit": hit,
                        "pib_topk_frac": frac,
                        "pib_topk": int(topk),
                        "grid": grid,
                    }
                )

    summary = summarize_pib(hits, topk=topk)
    fracs = [s["pib_topk_frac"] for s in per_sample if s["pib_topk_frac"] is not None]
    summary["pib_topk_frac_mean"] = float(sum(fracs) / len(fracs)) if fracs else float("nan")
    summary["mode"] = f"vanilla_vit_patch_cosine_{query}"
    summary["query"] = query
    summary["feature"] = "patch_tokens_768"
    summary["channels"] = 768
    summary["checkpoint"] = checkpoint
    summary["manifests"] = [str(p) for p in manifest_paths]
    summary["pib_by_split"] = {}
    for split in ("train", "val", "test"):
        split_hits = [s["pib_hit"] for s in per_sample if s["split"] == split]
        summary["pib_by_split"][split] = summarize_pib(split_hits, topk=topk)
    return {"summary": summary, "samples": per_sample}


def main() -> None:
    p = argparse.ArgumentParser(description="Compute PIB for the jointly trained ViT.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest-dir", default=str(default_manifest_dir()))
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--topk", type=int, default=1)
    p.add_argument("--query", choices=["cls", "mean"], default="cls")
    p.add_argument("--out-json", required=True)
    args = p.parse_args()
    result = run_pib(
        checkpoint=args.checkpoint,
        manifest_dir=args.manifest_dir,
        gpu=args.gpu,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        topk=args.topk,
        query=args.query,
    )
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
