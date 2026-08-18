#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from co_pretraining.data import BUSIManifestDataset
from co_pretraining.external.lvm_med import DEFAULT_LVM_MED_REPO, DEFAULT_LVM_WEIGHTS, ensure_lvm_med_repo
from co_pretraining.metrics import (
    pib_topk_fraction_in_bbox,
    pib_topk_in_bbox_grid_set,
    summarize_pib,
)
from co_pretraining.paths import default_manifest_dir


SAM_PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
SAM_PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)


def sam_preprocess_images(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    mean = SAM_PIXEL_MEAN.to(device=device, dtype=x.dtype)
    std = SAM_PIXEL_STD.to(device=device, dtype=x.dtype)
    return (x - mean) / std


def patch_scores_mean_pool(patch_flat: torch.Tensor) -> torch.Tensor:
    q = patch_flat.mean(dim=0, keepdim=True)
    return F.cosine_similarity(patch_flat, q.expand_as(patch_flat), dim=-1)


def make_loader(path: Path, batch_size: int, workers: int) -> DataLoader:
    ds = BUSIManifestDataset(path, image_size=1024, mask_size=256, image_backend="cv2_bgr")
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
    p = argparse.ArgumentParser(description="PIB for a frozen LVM-Med encoder.")
    p.add_argument("--manifest-dir", default=str(default_manifest_dir()))
    p.add_argument("--lvm-med-repo", default=str(DEFAULT_LVM_MED_REPO))
    p.add_argument("--lvm-weights", default=str(DEFAULT_LVM_WEIGHTS))
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--topk", type=int, default=1)
    p.add_argument("--out-json", required=True)
    args = p.parse_args()

    ensure_lvm_med_repo(args.lvm_med_repo)
    from segment_anything import our_vit

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    enc = our_vit.__dict__["vit_encoder_b"]()
    enc.load_state_dict(torch.load(args.lvm_weights, map_location="cpu"))
    enc.eval().to(device)

    manifest_dir = Path(args.manifest_dir)
    all_hits: list[float | None] = []
    all_fracs: list[float] = []
    per_sample: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        loader = make_loader(manifest_dir / f"{split}.json", args.batch_size, args.num_workers)
        for batch in tqdm(loader, desc=f"lvm pib {split}", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            x = sam_preprocess_images(images, device)
            tok = enc.forward_tokens_pre_neck(x)
            b, gh, gw, c = tok.shape
            flat = tok.reshape(b, gh * gw, c)
            labels = batch["label"]
            masks = batch["mask"].squeeze(1)
            bboxes = batch["bboxes"]
            ids = batch["id"]
            for i in range(b):
                scores = patch_scores_mean_pool(flat[i])
                label_i = int(labels[i].item())
                bbox_i = [int(bboxes[i, j].item()) for j in range(4)]
                hit = pib_topk_in_bbox_grid_set(scores.detach().cpu(), bbox_i, masks[i], label_i, topk=args.topk)
                frac = pib_topk_fraction_in_bbox(scores.detach().cpu(), bbox_i, masks[i], label_i, topk=args.topk)
                all_hits.append(hit)
                if frac is not None:
                    all_fracs.append(float(frac))
                per_sample.append({"id": ids[i], "split": split, "label": label_i, "pib_hit": hit, "pib_topk_frac": frac, "grid": gh})
    summary = summarize_pib(all_hits, topk=args.topk)
    summary["pib_topk_frac_mean"] = float(sum(all_fracs) / len(all_fracs)) if all_fracs else float("nan")
    summary["mode"] = "lvm_med_frozen_mean_pool_pre_neck"
    summary["feature"] = "pre_neck_patch_tokens"
    summary["channels"] = 768
    summary["lvm_weights"] = args.lvm_weights
    summary["pib_by_split"] = {
        split: summarize_pib([s["pib_hit"] for s in per_sample if s["split"] == split], topk=args.topk)
        for split in ("train", "val", "test")
    }
    out = {"summary": summary, "samples": per_sample}
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
