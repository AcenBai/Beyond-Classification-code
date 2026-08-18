#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from co_pretraining.data import BUSIManifestDataset
from co_pretraining.metrics import (
    classification_metrics,
    pib_topk_fraction_in_bbox,
    pib_topk_in_bbox_grid_set,
    reduce_metric_dict,
    summarize_pib,
)
from co_pretraining.models import build_spatial_collapse_model, load_spatial_collapse_checkpoint
from co_pretraining.paths import default_manifest_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ViT or ResNet classifiers on BUSI.")
    p.add_argument("--manifest-dir", default=str(default_manifest_dir()))
    p.add_argument("--output-dir", required=True)
    p.add_argument("--backbone", choices=["vit", "dense", "dense_inv", "resnet50"], default="vit")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--patch-size", type=int, choices=[16, 32], default=16)
    p.add_argument("--train-batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--run-pib", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--checkpoint", default=None, help="Load this checkpoint for eval-only or resume.")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(path: Path, image_size: int, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
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
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }
    if workers > 0:
        kwargs["multiprocessing_context"] = "fork"
    return DataLoader(ds, **kwargs)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    logits_all, labels_all = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits, _ = model(images)
        logits_all.append(logits.detach().cpu())
        labels_all.append(labels.detach().cpu())
    return classification_metrics(torch.cat(logits_all), torch.cat(labels_all))


@torch.no_grad()
def compute_pib(model: torch.nn.Module, loaders: dict[str, DataLoader], device: torch.device) -> dict[str, Any]:
    model.eval()
    all_hits: list[float | None] = []
    all_fracs: list[float] = []
    by_split: dict[str, dict[str, Any]] = {}
    for split, loader in loaders.items():
        split_hits: list[float | None] = []
        for batch in tqdm(loader, desc=f"pib {split}", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"]
            masks = batch["mask_2d"]
            bboxes = batch["bbox"]
            _, patch_scores = model(images)
            for i in range(images.shape[0]):
                label_i = int(labels[i].item())
                bbox_i = [int(bboxes[i, j].item()) for j in range(4)]
                hit = pib_topk_in_bbox_grid_set(patch_scores[i].detach().cpu(), bbox_i, masks[i], label_i, topk=1)
                frac = pib_topk_fraction_in_bbox(patch_scores[i].detach().cpu(), bbox_i, masks[i], label_i, topk=1)
                split_hits.append(hit)
                all_hits.append(hit)
                if frac is not None:
                    all_fracs.append(float(frac))
        by_split[split] = summarize_pib(split_hits, topk=1)
    summary = summarize_pib(all_hits, topk=1)
    summary["pib_topk_frac_mean"] = float(np.mean(all_fracs)) if all_fracs else float("nan")
    summary["pib_by_split"] = by_split
    return summary


def main() -> None:
    args = parse_args()
    if args.backbone == "resnet50" and args.image_size == 224:
        print("[INFO] backbone=resnet50: using image_size=448 so layer4 is a 14x14 grid.")
        args.image_size = 448
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    manifest_dir = Path(args.manifest_dir)
    loaders = {
        "train": make_loader(manifest_dir / "train.json", args.image_size, args.train_batch_size, True, args.num_workers),
        "val": make_loader(manifest_dir / "val.json", args.image_size, args.eval_batch_size, False, args.num_workers),
        "test": make_loader(manifest_dir / "test.json", args.image_size, args.eval_batch_size, False, args.num_workers),
    }
    model = build_spatial_collapse_model(
        backbone=args.backbone,
        image_size=args.image_size,
        patch_size=args.patch_size,
        pretrained=not args.eval_only,
    ).to(device)

    if args.eval_only:
        ckpt_path = args.checkpoint or str(out_dir / "model_final.pth")
        info = load_spatial_collapse_checkpoint(model, ckpt_path, strict=False)
        print(json.dumps(info, indent=2))
        model.to(device)
        test = evaluate(model, loaders["test"], device)
        test["best_checkpoint"] = ckpt_path
        (out_dir / "eval_test_metrics.json").write_text(json.dumps(test, indent=2), encoding="utf-8")
        print(json.dumps(test, indent=2))
        if args.run_pib:
            pib = compute_pib(
                model,
                {
                    k: make_loader(manifest_dir / f"{k}.json", args.image_size, args.eval_batch_size, False, args.num_workers)
                    for k in ("train", "val", "test")
                },
                device,
            )
            pib.update({"mode": args.backbone, "checkpoint": ckpt_path})
            (out_dir / "pib_all_splits.json").write_text(json.dumps(pib, indent=2), encoding="utf-8")
            print(json.dumps({k: pib[k] for k in pib if k != "pib_by_split"}, indent=2))
            print(json.dumps(pib["pib_by_split"], indent=2))
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    cls_loss_fn = torch.nn.CrossEntropyLoss()
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history: list[dict[str, Any]] = []
    best_acc = -1.0
    best_path = out_dir / "model_final.pth"

    config = vars(args)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(loaders["train"], desc=f"train epoch {epoch}/{args.epochs}"):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits, _ = model(images)
                loss = cls_loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append({"loss": float(loss.detach().cpu().item())})
        val = evaluate(model, loaders["val"], device)
        row = {"epoch": epoch, "train": reduce_metric_dict(losses), "val": val}
        history.append(row)
        (out_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"epoch={epoch} loss={row['train']['loss']:.4f} val_acc={val.get('acc', float('nan')):.4f} val_auc={val.get('auc_macro', float('nan')):.4f}")
        if float(val.get("acc", 0.0)) > best_acc:
            best_acc = float(val.get("acc", 0.0))
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)}, best_path)

    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    test = evaluate(model, loaders["test"], device)
    test["best_checkpoint"] = str(best_path)
    test["best_epoch"] = int(ckpt["epoch"])
    (out_dir / "eval_test_metrics.json").write_text(json.dumps(test, indent=2), encoding="utf-8")
    print(json.dumps(test, indent=2))
    if args.run_pib:
        pib = compute_pib(model, {k: make_loader(manifest_dir / f"{k}.json", args.image_size, args.eval_batch_size, False, args.num_workers) for k in ("train", "val", "test")}, device)
        pib.update({"mode": args.backbone, "channels": 768 if args.backbone != "resnet50" else "resnet_feature_channels"})
        (out_dir / "pib_all_splits.json").write_text(json.dumps(pib, indent=2), encoding="utf-8")
        print(json.dumps(pib, indent=2))


if __name__ == "__main__":
    main()
