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
    LesionAwareSegLoss,
    classification_metrics,
    dice_from_logits,
    pib_topk_fraction_in_bbox,
    pib_topk_in_bbox_grid_set,
    reduce_metric_dict,
    summarize_pib,
)
from co_pretraining.models import VanillaViTB16Joint, load_vanilla_vit_init, patch_scores
from co_pretraining.runners.pib_vanilla_vit import run_pib
from co_pretraining.paths import default_manifest_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Joint classification and segmentation on BUSI with ViT-B/16.")
    p.add_argument("--manifest-dir", default=str(default_manifest_dir()))
    p.add_argument("--output-dir", default="./outputs/vit_cotraining")
    p.add_argument("--init", choices=["random", "torchvision_imagenet", "torchvision_vit_ckpt"], default="torchvision_imagenet")
    p.add_argument("--encoder-checkpoint", default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--train-batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--encoder-lr", type=float, default=1e-5)
    p.add_argument("--decoder-lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--lambda-seg", type=float, default=2.0)
    p.add_argument("--lambda-cls", type=float, default=1.0)
    p.add_argument("--normal-empty-weight", type=float, default=0.1)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--run-pib", action="store_true")
    p.add_argument("--pib-topk", type=int, default=1)
    p.add_argument("--epoch-pib", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-cls-seg-fusion", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--checkpoint", default=None)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(path: Path, image_size: int, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    ds = BUSIManifestDataset(path, image_size=image_size, mask_size=image_size, image_backend="cv2_bgr")
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


def make_optimizer(model: VanillaViTB16Joint, args: argparse.Namespace) -> torch.optim.Optimizer:
    param_groups = [
        {"params": model.vit.parameters(), "lr": args.encoder_lr},
        {"params": model.seg_decoder.parameters(), "lr": args.decoder_lr},
        {"params": model.cls_head.parameters(), "lr": args.head_lr},
    ]
    cls_to_patch_params = list(model.cls_to_patch.parameters())
    if cls_to_patch_params:
        param_groups.append({"params": cls_to_patch_params, "lr": args.decoder_lr})
    return torch.optim.AdamW(param_groups, weight_decay=0.01)


@torch.no_grad()
def evaluate(
    model: VanillaViTB16Joint,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> dict[str, Any]:
    model.eval()
    logits_all, labels_all = [], []
    dice_batches: list[dict[str, float]] = []
    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True).float()
        labels = batch["label"].to(device, non_blocking=True)
        out = model(images)
        logits_all.append(out["logits"].detach().cpu())
        labels_all.append(labels.detach().cpu())
        dice_batches.append(dice_from_logits(out["mask_logits"], masks, labels, threshold=threshold))
    return {
        "classification": classification_metrics(torch.cat(logits_all), torch.cat(labels_all)),
        "segmentation": reduce_metric_dict(dice_batches),
    }


@torch.no_grad()
def evaluate_pib(
    model: VanillaViTB16Joint,
    loader: DataLoader,
    device: torch.device,
    topk: int,
    query: str,
) -> dict[str, float]:
    model.eval()
    hits: list[float | None] = []
    fracs: list[float] = []
    for batch in tqdm(loader, desc=f"val pib {query}", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"]
        masks = batch["mask"].squeeze(1)
        bboxes = batch["bboxes"]
        out = model(images)
        for i in range(out["patch_tokens"].shape[0]):
            scores = patch_scores(out["patch_tokens"][i], out["cls_token"][i], query=query)
            label_i = int(labels[i].item())
            bbox_i = [int(bboxes[i, j].item()) for j in range(4)]
            mask_i = masks[i].to(device)
            hit = pib_topk_in_bbox_grid_set(scores, bbox_i, mask_i, label_i, topk=topk)
            frac = pib_topk_fraction_in_bbox(scores, bbox_i, mask_i, label_i, topk=topk)
            hits.append(hit)
            if frac is not None:
                fracs.append(float(frac))
    summary = summarize_pib(hits, topk=topk)
    return {
        "pib": float(summary["pib"]),
        "pib_count": float(summary["pib_count"]),
        "pib_topk_frac_mean": float(np.mean(fracs)) if fracs else float("nan"),
    }


def main() -> None:
    args = parse_args()
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

    model_kwargs = {
        "num_classes": 3,
        "image_size": args.image_size,
        "use_cls_seg_fusion": args.use_cls_seg_fusion,
    }
    model = VanillaViTB16Joint(**model_kwargs)
    if args.eval_only:
        ckpt_path = args.checkpoint or str(out_dir / "best_joint_model.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_kwargs = ckpt.get("model_kwargs", model_kwargs)
        model = VanillaViTB16Joint(**model_kwargs)
        model.load_state_dict(ckpt["model"])
        model.to(device)
        test_metrics = evaluate(model, loaders["test"], device, args.threshold)
        test_metrics["best_checkpoint"] = ckpt_path
        test_metrics["best_epoch"] = int(ckpt.get("epoch", -1))
        (out_dir / "eval_test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
        print(json.dumps(test_metrics, indent=2))
        if args.run_pib:
            pib_cls = run_pib(ckpt_path, str(manifest_dir), args.gpu, args.eval_batch_size, args.num_workers, args.pib_topk, "cls")
            (out_dir / "pib_all_splits_cls.json").write_text(json.dumps(pib_cls, indent=2), encoding="utf-8")
            print(json.dumps(pib_cls.get("summary", pib_cls), indent=2))
        return

    init_info = load_vanilla_vit_init(model, init=args.init, checkpoint=args.encoder_checkpoint)
    model.to(device)

    optimizer = make_optimizer(model, args)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    seg_loss_fn = LesionAwareSegLoss(normal_empty_weight=args.normal_empty_weight)
    cls_loss_fn = torch.nn.CrossEntropyLoss()

    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_path = out_dir / "best_joint_model.pt"
    config = vars(args) | {"init_info": init_info, "model_kwargs": model_kwargs}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(init_info, indent=2))

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(loaders["train"], desc=f"train epoch {epoch}/{args.epochs}"):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True).float()
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                out = model(images)
                cls_loss = cls_loss_fn(out["logits"], labels)
                seg_loss = seg_loss_fn(out["mask_logits"], masks, labels)
                loss = args.lambda_cls * cls_loss + args.lambda_seg * seg_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(
                {
                    "loss": float(loss.detach().cpu().item()),
                    "cls_loss": float(cls_loss.detach().cpu().item()),
                    "seg_loss": float(seg_loss.detach().cpu().item()),
                }
            )

        val_metrics = evaluate(model, loaders["val"], device, args.threshold)
        if args.epoch_pib:
            val_metrics["pib_cls"] = evaluate_pib(model, loaders["val"], device, args.pib_topk, "cls")
            val_metrics["pib_mean"] = evaluate_pib(model, loaders["val"], device, args.pib_topk, "mean")
        epoch_row = {"epoch": epoch, "train": reduce_metric_dict(losses), "val": val_metrics}
        history.append(epoch_row)
        (out_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        val_seg = val_metrics["segmentation"]
        val_cls = val_metrics["classification"]
        score = float(val_seg.get("dice_lesion", float("nan"))) + float(val_cls.get("acc", 0.0))
        print(
            f"epoch={epoch} loss={epoch_row['train']['loss']:.4f} "
            f"val_acc={val_cls.get('acc', float('nan')):.4f} "
            f"val_dice_lesion={val_seg.get('dice_lesion', float('nan')):.4f} "
            f"val_dice_all={val_seg.get('dice_all', float('nan')):.4f} "
            f"val_pib_cls={val_metrics.get('pib_cls', {}).get('pib', float('nan')):.4f} "
            f"val_pib_mean={val_metrics.get('pib_mean', {}).get('pib', float('nan')):.4f}"
        )
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "score": best_score,
                    "args": vars(args),
                    "init_info": init_info,
                    "model_kwargs": model_kwargs,
                },
                best_path,
            )

    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    test_metrics = evaluate(model, loaders["test"], device, args.threshold)
    test_metrics["best_checkpoint"] = str(best_path)
    test_metrics["best_epoch"] = int(ckpt["epoch"])
    test_metrics["empty_mask_dice_rule"] = (
        "GT empty + pred empty => 1; GT empty + pred non-empty => 0; "
        "non-empty GT uses standard Dice. dice_lesion excludes label==2 normal."
    )
    (out_dir / "eval_test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    print(json.dumps(test_metrics, indent=2))

    if args.run_pib:
        pib_cls = run_pib(str(best_path), str(manifest_dir), args.gpu, args.eval_batch_size, args.num_workers, args.pib_topk, "cls")
        pib_mean = run_pib(str(best_path), str(manifest_dir), args.gpu, args.eval_batch_size, args.num_workers, args.pib_topk, "mean")
        (out_dir / "pib_all_splits_cls.json").write_text(json.dumps(pib_cls, indent=2), encoding="utf-8")
        (out_dir / "pib_all_splits_mean.json").write_text(json.dumps(pib_mean, indent=2), encoding="utf-8")
        combined = {"cls_query": pib_cls["summary"], "mean_query": pib_mean["summary"]}
        (out_dir / "pib_all_splits_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
        print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
