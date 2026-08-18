#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from co_pretraining.data import BUSIManifestDataset, CLASS_NAMES
from co_pretraining.metrics import classification_metrics, format_classification_metrics, pib_topk_in_bbox_grid_set, summarize_pib
from co_pretraining.models.biomedclip import (
    DEFAULT_CKPT_DIR,
    DEFAULT_MODEL_NAME,
    BiomedCLIPLinearProbe,
    load_biomedclip,
    load_probe_checkpoint,
    save_probe_checkpoint,
)
from co_pretraining.paths import default_manifest_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Linear probe on BiomedCLIP for BUSI.")
    p.add_argument("--stage", choices=["train", "eval-test", "analysis", "both"], default="train")
    p.add_argument("--manifest-dir", default=str(default_manifest_dir()))
    p.add_argument("--output-dir", default="./outputs/biomedclip_linear_probe")
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    p.add_argument("--probe-checkpoint", default=None)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--image-size", type=int, default=224)
    return p.parse_args()


def make_loader(manifest: Path, image_size: int, batch_size: int, shuffle: bool, workers: int, mean, std) -> DataLoader:
    ds = BUSIManifestDataset(
        manifest,
        image_size=image_size,
        mask_size=image_size,
        normalize_mean=mean,
        normalize_std=std,
        image_backend="pil_rgb",
    )
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": shuffle,
    }
    if workers > 0:
        kwargs["multiprocessing_context"] = "fork"
    return DataLoader(ds, **kwargs)


@torch.inference_mode()
def eval_split(model, loader, device) -> tuple[dict[str, Any], list[float | None]]:
    model.eval()
    logits_all, labels_all = [], []
    hits: list[float | None] = []
    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"]
        logits, patch_scores = model(images, return_patch_scores=True)
        logits_all.append(logits.detach().cpu())
        labels_all.append(labels.detach().cpu())
        masks = batch["mask_2d"]
        bboxes = batch["bbox"]
        for i in range(images.shape[0]):
            hits.append(
                pib_topk_in_bbox_grid_set(
                    patch_scores[i].detach().cpu(),
                    [int(v.item()) for v in bboxes[i]],
                    masks[i],
                    int(labels[i].item()),
                    topk=1,
                )
            )
    metrics = classification_metrics(torch.cat(logits_all), torch.cat(labels_all), class_names=CLASS_NAMES)
    return metrics, hits


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    clip_model, _, _, preprocess_cfg, embed_dim = load_biomedclip(args.ckpt_dir, args.model_name, device=device)
    clip_model.eval()
    mean = tuple(preprocess_cfg["mean"])
    std = tuple(preprocess_cfg["std"])
    manifest_dir = Path(args.manifest_dir)
    probe_path = Path(args.probe_checkpoint) if args.probe_checkpoint else out_dir / "best_probe.pt"

    if args.stage in {"train"}:
        probe = BiomedCLIPLinearProbe(clip_model, num_classes=3, embed_dim=embed_dim).to(device)
        train_loader = make_loader(manifest_dir / "train.json", args.image_size, args.batch_size, True, args.num_workers, mean, std)
        val_loader = make_loader(manifest_dir / "val.json", args.image_size, args.batch_size, False, args.num_workers, mean, std)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(probe.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        history = []
        best_acc = -1.0
        for epoch in range(1, args.epochs + 1):
            probe.train()
            probe.clip.eval()
            total, correct, loss_sum = 0, 0, 0.0
            for batch in tqdm(train_loader, desc=f"lp epoch {epoch}/{args.epochs}"):
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = probe(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.item()) * images.size(0)
                correct += int((logits.argmax(1) == labels).sum().item())
                total += int(images.size(0))
            val_metrics, _ = eval_split(probe, val_loader, device)
            row = {"epoch": epoch, "train_loss": loss_sum / max(total, 1), "train_acc": correct / max(total, 1), "val": val_metrics}
            history.append(row)
            (out_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            print(f"epoch={epoch} train_acc={row['train_acc']:.4f} val_acc={val_metrics['acc']:.4f}")
            if float(val_metrics["acc"]) > best_acc:
                best_acc = float(val_metrics["acc"])
                save_probe_checkpoint(probe_path, probe, extra={"epoch": epoch, "val": val_metrics})
        args.stage = "both"

    probe = load_probe_checkpoint(probe_path, clip_model, device=device)
    probe.eval()
    if args.stage in {"eval-test", "both"}:
        test_loader = make_loader(manifest_dir / "test.json", args.image_size, args.batch_size, False, args.num_workers, mean, std)
        cls_metrics, hits = eval_split(probe, test_loader, device)
        result = {
            "manifest": str(manifest_dir / "test.json"),
            "probe_checkpoint": str(probe_path),
            "pooling_mode": "cls",
            "classification": cls_metrics,
            "pib": summarize_pib(hits, topk=1),
        }
        (out_dir / "eval_test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(format_classification_metrics(cls_metrics))
        print(json.dumps({"pib_test": result["pib"]}, indent=2))

    if args.stage in {"analysis", "both"}:
        all_hits: list[float | None] = []
        by_split = {}
        for split in ("train", "val", "test"):
            loader = make_loader(manifest_dir / f"{split}.json", args.image_size, args.batch_size, False, args.num_workers, mean, std)
            _, hits = eval_split(probe, loader, device)
            by_split[split] = summarize_pib(hits, topk=1)
            all_hits.extend(hits)
        analysis = {
            "probe_checkpoint": str(probe_path),
            "pooling_mode": "cls",
            "pib": summarize_pib(all_hits, topk=1)["pib"],
            "pib_count": summarize_pib(all_hits, topk=1)["pib_count"],
            "pib_by_split": by_split,
        }
        analysis.update(summarize_pib(all_hits, topk=1))
        (out_dir / "busi_analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        print(json.dumps({"pib_all_splits": analysis["pib"], "pib_by_split": by_split}, indent=2))


if __name__ == "__main__":
    main()
