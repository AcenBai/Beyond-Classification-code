#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from co_pretraining.data import BUSIManifestDataset, CLASS_NAMES
from co_pretraining.models.biomedclip import (
    DEFAULT_CKPT_DIR,
    DEFAULT_MODEL_NAME,
    BiomedCLIPPartialFinetune,
    load_biomedclip,
    load_finetune_checkpoint,
    save_finetune_checkpoint,
    trainable_visual_parameters,
)
from co_pretraining.metrics import (
    classification_metrics,
    format_classification_metrics,
    infer_grid_hw,
    pib_topk_in_bbox_grid_set,
    summarize_pib,
)
from co_pretraining.visualization.patch_scores import (
    save_heatmap_overlay,
    save_score_distribution,
    split_foreground_background_scores,
)
from co_pretraining.paths import default_manifest_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune BiomedCLIP on BUSI.")
    p.add_argument("--stage", choices=["train", "eval-test", "analysis", "both"], default="both")
    p.add_argument("--manifest-dir", default=str(default_manifest_dir()))
    p.add_argument("--finetune-checkpoint", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    p.add_argument("--unfreeze-last-n-blocks", type=int, default=12)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--visual-lr", type=float, default=2e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--max-heatmaps", type=int, default=24)
    return p.parse_args()


def make_loader(
    manifest_path: Path,
    image_size: int,
    batch_size: int,
    workers: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    shuffle: bool = False,
) -> DataLoader:
    ds = BUSIManifestDataset(
        manifest_path,
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


def _make_model(args: argparse.Namespace, device: torch.device):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    clip_model, _, _, preprocess_cfg, _ = load_biomedclip(args.ckpt_dir, args.model_name, device=device)
    ckpt_path = args.finetune_checkpoint
    if not ckpt_path:
        raise SystemExit("--finetune-checkpoint is required for eval/analysis")
    probe = load_finetune_checkpoint(ckpt_path, clip_model, device=device)
    probe.eval()
    mean = tuple(preprocess_cfg["mean"])
    std = tuple(preprocess_cfg["std"])
    return probe, mean, std


def train_finetune(args: argparse.Namespace, device: torch.device) -> str:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_model, _, _, preprocess_cfg, embed_dim = load_biomedclip(args.ckpt_dir, args.model_name, device=device)
    model = BiomedCLIPPartialFinetune(
        clip_model,
        num_classes=3,
        embed_dim=embed_dim,
        unfreeze_last_n_blocks=args.unfreeze_last_n_blocks,
    ).to(device)
    mean = tuple(preprocess_cfg["mean"])
    std = tuple(preprocess_cfg["std"])
    manifest_dir = Path(args.manifest_dir)
    train_loader = make_loader(manifest_dir / "train.json", args.image_size, args.batch_size, args.num_workers, mean, std, shuffle=True)
    val_loader = make_loader(manifest_dir / "val.json", args.image_size, args.batch_size, args.num_workers, mean, std, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [
            {"params": list(trainable_visual_parameters(model.clip)), "lr": args.visual_lr},
            {"params": model.head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    best_path = out_dir / "best_finetune.pt"
    history = []
    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.set_train_mode()
        total, correct, loss_sum = 0, 0, 0.0
        for batch in tqdm(train_loader, desc=f"ft epoch {epoch}/{args.epochs}"):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * images.size(0)
            correct += int((logits.argmax(1) == labels).sum().item())
            total += int(images.size(0))
        model.eval()
        logits_all, labels_all = [], []
        with torch.inference_mode():
            for batch in val_loader:
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"]
                logits = model(images)
                logits_all.append(logits.detach().cpu())
                labels_all.append(labels.detach().cpu())
        val = classification_metrics(torch.cat(logits_all), torch.cat(labels_all), class_names=CLASS_NAMES)
        row = {"epoch": epoch, "train_loss": loss_sum / max(total, 1), "train_acc": correct / max(total, 1), "val": val}
        history.append(row)
        (out_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"epoch={epoch} train_acc={row['train_acc']:.4f} val_acc={val['acc']:.4f}")
        if float(val["acc"]) > best_acc:
            best_acc = float(val["acc"])
            save_finetune_checkpoint(best_path, model, extra={"epoch": epoch, "val": val})
    args.finetune_checkpoint = str(best_path)
    return str(best_path)


@torch.inference_mode()
def eval_test(args: argparse.Namespace, probe, mean, std, device: torch.device) -> dict[str, Any]:
    manifest = Path(args.manifest_dir) / "test.json"
    loader = make_loader(manifest, args.image_size, args.batch_size, args.num_workers, mean, std)
    logits_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    hits: list[float | None] = []
    normal_skipped = 0

    for batch in tqdm(loader, desc="biomedclip finetune eval-test", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"]
        masks = batch["mask_2d"]
        bboxes = batch["bbox"]
        logits, patch_scores = probe(images, return_patch_scores=True)
        logits_all.append(logits.detach().cpu())
        labels_all.append(labels.detach().cpu())

        for i in range(images.shape[0]):
            label_i = int(labels[i].item())
            if label_i == 2:
                normal_skipped += 1
                continue
            bbox_i = [int(v.item()) for v in bboxes[i]]
            hits.append(pib_topk_in_bbox_grid_set(patch_scores[i].detach().cpu(), bbox_i, masks[i], label_i, topk=1))

    cls_metrics = classification_metrics(
        torch.cat(logits_all),
        torch.cat(labels_all),
        num_classes=getattr(probe, "num_classes", len(CLASS_NAMES)),
        class_names=CLASS_NAMES,
    )
    result = {
        "manifest": str(manifest),
        "finetune_checkpoint": args.finetune_checkpoint,
        "pooling_mode": probe.pooling_mode,
        "lazy_strike_topk": int(getattr(probe, "lazy_strike_topk", 1)),
        "classification": cls_metrics,
        "pib": summarize_pib(hits, topk=1),
        "pib_feature": "probe_return_patch_scores",
        "pib_channels": 768,
        "normal_skipped_for_pib": normal_skipped,
    }
    out = Path(args.output_dir) / "eval_test_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(format_classification_metrics(cls_metrics))
    print(json.dumps({"pib": result["pib"]}, indent=2))
    return result


@torch.inference_mode()
def analysis_all_splits(args: argparse.Namespace, probe, mean, std, device: torch.device) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    heat_dir = out_dir / "heatmaps"
    manifest_dir = Path(args.manifest_dir)
    all_hits: list[float | None] = []
    split_hits: dict[str, list[float | None]] = {split: [] for split in ("train", "val", "test")}
    fg_scores_all: list[np.ndarray] = []
    bg_scores_all: list[np.ndarray] = []
    heatmaps: list[dict[str, Any]] = []
    normal_skipped = 0
    lesion_images = 0

    use_votes_for_heatmap = probe.pooling_mode in {"lazy_strike", "lazy_strike_inv"}
    ranking_source = "lazy_strike_vote_count" if use_votes_for_heatmap else "patch_score"

    for split in ("train", "val", "test"):
        manifest = manifest_dir / f"{split}.json"
        loader = make_loader(manifest, args.image_size, args.batch_size, args.num_workers, mean, std)
        for batch in tqdm(loader, desc=f"biomedclip finetune analysis {split}", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"]
            masks = batch["mask_2d"]
            bboxes = batch["bbox"]
            ids = batch["id"]
            img_files = batch["img_file"]
            logits, patch_scores = probe(images, return_patch_scores=True)
            ranking_scores = probe.ranking_scores(images, patch_scores)
            hist_scores = patch_scores.detach().cpu()
            ranking_scores = ranking_scores.detach().cpu()
            infer_grid_hw(int(hist_scores.shape[1]))

            for i in range(images.shape[0]):
                label_i = int(labels[i].item())
                if label_i == 2:
                    normal_skipped += 1
                    continue
                lesion_images += 1
                bbox_i = [int(v.item()) for v in bboxes[i]]
                hit = pib_topk_in_bbox_grid_set(hist_scores[i], bbox_i, masks[i], label_i, topk=1)
                all_hits.append(hit)
                split_hits[split].append(hit)

                fg, bg = split_foreground_background_scores(hist_scores[i], masks[i])
                if fg.size:
                    fg_scores_all.append(fg)
                if bg.size:
                    bg_scores_all.append(bg)

                if len(heatmaps) < args.max_heatmaps:
                    safe_id = str(ids[i]).replace("/", "_")
                    out_png = heat_dir / f"{split}_{safe_id}_{ranking_source}.png"
                    save_heatmap_overlay(img_files[i], ranking_scores[i], out_png)
                    heatmaps.append(
                        {
                            "id": ids[i],
                            "split": split,
                            "label": label_i,
                            "ranking_source": ranking_source,
                            "path": str(out_png),
                            "top_patch_index": int(ranking_scores[i].argmax().item()),
                            "top_patch_value": float(ranking_scores[i].max().item()),
                        }
                    )

    fg_scores = np.concatenate(fg_scores_all) if fg_scores_all else np.array([], dtype=np.float32)
    bg_scores = np.concatenate(bg_scores_all) if bg_scores_all else np.array([], dtype=np.float32)
    hist_summary = save_score_distribution(
        fg_scores,
        bg_scores,
        out_dir / "patch_score_distribution.png",
        title=f"BiomedCLIP finetune patch scores ({probe.pooling_mode})",
    )
    pib_summary = summarize_pib(all_hits, topk=1)
    result = {
        "split_manifests": [str(manifest_dir / f"{split}.json") for split in ("train", "val", "test")],
        "finetune_checkpoint": args.finetune_checkpoint,
        "pooling_mode": probe.pooling_mode,
        "lazy_strike_topk": int(getattr(probe, "lazy_strike_topk", 1)),
        "pib": pib_summary["pib"],
        "pib_count": pib_summary["pib_count"],
        "pib_rule": pib_summary["pib_rule"],
        "pib_by_split": {split: summarize_pib(vals, topk=1) for split, vals in split_hits.items()},
        "pib_feature": "probe_return_patch_scores",
        "pib_channels": 768,
        "patch_score_hist": hist_summary,
        "heatmap_ranking_source": ranking_source,
        "heatmaps": heatmaps,
        "diagnostics": {
            "lesion_images": lesion_images,
            "normal_skipped": normal_skipped,
            "distribution_source": "patch_score",
            "heatmap_source": "lazy_strike_vote_count" if use_votes_for_heatmap else "patch_score",
        },
    }
    out = out_dir / "busi_analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if args.stage == "train":
        train_finetune(args, device)
        args.stage = "both"
    if args.finetune_checkpoint is None:
        guess = Path(args.output_dir) / "best_finetune.pt"
        if guess.exists():
            args.finetune_checkpoint = str(guess)
    probe, mean, std = _make_model(args, device)
    if args.stage in {"eval-test", "both"}:
        eval_test(args, probe, mean, std, device)
    if args.stage in {"analysis", "both"}:
        analysis_all_splits(args, probe, mean, std, device)


if __name__ == "__main__":
    main()
