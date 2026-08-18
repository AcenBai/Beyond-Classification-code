#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from co_pretraining.data import BUSIManifestDataset, CLASS_NAMES
from co_pretraining.metrics import classification_metrics, format_classification_metrics, pib_topk_in_bbox_grid_set, summarize_pib
from co_pretraining.models.lvm_med import LVMMedClassifier
from co_pretraining.paths import default_lvm_weights, default_manifest_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LVM-Med linear probe or fine-tuning on BUSI.")
    p.add_argument("--stage", choices=["train", "eval"], default="train")
    p.add_argument("--frozen", action=argparse.BooleanOptionalAction, default=True, help="True=linear probe, False=finetune encoder.")
    p.add_argument("--manifest-dir", default=str(default_manifest_dir()))
    p.add_argument("--output-dir", required=True)
    p.add_argument("--lvm-weights", default=str(default_lvm_weights()))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=100)
    return p.parse_args()


def make_loader(path: Path, image_size: int, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    ds = BUSIManifestDataset(
        path,
        image_size=image_size,
        mask_size=image_size,
        image_backend="cv2_bgr",
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
def evaluate(model: LVMMedClassifier, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    logits_all, labels_all = [], []
    for batch in tqdm(loader, desc="lvm eval", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"]
        logits, _ = model.forward_logits_and_scores(images)
        logits_all.append(logits.detach().float().cpu())
        labels_all.append(labels.detach().cpu())
    return classification_metrics(torch.cat(logits_all), torch.cat(labels_all), class_names=CLASS_NAMES)


@torch.no_grad()
def compute_pib(model: LVMMedClassifier, loaders: dict[str, DataLoader], device: torch.device) -> dict[str, Any]:
    model.eval()
    all_hits: list[float | None] = []
    by_split: dict[str, dict[str, Any]] = {}
    for split, loader in loaders.items():
        split_hits: list[float | None] = []
        for batch in tqdm(loader, desc=f"pib {split}", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"]
            masks = batch["mask_2d"]
            bboxes = batch["bbox"]
            _, patch_scores = model.forward_logits_and_scores(images)
            for i in range(images.shape[0]):
                hit = pib_topk_in_bbox_grid_set(
                    patch_scores[i].detach().cpu(),
                    [int(bboxes[i, j].item()) for j in range(4)],
                    masks[i],
                    int(labels[i].item()),
                    topk=1,
                )
                split_hits.append(hit)
                all_hits.append(hit)
        by_split[split] = summarize_pib(split_hits, topk=1)
    summary = summarize_pib(all_hits, topk=1)
    summary["pib_by_split"] = by_split
    return summary


def write_eval_outputs(
    model: LVMMedClassifier,
    loaders: dict[str, DataLoader],
    device: torch.device,
    out_dir: Path,
    extra: dict[str, Any],
    manifest_dir: Path,
    args: argparse.Namespace,
) -> None:
    test = evaluate(model, loaders["test"], device)
    test.update(extra)
    (out_dir / "eval_test_metrics.json").write_text(json.dumps(test, indent=2), encoding="utf-8")
    pib_loaders = {
        split: make_loader(manifest_dir / f"{split}.json", args.image_size, args.batch_size, False, args.num_workers)
        for split in ("train", "val", "test")
    }
    pib = compute_pib(model, pib_loaders, device)
    (out_dir / "pib_all_splits.json").write_text(json.dumps(pib, indent=2), encoding="utf-8")
    print(format_classification_metrics(test))
    print(json.dumps({k: pib[k] for k in pib if k != "pib_by_split"}, indent=2))
    print(json.dumps(pib["pib_by_split"], indent=2))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    manifest_dir = Path(args.manifest_dir)
    loaders = {
        "train": make_loader(manifest_dir / "train.json", args.image_size, args.batch_size, True, args.num_workers),
        "val": make_loader(manifest_dir / "val.json", args.image_size, args.batch_size, False, args.num_workers),
        "test": make_loader(manifest_dir / "test.json", args.image_size, args.batch_size, False, args.num_workers),
    }
    model = LVMMedClassifier(num_classes=3)
    best_path = Path(args.checkpoint) if args.checkpoint else out_dir / "best_model.pth"

    if args.stage == "eval":
        if not best_path.exists():
            raise FileNotFoundError(best_path)
        model.load_busi_checkpoint(best_path)
        model.to(device)
        write_eval_outputs(
            model,
            loaders,
            device,
            out_dir,
            {"checkpoint": str(best_path), "frozen": bool(args.frozen)},
            manifest_dir,
            args,
        )
        return

    if not Path(args.lvm_weights).exists():
        raise FileNotFoundError(
            f"LVM-Med checkpoint not found: {args.lvm_weights}. "
            "Download lvmmed_vit.pth from https://github.com/duyhominhnguyen/LVM-Med "
            "and set --lvm-weights / LVM_MED_WEIGHTS."
        )
    model.load_official_encoder(args.lvm_weights, strict=False)
    if args.frozen:
        model.freeze_encoder()
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    criterion = torch.nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history = []
    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.frozen:
            model.encoder.model.eval()
        train_loss = 0.0
        train_count = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(loaders["train"], desc=f"lvm epoch {epoch}/{args.epochs}")):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, _ = model.forward_logits_and_scores(images)
                loss = criterion(logits, labels) / float(max(1, args.grad_accum_steps))
            scaler.scale(loss).backward()
            if ((step + 1) % args.grad_accum_steps == 0) or (step + 1 == len(loaders["train"])):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_loss += float(loss.item()) * args.grad_accum_steps * images.size(0)
            train_count += int(images.size(0))
        val = evaluate(model, loaders["val"], device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_count, 1),
            "val_acc": val.get("acc"),
        }
        history.append(row)
        (out_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"epoch={epoch} train_loss={row['train_loss']:.4f} val_acc={val.get('acc', float('nan')):.4f}")
        if float(val.get("acc", 0.0)) >= best_acc:
            best_acc = float(val.get("acc", 0.0))
            torch.save({"epoch": epoch, "state_dict": model.encoder.state_dict(), "args": vars(args)}, best_path)

    model.load_busi_checkpoint(best_path)
    model.to(device)
    write_eval_outputs(
        model,
        loaders,
        device,
        out_dir,
        {"checkpoint": str(best_path), "frozen": bool(args.frozen), "lvm_weights": str(args.lvm_weights)},
        manifest_dir,
        args,
    )


if __name__ == "__main__":
    main()
