#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from co_pretraining.data import BUSIManifestDataset, CLASS_NAMES
from co_pretraining.models.biomedclip import (
    DEFAULT_CKPT_DIR,
    DEFAULT_MODEL_NAME,
    encode_image_features,
    load_biomedclip,
)
from co_pretraining.metrics import (
    classification_metrics,
    format_classification_metrics,
    pib_topk_in_bbox_grid_set,
    summarize_pib,
)
from co_pretraining.paths import default_manifest_dir


DEFAULT_PROMPTS = {
    "benign": "a breast ultrasound image of a benign breast lesion",
    "malignant": "a breast ultrasound image of a malignant breast lesion",
    "normal": "a breast ultrasound image of normal breast tissue",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zero-shot BiomedCLIP evaluation on BUSI.")
    p.add_argument("--manifest-dir", default=str(default_manifest_dir()))
    p.add_argument("--output-dir", default="./outputs/biomedclip_zeroshot")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--prompts-json", default=None)
    p.add_argument("--out-json", default=None)
    return p.parse_args()


def make_loader(
    manifest_path: Path,
    image_size: int,
    batch_size: int,
    workers: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> DataLoader:
    ds = BUSIManifestDataset(
        manifest_path=manifest_path,
        image_size=image_size,
        mask_size=image_size,
        normalize_mean=mean,
        normalize_std=std,
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


def load_prompts(path: str | None) -> dict[str, str]:
    if path is None:
        return dict(DEFAULT_PROMPTS)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [name for name in CLASS_NAMES if name not in data]
    if missing:
        raise ValueError(f"Prompts JSON missing classes: {missing}")
    return {name: str(data[name]) for name in CLASS_NAMES}


@torch.inference_mode()
def encode_text_prompts(model: torch.nn.Module, tokenizer, prompts: dict[str, str], device: torch.device) -> torch.Tensor:
    texts = [prompts[name] for name in CLASS_NAMES]
    tokens = tokenizer(texts, context_length=256).to(device)
    try:
        text_features = model.encode_text(tokens, normalize=True)
    except TypeError:
        text_features = F.normalize(model.encode_text(tokens), dim=-1)
    return text_features


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, _, tokenizer, preprocess_cfg, _ = load_biomedclip(
        ckpt_dir=args.ckpt_dir,
        model_name=args.model_name,
        device=device,
    )
    model.eval()
    prompts = load_prompts(args.prompts_json)
    text_features = encode_text_prompts(model, tokenizer, prompts, device)
    logit_scale = model.logit_scale.exp() if hasattr(model, "logit_scale") else torch.tensor(1.0, device=device)

    mean = tuple(preprocess_cfg["mean"])
    std = tuple(preprocess_cfg["std"])
    manifest_dir = Path(args.manifest_dir)
    loaders = {
        split: make_loader(manifest_dir / f"{split}.json", args.image_size, args.batch_size, args.num_workers, mean, std)
        for split in ("train", "val", "test")
    }

    test_logits: list[torch.Tensor] = []
    test_labels: list[torch.Tensor] = []
    pib_hits: list[float | None] = []
    split_hits: dict[str, list[float | None]] = {split: [] for split in loaders}
    normal_skipped = 0

    for split, loader in loaders.items():
        for batch in tqdm(loader, desc=f"zeroshot {split}", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"]
            masks = batch["mask_2d"]
            bboxes = batch["bbox"]
            image_features, patch_scores = encode_image_features(
                model,
                images,
                normalize=True,
            )
            logits = logit_scale * image_features @ text_features.t()
            if split == "test":
                test_logits.append(logits.detach().cpu())
                test_labels.append(labels.detach().cpu())

            for i in range(images.shape[0]):
                label_i = int(labels[i].item())
                if label_i == 2:
                    normal_skipped += 1
                    continue
                bbox_i = [int(v.item()) for v in bboxes[i]]
                hit = pib_topk_in_bbox_grid_set(patch_scores[i].detach().cpu(), bbox_i, masks[i], label_i, topk=1)
                pib_hits.append(hit)
                split_hits[split].append(hit)

    cls_metrics = classification_metrics(torch.cat(test_logits), torch.cat(test_labels), num_classes=len(CLASS_NAMES))
    return {
        "mode": "biomedclip_zeroshot",
        "class_names": CLASS_NAMES,
        "prompts": prompts,
        "classification_split": "test",
        "classification": cls_metrics,
        "pib": summarize_pib(pib_hits, topk=1),
        "pib_by_split": {split: summarize_pib(vals, topk=1) for split, vals in split_hits.items()},
        "pib_feature": "visual_cls_query_patch_cosine",
        "pib_channels": 768,
        "normal_skipped_for_pib": normal_skipped,
        "manifests": {split: str(manifest_dir / f"{split}.json") for split in loaders},
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run(args)
    out_json = Path(args.out_json) if args.out_json else out_dir / "zeroshot_busi_eval.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(format_classification_metrics(result["classification"]))
    print(json.dumps({"pib": result["pib"], "pib_by_split": result["pib_by_split"]}, indent=2))
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
