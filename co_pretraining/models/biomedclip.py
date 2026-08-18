from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS

from co_pretraining.data import CLASS_NAMES
from co_pretraining.paths import default_biomedclip_dir


DEFAULT_CKPT_DIR = str(default_biomedclip_dir())
DEFAULT_MODEL_NAME = "biomedclip_local"
DEFAULT_UNFREEZE_LAST_N_BLOCKS = 12
HF_HUB_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def resolve_ckpt_dir(ckpt_dir: str | Path | None = None) -> Path:
    path = Path(ckpt_dir) if ckpt_dir else Path(os.environ.get("BIOMEDCLIP_CKPT_DIR", DEFAULT_CKPT_DIR))
    return path


def load_biomedclip_config(ckpt_dir: str | Path) -> dict[str, Any]:
    cfg_path = Path(ckpt_dir) / "open_clip_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"BiomedCLIP config not found: {cfg_path}. "
            "Download microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 "
            "(open_clip_config.json + open_clip_pytorch_model.bin)."
        )
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def register_model_config(ckpt_dir: str | Path, model_name: str = DEFAULT_MODEL_NAME) -> dict[str, Any]:
    config = load_biomedclip_config(ckpt_dir)
    model_cfg = config["model_cfg"]
    if not model_name.startswith(HF_HUB_PREFIX) and model_name not in _MODEL_CONFIGS:
        _MODEL_CONFIGS[model_name] = model_cfg
    return config


def load_biomedclip(
    ckpt_dir: str | Path | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | torch.device = "cpu",
):
    """Load BiomedCLIP from an OpenCLIP checkpoint directory."""
    ckpt_dir = resolve_ckpt_dir(ckpt_dir)
    weight_path = ckpt_dir / "open_clip_pytorch_model.bin"
    if not weight_path.exists():
        raise FileNotFoundError(
            f"BiomedCLIP weight not found: {weight_path}. "
            f"Place the OpenCLIP checkpoint there, or set BIOMEDCLIP_CKPT_DIR. "
            f"HuggingFace id: {HF_HUB_ID}"
        )
    config = register_model_config(ckpt_dir, model_name)
    preprocess_cfg = config["preprocess_cfg"]
    model, _, preprocess = create_model_and_transforms(
        model_name=model_name,
        pretrained=str(weight_path),
        **{f"image_{k}": v for k, v in preprocess_cfg.items()},
    )
    tokenizer = get_tokenizer(model_name)
    model = model.to(device)
    embed_dim = int(config["model_cfg"]["embed_dim"])
    return model, preprocess, tokenizer, preprocess_cfg, embed_dim


def forward_visual_tokens(visual: nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    trunk = visual.trunk
    tokens = trunk.forward_features(images)
    n_prefix = int(getattr(trunk, "num_prefix_tokens", 1))
    return tokens, tokens[:, n_prefix:]


def encode_image_features(
    clip_model: nn.Module,
    images: torch.Tensor,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CLS-pooled image features and CLS-to-patch cosine scores."""
    visual = clip_model.visual
    tokens, patch_tokens = forward_visual_tokens(visual, images)
    raw_cls = tokens[:, 0]
    pooled_768 = visual.trunk.forward_head(tokens, pre_logits=True)
    feat = visual.head(pooled_768)
    if normalize:
        feat = F.normalize(feat, dim=-1)
    patch_scores = F.cosine_similarity(patch_tokens, raw_cls.unsqueeze(1).expand_as(patch_tokens), dim=-1)
    return feat, patch_scores


class BiomedCLIPLinearProbe(nn.Module):
    def __init__(self, clip_model: nn.Module, num_classes: int = 3, embed_dim: int = 512) -> None:
        super().__init__()
        self.clip = clip_model
        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)
        for p in self.clip.parameters():
            p.requires_grad = False
        self.head = nn.Linear(self.embed_dim, self.num_classes)
        nn.init.zeros_(self.head.bias)

    def forward(self, images: torch.Tensor, return_patch_scores: bool = False):
        with torch.no_grad():
            feats, patch_scores = encode_image_features(self.clip, images, normalize=True)
        logits = self.head(feats)
        if return_patch_scores:
            return logits, patch_scores
        return logits


def configure_partial_visual_unfreeze(clip_model: nn.Module, unfreeze_last_n_blocks: int) -> None:
    for p in clip_model.parameters():
        p.requires_grad = False
    visual = clip_model.visual
    for p in visual.head.parameters():
        p.requires_grad = True
    blocks = visual.trunk.blocks
    n = min(int(unfreeze_last_n_blocks), len(blocks))
    for block in blocks[-n:]:
        for p in block.parameters():
            p.requires_grad = True


def trainable_visual_parameters(clip_model: nn.Module) -> Iterable[nn.Parameter]:
    return (p for p in clip_model.visual.parameters() if p.requires_grad)


class BiomedCLIPPartialFinetune(nn.Module):
    def __init__(
        self,
        clip_model: nn.Module,
        num_classes: int = 3,
        embed_dim: int = 512,
        unfreeze_last_n_blocks: int = DEFAULT_UNFREEZE_LAST_N_BLOCKS,
    ) -> None:
        super().__init__()
        self.clip = clip_model
        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)
        self.unfreeze_last_n_blocks = int(unfreeze_last_n_blocks)
        configure_partial_visual_unfreeze(self.clip, self.unfreeze_last_n_blocks)
        self.head = nn.Linear(self.embed_dim, self.num_classes)
        nn.init.zeros_(self.head.bias)
        self.pooling_mode = "cls"
        self.lazy_strike_topk = 1

    def ranking_scores(self, images: torch.Tensor, patch_scores: torch.Tensor) -> torch.Tensor:
        return patch_scores

    def forward(self, images: torch.Tensor, return_patch_scores: bool = False):
        feats, patch_scores = encode_image_features(self.clip, images, normalize=True)
        logits = self.head(feats)
        if return_patch_scores:
            return logits, patch_scores
        return logits

    def set_train_mode(self) -> None:
        self.train()
        self.clip.eval()
        self.clip.visual.train()
        for p in self.clip.visual.parameters():
            if not p.requires_grad:
                p.requires_grad = False


def save_probe_checkpoint(path: str | Path, probe: BiomedCLIPLinearProbe, extra: Optional[dict] = None) -> None:
    payload = {
        "head": probe.head.state_dict(),
        "num_classes": probe.num_classes,
        "embed_dim": probe.embed_dim,
        "pooling_mode": "cls",
        "class_names": CLASS_NAMES,
    }
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_probe_checkpoint(
    path: str | Path,
    clip_model: nn.Module,
    device: str | torch.device = "cpu",
) -> BiomedCLIPLinearProbe:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    probe = BiomedCLIPLinearProbe(
        clip_model=clip_model,
        num_classes=int(ckpt.get("num_classes", 3)),
        embed_dim=int(ckpt.get("embed_dim", 512)),
    )
    probe.head.load_state_dict(ckpt["head"])
    return probe.to(device)


def save_finetune_checkpoint(path: str | Path, model: BiomedCLIPPartialFinetune, extra: Optional[dict] = None) -> None:
    payload = {
        "head": model.head.state_dict(),
        "visual": model.clip.visual.state_dict(),
        "num_classes": model.num_classes,
        "embed_dim": model.embed_dim,
        "pooling_mode": "cls",
        "lazy_strike_topk": 1,
        "unfreeze_last_n_blocks": model.unfreeze_last_n_blocks,
        "class_names": CLASS_NAMES,
    }
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_finetune_checkpoint(
    path: str | Path,
    clip_model: nn.Module,
    device: str | torch.device = "cpu",
) -> BiomedCLIPPartialFinetune:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    model = BiomedCLIPPartialFinetune(
        clip_model=clip_model,
        num_classes=int(ckpt.get("num_classes", 3)),
        embed_dim=int(ckpt.get("embed_dim", 512)),
        unfreeze_last_n_blocks=int(ckpt.get("unfreeze_last_n_blocks", DEFAULT_UNFREEZE_LAST_N_BLOCKS)),
    )
    model.clip.visual.load_state_dict(ckpt["visual"], strict=True)
    model.head.load_state_dict(ckpt["head"])
    return model.to(device)
