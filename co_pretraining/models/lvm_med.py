from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lvm_med_encoder import vit_encoder_b


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_imagenet(images: torch.Tensor) -> torch.Tensor:
    x = images.float()
    if float(x.max().item()) > 1.5:
        x = x / 255.0
    mean = IMAGENET_MEAN.to(device=x.device, dtype=x.dtype)
    std = IMAGENET_STD.to(device=x.device, dtype=x.dtype)
    return (x - mean) / std


def official_lvm_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    weight = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(weight, dict) and "state_dict" in weight:
        weight = weight["state_dict"]
    if not isinstance(weight, dict):
        raise TypeError(f"Unexpected LVM-Med checkpoint type: {type(weight)}")
    prefixed: dict[str, torch.Tensor] = {}
    for key, value in weight.items():
        if not torch.is_tensor(value):
            continue
        if key.startswith("model.") or key.startswith("fc.") or key.startswith("avgpool."):
            prefixed[key] = value
        else:
            prefixed[f"model.{key}"] = value
    return prefixed


class LVMMedClassifier(nn.Module):
    """LVM-Med ViT encoder with a linear classification head."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.encoder = vit_encoder_b(num_classes=1000)
        in_dim = int(self.encoder.fc.in_features)
        self.encoder.fc = nn.Linear(in_dim, num_classes)
        self.num_classes = int(num_classes)

    def freeze_encoder(self) -> None:
        for name, param in self.encoder.named_parameters():
            if name.startswith("fc."):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def load_official_encoder(self, path: str | Path, strict: bool = False) -> None:
        state = official_lvm_state_dict(path)
        missing, unexpected = self.encoder.load_state_dict(state, strict=strict)
        if missing or unexpected:
            print(f"LVM-Med load: missing={len(missing)} unexpected={len(unexpected)}")

    def load_busi_checkpoint(self, path: str | Path) -> dict:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        self.encoder.load_state_dict(state, strict=True)
        return ckpt if isinstance(ckpt, dict) else {"state_dict": state}

    def forward_logits_and_scores(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = normalize_imagenet(images)
        feat = self.encoder.model(x)
        pooled = self.encoder.avgpool(feat).flatten(1)
        logits = self.encoder.fc(pooled)
        patch_tokens = feat.flatten(2).transpose(1, 2)
        patch_scores = F.cosine_similarity(patch_tokens, pooled.unsqueeze(1).expand_as(patch_tokens), dim=-1)
        return logits, patch_scores

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_logits_and_scores(images)
        return logits
