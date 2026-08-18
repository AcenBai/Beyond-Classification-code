from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ViT_B_16_Weights, vit_b_16


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)


def imagenet_preprocess(x: torch.Tensor) -> torch.Tensor:
    x = x / 255.0
    mean = IMAGENET_MEAN.to(device=x.device, dtype=x.dtype)
    std = IMAGENET_STD.to(device=x.device, dtype=x.dtype)
    return (x - mean) / std


def patch_scores(patch_tokens: torch.Tensor, cls_token: torch.Tensor, query: str = "cls") -> torch.Tensor:
    if query == "cls":
        q = cls_token.unsqueeze(0)
    elif query == "mean":
        q = patch_tokens.mean(dim=0, keepdim=True)
    else:
        raise ValueError(f"Unsupported query={query!r}")
    return F.cosine_similarity(patch_tokens, q.expand_as(patch_tokens), dim=-1)


class ConvBNAct(nn.Module):
    def __init__(self, in_chans: int, out_chans: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.GELU(),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ViTPatchDecoder(nn.Module):
    def __init__(self, in_chans: int = 768) -> None:
        super().__init__()
        self.stage1 = ConvBNAct(in_chans, 512)
        self.stage2 = ConvBNAct(512, 256)
        self.stage3 = ConvBNAct(256, 128)
        self.stage4 = ConvBNAct(128, 64)
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.stage2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.stage3(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.stage4(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.out(x)


class VanillaViTB16Joint(nn.Module):
    """ViT-B/16 with a classification head and a patch-token segmentation decoder."""

    def __init__(
        self,
        num_classes: int = 3,
        image_size: int = 256,
        use_cls_seg_fusion: bool = False,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.patch_size = 16
        self.grid_size = self.image_size // self.patch_size
        self.use_cls_seg_fusion = bool(use_cls_seg_fusion)
        self.vit = vit_b_16(weights=None, image_size=self.image_size, num_classes=num_classes)
        self.vit.heads = nn.Identity()
        self.cls_head = nn.Linear(768, num_classes)
        self.cls_to_patch = nn.Linear(768, 768) if self.use_cls_seg_fusion else nn.Identity()
        self.seg_decoder = ViTPatchDecoder(in_chans=768)

    def forward_tokens(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = imagenet_preprocess(images)
        x = self.vit._process_input(x)
        batch = x.shape[0]
        cls = self.vit.class_token.expand(batch, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.vit.encoder(x)
        return x[:, 0], x[:, 1:]

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        cls_token, patch_tokens = self.forward_tokens(images)
        b, _, c = patch_tokens.shape
        patch_map = patch_tokens.transpose(1, 2).reshape(b, c, self.grid_size, self.grid_size)
        if self.use_cls_seg_fusion:
            patch_map = patch_map + self.cls_to_patch(cls_token).view(b, c, 1, 1)
        return {
            "logits": self.cls_head(cls_token),
            "mask_logits": self.seg_decoder(patch_map),
            "cls_token": cls_token,
            "patch_tokens": patch_tokens,
            "patch_map": patch_map,
        }


def _interpolate_pos_embedding(value: torch.Tensor, image_size: int, patch_size: int = 16) -> torch.Tensor:
    cls_pos = value[:, :1]
    patch_pos = value[:, 1:]
    old_grid = int(round(patch_pos.shape[1] ** 0.5))
    new_grid = image_size // patch_size
    if old_grid == new_grid:
        return value
    patch_pos = patch_pos.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(patch_pos, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, -1)
    return torch.cat([cls_pos, patch_pos], dim=1)


def _torchvision_imagenet_state() -> dict[str, torch.Tensor]:
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    return weights.get_state_dict(progress=True, check_hash=True)


def load_vanilla_vit_init(
    model: VanillaViTB16Joint,
    init: str = "torchvision_imagenet",
    checkpoint: str | None = None,
) -> dict[str, Any]:
    if init == "random":
        return {"init": init, "loaded": 0, "missing": [], "unexpected": []}
    if init == "torchvision_imagenet":
        state = _torchvision_imagenet_state()
        checkpoint_name = "torchvision://ViT_B_16_Weights.IMAGENET1K_V1"
    elif init == "torchvision_vit_ckpt":
        if not checkpoint:
            raise ValueError("--encoder-checkpoint is required for --init torchvision_vit_ckpt")
        ckpt = torch.load(checkpoint, map_location="cpu")
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        checkpoint_name = checkpoint
    else:
        raise ValueError(f"Unsupported init={init!r}")

    dst_state = model.vit.state_dict()
    mapped: OrderedDict[str, torch.Tensor] = OrderedDict()
    skipped: list[str] = []
    ignored: list[str] = []
    for key, value in state.items():
        clean_key = key.removeprefix("model.")
        clean_key = clean_key.replace(".mlp.linear_1.", ".mlp.0.")
        clean_key = clean_key.replace(".mlp.linear_2.", ".mlp.3.")
        if clean_key.startswith("heads."):
            ignored.append(key)
            continue
        if clean_key == "encoder.pos_embedding":
            value = _interpolate_pos_embedding(value, image_size=model.image_size, patch_size=model.patch_size)
        if clean_key not in dst_state or tuple(value.shape) != tuple(dst_state[clean_key].shape):
            skipped.append(key)
            continue
        mapped[clean_key] = value

    result = model.vit.load_state_dict(mapped, strict=False)
    return {
        "init": init,
        "checkpoint": checkpoint_name,
        "loaded": len(mapped),
        "missing": list(result.missing_keys),
        "unexpected": list(result.unexpected_keys),
        "skipped": skipped,
        "ignored": ignored,
        "note": "Loaded ViT-B/16 backbone; interpolated absolute position embedding when image_size differs from 224.",
    }
