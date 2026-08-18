from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, ViT_B_32_Weights, resnet50, vit_b_16, vit_b_32
from torchvision.models.vision_transformer import VisionTransformer


class ViTWithPatchScores(VisionTransformer):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._process_input(x)
        n = x.shape[0]
        cls = self.class_token.expand(n, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        cls_token = x[:, 0]
        tokens = x[:, 1:]
        patch_scores = F.cosine_similarity(tokens, cls_token.unsqueeze(1).expand_as(tokens), dim=-1)
        return self.heads(cls_token), patch_scores


class DenseViTWithPatchScores(VisionTransformer):
    def __init__(self, *args: Any, score_mode: str = "original", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if score_mode not in {"original", "inverse"}:
            raise ValueError(f"Unsupported score_mode={score_mode!r}")
        self.score_mode = score_mode
        self.cached_kernel: torch.Tensor | None = None

    @staticmethod
    def gaussian_kernel_1d(kernel_size: int, sigma: float) -> torch.Tensor:
        vals = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1).float()
        kernel = torch.exp(-0.5 * (vals / sigma) ** 2)
        return kernel / torch.max(kernel)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._process_input(x)
        n = x.shape[0]
        cls = self.class_token.expand(n, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        patch_tokens = x[:, 1:]
        if self.cached_kernel is None or self.cached_kernel.device != x.device:
            self.cached_kernel = (
                self.gaussian_kernel_1d(patch_tokens.shape[-1], patch_tokens.shape[-1] ** 0.5)
                .to(x.device)
                .unsqueeze(0)
                .unsqueeze(0)
            )
        x_fft = torch.fft.fft(patch_tokens, dim=-1)
        x_fft = torch.fft.fftshift(x_fft, dim=-1)
        x_fft = x_fft * self.cached_kernel
        x_fft = torch.fft.ifftshift(x_fft, dim=-1)
        x_recon = torch.fft.ifft(x_fft, dim=-1).real
        diff = patch_tokens / (torch.abs(x_recon - patch_tokens) + 1e-6)
        token_select_scores = 1.0 / (torch.abs(diff) + 1e-6) if self.score_mode == "inverse" else diff
        _, top_patch = torch.topk(token_select_scores, k=1, dim=1, largest=True)
        selected = torch.gather(patch_tokens, 1, top_patch)
        pooled_token = torch.mean(selected, dim=1)
        patch_scores = F.cosine_similarity(patch_tokens, pooled_token.unsqueeze(1).expand_as(patch_tokens), dim=-1)
        return self.heads(pooled_token), patch_scores

    @torch.no_grad()
    def vote_counts(self, x: torch.Tensor) -> torch.Tensor:
        x = self._process_input(x)
        n = x.shape[0]
        cls = self.class_token.expand(n, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        patch_tokens = x[:, 1:]
        if self.cached_kernel is None or self.cached_kernel.device != x.device:
            self.cached_kernel = (
                self.gaussian_kernel_1d(patch_tokens.shape[-1], patch_tokens.shape[-1] ** 0.5)
                .to(x.device)
                .unsqueeze(0)
                .unsqueeze(0)
            )
        x_fft = torch.fft.fft(patch_tokens, dim=-1)
        x_fft = torch.fft.fftshift(x_fft, dim=-1)
        x_fft = x_fft * self.cached_kernel
        x_fft = torch.fft.ifftshift(x_fft, dim=-1)
        x_recon = torch.fft.ifft(x_fft, dim=-1).real
        diff = patch_tokens / (torch.abs(x_recon - patch_tokens) + 1e-6)
        token_select_scores = 1.0 / (torch.abs(diff) + 1e-6) if self.score_mode == "inverse" else diff
        top_idx = torch.topk(token_select_scores, k=1, dim=1, largest=True).indices.squeeze(1)
        return F.one_hot(top_idx, num_classes=token_select_scores.shape[1]).sum(dim=1).to(torch.float32)


class ResNet50WithPatchScores(nn.Module):
    def __init__(self, num_classes: int = 3, pretrained: bool = True, feature_layer: str = "layer4") -> None:
        super().__init__()
        if feature_layer not in {"layer3", "layer4"}:
            raise ValueError(f"Unsupported feature_layer={feature_layer!r}")
        self.feature_layer = feature_layer
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)
        in_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        feat_l3 = self.backbone.layer3(x)
        feat_l4 = self.backbone.layer4(feat_l3)
        feat = feat_l4 if self.feature_layer == "layer4" else feat_l3
        pooled = self.backbone.avgpool(feat_l4)
        logits = self.backbone.fc(torch.flatten(pooled, 1))
        q_gap = feat.mean(dim=(2, 3))
        patch_tokens = feat.flatten(2).transpose(1, 2)
        patch_scores = F.cosine_similarity(patch_tokens, q_gap.unsqueeze(1).expand_as(patch_tokens), dim=-1)
        return logits, patch_scores


def _load_vit_weights_with_pos_resize(model: VisionTransformer, state_dict: dict[str, torch.Tensor]) -> None:
    state = OrderedDict((k, v) for k, v in state_dict.items())
    pos_key = "encoder.pos_embedding"
    if pos_key in state and pos_key in model.state_dict():
        src_pos = state[pos_key]
        dst_pos = model.state_dict()[pos_key]
        if src_pos.shape != dst_pos.shape:
            cls_src = src_pos[:, :1, :]
            patch_src = src_pos[:, 1:, :]
            src_hw = int(math.sqrt(patch_src.shape[1]))
            dst_hw = int(math.sqrt(dst_pos.shape[1] - 1))
            if src_hw * src_hw == patch_src.shape[1] and dst_hw * dst_hw == (dst_pos.shape[1] - 1):
                patch_src = patch_src.reshape(1, src_hw, src_hw, -1).permute(0, 3, 1, 2)
                patch_src = F.interpolate(patch_src, size=(dst_hw, dst_hw), mode="bicubic", align_corners=False)
                patch_src = patch_src.permute(0, 2, 3, 1).reshape(1, dst_hw * dst_hw, -1)
                state[pos_key] = torch.cat([cls_src, patch_src], dim=1)
            else:
                state.pop(pos_key, None)
    model.load_state_dict(state, strict=False)


def unwrap_state_dict(ckpt: Any) -> dict[str, torch.Tensor]:
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
    elif isinstance(ckpt, dict):
        tensor_keys = [k for k, v in ckpt.items() if torch.is_tensor(v)]
        state = {k: ckpt[k] for k in tensor_keys} if tensor_keys else ckpt
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")
    keys = list(state.keys())
    if keys and sum(k.startswith("model.") for k in keys) > len(keys) / 2:
        state = {k[len("model.") :]: v for k, v in state.items() if k.startswith("model.")}
    return state


def load_spatial_collapse_checkpoint(model: nn.Module, path: str, strict: bool = False) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = unwrap_state_dict(ckpt)
    result = model.load_state_dict(state, strict=strict)
    return {
        "checkpoint": path,
        "missing": list(result.missing_keys),
        "unexpected": list(result.unexpected_keys),
        "n_loaded": len(state),
    }


def build_spatial_collapse_model(
    backbone: str = "vit",
    num_classes: int = 3,
    pretrained: bool = True,
    image_size: int = 224,
    patch_size: int = 16,
    dense_score_mode: str = "original",
    resnet_feature_layer: str = "layer4",
) -> nn.Module:
    if backbone == "resnet50":
        return ResNet50WithPatchScores(num_classes=num_classes, pretrained=pretrained, feature_layer=resnet_feature_layer)
    if patch_size not in {16, 32}:
        raise ValueError(f"Unsupported patch_size={patch_size!r}")
    weights = ViT_B_16_Weights.IMAGENET1K_V1 if patch_size == 16 else ViT_B_32_Weights.IMAGENET1K_V1
    if not pretrained:
        weights = None
    if backbone in {"dense", "dense_inv"}:
        model = DenseViTWithPatchScores(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            num_classes=1000,
            score_mode="inverse" if backbone == "dense_inv" else dense_score_mode,
        )
    elif backbone == "vit":
        model = ViTWithPatchScores(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            num_classes=1000,
        )
    else:
        raise ValueError(f"Unsupported backbone={backbone!r}")
    if weights is not None:
        base_model = vit_b_16(weights=weights) if patch_size == 16 else vit_b_32(weights=weights)
        _load_vit_weights_with_pos_resize(model, base_model.state_dict())
    in_dim = model.heads.head.in_features
    model.heads.head = nn.Linear(in_dim, num_classes)
    return model
