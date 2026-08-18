from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.transform import resize
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset


LABEL_MAP = {"benign": 0, "malignant": 1, "normal": 2}
CLASS_NAMES = ["benign", "malignant", "normal"]


def resolve_busi_path(path: str) -> str:
    """Resolve a split-file path against BUSI_DATA_ROOT when it is not absolute."""
    expanded = os.path.expandvars(path)
    candidate = Path(expanded)
    if candidate.is_absolute():
        return str(candidate)
    root = os.environ.get("BUSI_DATA_ROOT", "").strip()
    if root:
        return str(Path(root) / candidate)
    from co_pretraining.paths import default_busi_data_root

    return str(default_busi_data_root() / candidate)


def _load_manifest(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Manifest must be a JSON list: {path}")
    return data


def _merge_mask_paths(mask_paths: Sequence[str], ref_hw: tuple[int, int]) -> np.ndarray:
    h, w = ref_hw
    merged = np.zeros((h, w), dtype=np.uint8)
    for mp in mask_paths:
        if not mp or not os.path.exists(mp):
            continue
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        merged = np.maximum(merged, (m > 0).astype(np.uint8) * 255)
    return merged


def bbox_from_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        h, w = mask.shape[:2]
        return np.array([0.0, 0.0, float(w - 1), float(h - 1)], dtype=np.float32)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def preprocess_image_and_mask(
    image_bgr: np.ndarray,
    mask_full: np.ndarray,
    image_size: int,
    mask_size: int | None = None,
    normalize_0_255: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    mask_size = image_size if mask_size is None else mask_size
    img = resize(
        image_bgr,
        (image_size, image_size),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype("uint8")
    if normalize_0_255:
        img = ((img - img.min()) * (1 / (0.01 + img.max() - img.min()) * 255)).astype("uint8")
    mask = resize(
        mask_full,
        (mask_size, mask_size),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype("uint8")
    mask[mask > 0] = 1
    return img, mask


class BUSIManifestDataset(Dataset):
    """BUSI images and lesion masks listed by a JSON split file."""

    def __init__(
        self,
        manifest_path: str | Path,
        image_size: int = 224,
        mask_size: int | None = None,
        normalize_mean: tuple[float, float, float] | None = None,
        normalize_std: tuple[float, float, float] | None = None,
        image_backend: Literal["cv2_bgr", "pil_rgb"] = "cv2_bgr",
        cache: bool = False,
    ) -> None:
        self.rows = _load_manifest(manifest_path)
        self.manifest_path = str(manifest_path)
        self.image_size = int(image_size)
        self.mask_size = int(mask_size if mask_size is not None else image_size)
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.image_backend = image_backend
        self.cache_enabled = cache
        self.cache: dict[int, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.cache_enabled and idx in self.cache:
            return self.cache[idx]

        row = self.rows[idx]
        img_path = resolve_busi_path(row["image_path"])
        mask_paths = [resolve_busi_path(p) for p in (row.get("mask_paths") or [])]
        label = int(row.get("label", -1))
        sample_id = str(row.get("id", str(idx)))

        if self.image_backend == "pil_rgb":
            image = Image.open(img_path).convert("RGB")
            w0, h0 = image.size
            mask_full = _merge_mask_paths(mask_paths, (h0, w0))
            mask_img = Image.fromarray(mask_full, mode="L")
            image = TF.resize(image, [self.image_size, self.image_size], antialias=True)
            mask_img = TF.resize(mask_img, [self.mask_size, self.mask_size], interpolation=Image.NEAREST)
            image_t = TF.to_tensor(image)
            if self.normalize_mean is not None and self.normalize_std is not None:
                image_t = TF.normalize(image_t, self.normalize_mean, self.normalize_std)
            else:
                image_t = image_t * 255.0
            mask = (TF.to_tensor(mask_img).squeeze(0).numpy() > 0).astype(np.uint8)
        else:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                raise FileNotFoundError(img_path)
            h0, w0 = img_bgr.shape[:2]
            mask_full = _merge_mask_paths(mask_paths, (h0, w0))
            img, mask = preprocess_image_and_mask(
                img_bgr,
                mask_full,
                image_size=self.image_size,
                mask_size=self.mask_size,
                normalize_0_255=True,
            )
            image_t = torch.as_tensor(img.copy()).permute(2, 0, 1).float().contiguous()
            if self.normalize_mean is not None and self.normalize_std is not None:
                image_t = image_t / 255.0
                mean = torch.tensor(self.normalize_mean, dtype=image_t.dtype).view(3, 1, 1)
                std = torch.tensor(self.normalize_std, dtype=image_t.dtype).view(3, 1, 1)
                image_t = (image_t - mean) / std
        bbox = bbox_from_mask(mask)

        mask_t = torch.as_tensor(mask[None, :, :].copy()).long()
        data = {
            "image": image_t,
            "mask": mask_t,
            "mask_2d": mask_t.squeeze(0),
            "mask_ete": mask_t.squeeze(0),
            "bbox": torch.tensor(bbox).float(),
            "bboxes": torch.tensor(bbox).float(),
            "label": torch.tensor(label, dtype=torch.long),
            "id": sample_id,
            "mask_file": ";".join(mask_paths) if mask_paths else "",
            "img_file": img_path,
        }
        if self.cache_enabled:
            self.cache[idx] = data
        return data
