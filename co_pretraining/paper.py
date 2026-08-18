from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperBusiRow:
    experiment: str
    family: str
    model: str
    method: str
    paper_acc: float
    paper_pib: float


# ACC is test. PIB is train/val/test, excluding normal images.
PAPER_BUSI_ROWS: tuple[PaperBusiRow, ...] = (
    PaperBusiRow("vit_standard", "vanilla", "ViT-B/16", "Standard", 0.888, 0.071),
    PaperBusiRow("resnet50_standard", "vanilla", "ResNet50-L4", "Standard", 0.862, 0.690),
    PaperBusiRow("biomedclip_zeroshot", "BiomedCLIP", "BiomedCLIP", "Zero-shot", 0.362, 0.346),
    PaperBusiRow("biomedclip_linear_probe", "BiomedCLIP", "BiomedCLIP", "Linear probe", 0.759, 0.346),
    PaperBusiRow("biomedclip_finetune", "BiomedCLIP", "BiomedCLIP", "Fine-tuning", 0.888, 0.385),
    PaperBusiRow("lvm_med_linear_probe", "LVM-Med", "LVM-Med", "Linear probe", 0.750, 0.135),
    PaperBusiRow("lvm_med_finetune", "LVM-Med", "LVM-Med", "Fine-tuning", 0.715, 0.114),
    PaperBusiRow("vit_cotraining", "co-train", "ViT-B/16", "Co-training", 0.897, 0.128),
)
