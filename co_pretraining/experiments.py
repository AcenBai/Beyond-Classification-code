from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .paths import ProjectPaths


@dataclass(frozen=True)
class CommandSpec:
    stage: str
    cwd: Path
    argv: tuple[str, ...]
    description: str = ""


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    family: str
    description: str
    outputs: dict[str, str] = field(default_factory=dict)
    commands: dict[str, CommandSpec] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    paper: bool = False


def _py() -> str:
    return sys.executable


def build_registry(paths: ProjectPaths) -> dict[str, ExperimentSpec]:
    py = _py()
    manifest = str(paths.manifest_dir)
    gpu = "{gpu}"
    workers = "{workers}"
    batch = "{batch_size}"
    epochs = "{epochs}"
    root = paths.root
    biomed = str(paths.biomedclip_ckpt_dir)
    lvm = str(paths.lvm_weights)

    registry: dict[str, ExperimentSpec] = {}

    def add(exp: ExperimentSpec) -> None:
        if exp.name in registry:
            raise KeyError(f"Duplicate experiment name: {exp.name}")
        registry[exp.name] = exp

    add(
        ExperimentSpec(
            name="vit_standard",
            family="vanilla",
            paper=True,
            description="ViT-B/16 image classification on BUSI.",
            outputs={
                "metrics": "outputs/vit_standard/eval_test_metrics.json",
                "pib": "outputs/vit_standard/pib_all_splits.json",
                "checkpoint": "outputs/vit_standard/model_final.pth",
            },
            commands={
                "train": CommandSpec(
                    "train",
                    root,
                    (
                        py, "-m", "co_pretraining.runners.train_spatial_collapse",
                        "--backbone", "vit",
                        "--manifest-dir", manifest,
                        "--output-dir", "./outputs/vit_standard",
                        "--gpu", gpu, "--epochs", epochs,
                        "--train-batch-size", "8", "--eval-batch-size", batch,
                        "--num-workers", workers, "--lr", "5e-5", "--run-pib",
                    ),
                    "Train ViT-B/16 (AdamW, lr 5e-5, 50 epochs).",
                ),
                "eval": CommandSpec(
                    "eval",
                    root,
                    (
                        py, "-m", "co_pretraining.runners.train_spatial_collapse",
                        "--backbone", "vit",
                        "--manifest-dir", manifest,
                        "--output-dir", "./outputs/vit_standard",
                        "--gpu", gpu, "--eval-batch-size", batch,
                        "--num-workers", workers, "--eval-only", "--run-pib",
                        "--checkpoint", "./outputs/vit_standard/model_final.pth",
                    ),
                    "Evaluate a trained ViT-B/16 checkpoint.",
                ),
            },
        )
    )

    add(
        ExperimentSpec(
            name="resnet50_standard",
            family="vanilla",
            paper=True,
            description="ResNet-50 classification on BUSI with layer-4 patch scores.",
            outputs={
                "metrics": "outputs/resnet50_standard/eval_test_metrics.json",
                "pib": "outputs/resnet50_standard/pib_all_splits.json",
                "checkpoint": "outputs/resnet50_standard/model_final.pth",
            },
            commands={
                "train": CommandSpec(
                    "train",
                    root,
                    (
                        py, "-m", "co_pretraining.runners.train_spatial_collapse",
                        "--backbone", "resnet50",
                        "--manifest-dir", manifest,
                        "--output-dir", "./outputs/resnet50_standard",
                        "--gpu", gpu, "--epochs", epochs,
                        "--train-batch-size", "8", "--eval-batch-size", batch,
                        "--num-workers", workers, "--lr", "5e-5",
                        "--image-size", "448", "--run-pib",
                    ),
                    "Train ResNet-50 at 448x448 so that layer-4 is 14x14.",
                ),
                "eval": CommandSpec(
                    "eval",
                    root,
                    (
                        py, "-m", "co_pretraining.runners.train_spatial_collapse",
                        "--backbone", "resnet50",
                        "--manifest-dir", manifest,
                        "--output-dir", "./outputs/resnet50_standard",
                        "--gpu", gpu, "--eval-batch-size", batch,
                        "--num-workers", workers, "--eval-only", "--run-pib",
                        "--image-size", "448",
                        "--checkpoint", "./outputs/resnet50_standard/model_final.pth",
                    ),
                    "Evaluate a trained ResNet-50 checkpoint.",
                ),
            },
        )
    )

    add(
        ExperimentSpec(
            name="biomedclip_zeroshot",
            family="BiomedCLIP",
            paper=True,
            description="Zero-shot BUSI classification with BiomedCLIP.",
            outputs={"summary": "outputs/biomedclip_zeroshot/zeroshot_busi_eval.json"},
            commands={
                "eval": CommandSpec(
                    "eval",
                    root,
                    (
                        py, "-m", "co_pretraining.runners.biomedclip_zeroshot_busi",
                        "--manifest-dir", manifest,
                        "--output-dir", "./outputs/biomedclip_zeroshot",
                        "--ckpt-dir", biomed,
                        "--gpu", gpu, "--batch-size", batch, "--num-workers", workers,
                    ),
                    "Zero-shot evaluation with text prompts.",
                )
            },
        )
    )

    add(
        ExperimentSpec(
            name="biomedclip_linear_probe",
            family="BiomedCLIP",
            paper=True,
            description="Linear probe on a frozen BiomedCLIP visual encoder.",
            outputs={
                "metrics": "outputs/biomedclip_linear_probe/eval_test_metrics.json",
                "analysis": "outputs/biomedclip_linear_probe/busi_analysis.json",
                "checkpoint": "outputs/biomedclip_linear_probe/best_probe.pt",
            },
            commands={
                "train": CommandSpec(
                    "train",
                    root,
                    (
                        py, "-m", "co_pretraining.runners.biomedclip_linear_probe_busi",
                        "--stage", "train",
                        "--manifest-dir", manifest,
                        "--output-dir", "./outputs/biomedclip_linear_probe",
                        "--ckpt-dir", biomed,
                        "--gpu", gpu, "--epochs", "20", "--batch-size", "16",
                        "--num-workers", workers, "--lr", "1e-3",
                    ),
                    "Train a linear classifier on frozen BiomedCLIP features.",
                ),
                "eval": CommandSpec(
                    "eval",
                    root,
                    (
                        py, "-m", "co_pretraining.runners.biomedclip_linear_probe_busi",
                        "--stage", "both",
                        "--manifest-dir", manifest,
                        "--output-dir", "./outputs/biomedclip_linear_probe",
                        "--ckpt-dir", biomed,
                        "--probe-checkpoint", "./outputs/biomedclip_linear_probe/best_probe.pt",
                        "--gpu", gpu, "--batch-size", batch, "--num-workers", workers,
                    ),
                    "Evaluate a trained linear probe.",
                ),
            },
        )
    )

    add(
        ExperimentSpec(
            name="biomedclip_finetune",
            family="BiomedCLIP",
            paper=True,
            description="Partial fine-tuning of BiomedCLIP (last 12 visual blocks).",
            outputs={
                "metrics": "outputs/biomedclip_finetune/eval_test_metrics.json",
                "analysis": "outputs/biomedclip_finetune/busi_analysis.json",
                "checkpoint": "outputs/biomedclip_finetune/best_finetune.pt",
            },
            commands={
                "train": CommandSpec(
                    "train",
                    root,
                    (
                        py, "-m", "co_pretraining.runners.biomedclip_finetune_busi",
                        "--stage", "train",
                        "--manifest-dir", manifest,
                        "--output-dir", "./outputs/biomedclip_finetune",
                        "--ckpt-dir", biomed,
                        "--unfreeze-last-n-blocks", "12",
                        "--gpu", gpu, "--epochs", "30", "--batch-size", "8",
                        "--num-workers", workers,
                    ),
                    "Fine-tune the last 12 visual blocks (visual lr 2e-5, head lr 1e-3).",
                ),
                "eval": CommandSpec(
                    "eval",
                    root,
                    (
                        py, "-m", "co_pretraining.runners.biomedclip_finetune_busi",
                        "--stage", "both",
                        "--manifest-dir", manifest,
                        "--output-dir", "./outputs/biomedclip_finetune",
                        "--ckpt-dir", biomed,
                        "--finetune-checkpoint", "./outputs/biomedclip_finetune/best_finetune.pt",
                        "--gpu", gpu, "--batch-size", batch, "--num-workers", workers,
                    ),
                    "Evaluate a fine-tuned BiomedCLIP checkpoint.",
                ),
            },
        )
    )

    add(
        ExperimentSpec(
            name="lvm_med_linear_probe",
            family="LVM-Med",
            paper=True,
            description="Linear probe on a frozen LVM-Med ViT encoder.",
            outputs={
                "metrics": "outputs/lvm_med_linear_probe/eval_test_metrics.json",
                "pib": "outputs/lvm_med_linear_probe/pib_all_splits.json",
                "checkpoint": "outputs/lvm_med_linear_probe/best_model.pth",
            },
            commands={
                "train": CommandSpec(
                    "train",
                    root,
                    (
                        py, "-m", "co_pretraining.runners.lvm_med_cls_busi",
                        "--stage", "train", "--frozen",
                        "--manifest-dir", manifest,
                        "--output-dir", "./outputs/lvm_med_linear_probe",
                        "--lvm-weights", lvm,
                        "--gpu", gpu, "--epochs", "20", "--lr", "5e-4",
                        "--batch-size", "1", "--num-workers", workers,
                    ),
                    "Train a linear head on a frozen LVM-Med encoder.",
                ),
                "eval": CommandSpec(
                    "eval",
                    root,
                    (
                        py, "-m", "co_pretraining.runners.lvm_med_cls_busi",
                        "--stage", "eval", "--frozen",
                        "--manifest-dir", manifest,
                        "--output-dir", "./outputs/lvm_med_linear_probe",
                        "--checkpoint", "./outputs/lvm_med_linear_probe/best_model.pth",
                        "--gpu", gpu, "--batch-size", "1", "--num-workers", workers,
                    ),
                    "Evaluate the linear-probe checkpoint.",
                ),
            },
        )
    )

    add(
        ExperimentSpec(
            name="lvm_med_finetune",
            family="LVM-Med",
            paper=True,
            description="Fine-tuning of the LVM-Med ViT encoder on BUSI.",
            outputs={
                "metrics": "outputs/lvm_med_finetune/eval_test_metrics.json",
                "pib": "outputs/lvm_med_finetune/pib_all_splits.json",
                "checkpoint": "outputs/lvm_med_finetune/best_model.pth",
            },
            commands={
                "train": CommandSpec(
                    "train",
                    root,
                    (
                        py, "-m", "co_pretraining.runners.lvm_med_cls_busi",
                        "--stage", "train", "--no-frozen",
                        "--manifest-dir", manifest,
                        "--output-dir", "./outputs/lvm_med_finetune",
                        "--lvm-weights", lvm,
                        "--gpu", gpu, "--epochs", "30", "--lr", "1e-4",
                        "--batch-size", "1", "--grad-accum-steps", "2",
                        "--num-workers", workers,
                    ),
                    "Fine-tune the LVM-Med encoder (lr 1e-4, 30 epochs).",
                ),
                "eval": CommandSpec(
                    "eval",
                    root,
                    (
                        py, "-m", "co_pretraining.runners.lvm_med_cls_busi",
                        "--stage", "eval", "--no-frozen",
                        "--manifest-dir", manifest,
                        "--output-dir", "./outputs/lvm_med_finetune",
                        "--checkpoint", "./outputs/lvm_med_finetune/best_model.pth",
                        "--gpu", gpu, "--batch-size", "1", "--num-workers", workers,
                    ),
                    "Evaluate the fine-tuned LVM-Med checkpoint.",
                ),
            },
        )
    )

    add(
        ExperimentSpec(
            name="vit_cotraining",
            family="co-train",
            paper=True,
            description="Joint classification and lesion segmentation with ViT-B/16.",
            outputs={
                "metrics": "outputs/vit_cotraining/eval_test_metrics.json",
                "pib": "outputs/vit_cotraining/pib_all_splits_cls.json",
                "checkpoint": "outputs/vit_cotraining/best_joint_model.pt",
            },
            commands={
                "train": CommandSpec(
                    "train",
                    root,
                    (
                        py, "-m", "co_pretraining.runners.train_vanilla_vit_joint",
                        "--manifest-dir", manifest,
                        "--output-dir", "./outputs/vit_cotraining",
                        "--init", "torchvision_imagenet",
                        "--gpu", gpu, "--epochs", epochs,
                        "--image-size", "256",
                        "--train-batch-size", "8", "--eval-batch-size", "8",
                        "--num-workers", workers,
                        "--lambda-seg", "2.0", "--normal-empty-weight", "0.1",
                        "--run-pib",
                    ),
                    "Joint training with a patch-token segmentation decoder.",
                ),
                "eval": CommandSpec(
                    "eval",
                    root,
                    (
                        py, "-m", "co_pretraining.runners.train_vanilla_vit_joint",
                        "--manifest-dir", manifest,
                        "--output-dir", "./outputs/vit_cotraining",
                        "--gpu", gpu, "--image-size", "256",
                        "--eval-batch-size", "8", "--num-workers", workers,
                        "--eval-only", "--run-pib",
                        "--checkpoint", "./outputs/vit_cotraining/best_joint_model.pt",
                    ),
                    "Evaluate a jointly trained checkpoint.",
                ),
            },
        )
    )

    return registry


def format_command(cmd: CommandSpec, values: dict[str, Any]) -> list[str]:
    formatted: list[str] = []
    for part in cmd.argv:
        formatted.append(str(part).format(**values))
    return formatted
