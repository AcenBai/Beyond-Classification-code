from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


def default_manifest_dir() -> Path:
    return _env_path("BUSI_MANIFEST_DIR", REPO_ROOT / "configs" / "busi_splits")


def default_output_dir() -> Path:
    return _env_path("OUTPUT_DIR", REPO_ROOT / "outputs")


def default_biomedclip_dir() -> Path:
    return _env_path("BIOMEDCLIP_CKPT_DIR", REPO_ROOT / "checkpoints" / "BiomedCLIP")


def default_lvm_weights() -> Path:
    return _env_path("LVM_MED_WEIGHTS", REPO_ROOT / "checkpoints" / "lvmmed_vit.pth")


def default_busi_data_root() -> Path:
    return _env_path("BUSI_DATA_ROOT", REPO_ROOT / "data" / "BUSI")


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    manifest_dir: Path
    output_root: Path
    biomedclip_ckpt_dir: Path
    lvm_weights: Path
    busi_data_root: Path

    @property
    def package_root(self) -> Path:
        return self.root

    @classmethod
    def from_root(cls, root: str | Path | None = None) -> "ProjectPaths":
        root_path = Path(root) if root else REPO_ROOT
        return cls(
            root=root_path,
            manifest_dir=_env_path("BUSI_MANIFEST_DIR", root_path / "configs" / "busi_splits"),
            output_root=_env_path("OUTPUT_DIR", root_path / "outputs"),
            biomedclip_ckpt_dir=_env_path("BIOMEDCLIP_CKPT_DIR", root_path / "checkpoints" / "BiomedCLIP"),
            lvm_weights=_env_path("LVM_MED_WEIGHTS", root_path / "checkpoints" / "lvmmed_vit.pth"),
            busi_data_root=_env_path("BUSI_DATA_ROOT", root_path / "data" / "BUSI"),
        )
