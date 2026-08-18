from __future__ import annotations

import os
import sys
from pathlib import Path

from co_pretraining.paths import default_lvm_weights

DEFAULT_LVM_WEIGHTS = default_lvm_weights()
DEFAULT_LVM_MED_REPO = Path(os.environ.get("LVM_MED_REPO", "third_party/LVM-Med"))


def ensure_lvm_med_repo(repo_root: str | Path = DEFAULT_LVM_MED_REPO) -> Path:
    """Locate a local LVM-Med clone. Needed only for the frozen-encoder PIB script."""
    root = Path(repo_root)
    if not (root / "segment_anything" / "our_vit.py").exists():
        raise FileNotFoundError(
            f"LVM-Med repository not found at {root}. Set LVM_MED_REPO to the clone path. "
            "Linear probe and fine-tuning need only lvmmed_vit.pth (LVM_MED_WEIGHTS)."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
