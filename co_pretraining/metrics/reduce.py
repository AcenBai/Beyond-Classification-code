from __future__ import annotations

from typing import Any

import numpy as np


def reduce_metric_dict(items: list[dict[str, Any]]) -> dict[str, float]:
    keys = sorted({k for item in items for k in item})
    out: dict[str, float] = {}
    for key in keys:
        vals = [float(item[key]) for item in items if key in item and not np.isnan(float(item[key]))]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out
