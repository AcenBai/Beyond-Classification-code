from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .experiments import build_registry
from .paper import PAPER_BUSI_ROWS
from .paths import ProjectPaths


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            last: dict[str, Any] | None = None
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    last = obj
            return last
        except Exception:
            return None


def _dig(obj: Any, *keys: str) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _first_number(*values: Any) -> float | None:
    for value in values:
        if isinstance(value, dict) and "pib" in value:
            try:
                return float(value["pib"])
            except (TypeError, ValueError):
                continue
        if isinstance(value, (int, float)) and value == value:
            return float(value)
    return None


def _abs_for_output(rel: str, paths: ProjectPaths) -> Path:
    if rel.startswith("outputs/"):
        return paths.root / rel
    return Path(rel)


def _extract_acc(blob: dict[str, Any] | None) -> float | None:
    if not blob:
        return None
    return _first_number(
        _dig(blob, "classification", "acc"),
        blob.get("acc"),
        blob.get("acc_mean"),
        (blob.get("trials") or [{}])[0].get("acc") if isinstance(blob.get("trials"), list) and blob.get("trials") else None,
    )


def _extract_pib(blob: dict[str, Any] | None) -> float | None:
    if not blob:
        return None
    return _first_number(
        _dig(blob, "summary", "pib"),
        _dig(blob, "metrics", "pib"),
        blob.get("pib") if not isinstance(blob.get("pib"), dict) else blob["pib"].get("pib"),
        _dig(blob, "pib", "pib"),
        _dig(blob, "cls_query", "pib"),
        blob.get("pib_mean"),
    )


def summarize(paths: ProjectPaths) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for exp in build_registry(paths).values():
        row: dict[str, Any] = {
            "experiment": exp.name,
            "family": exp.family,
            "paper": exp.paper,
            "description": exp.description,
        }
        loaded: dict[str, dict[str, Any] | None] = {}
        for key, rel in exp.outputs.items():
            p = _abs_for_output(rel, paths)
            row[f"path_{key}"] = str(p)
            loaded[key] = _load_json(p) if p.suffix == ".json" else None
            row[f"exists_{key}"] = p.exists()

        metrics = loaded.get("metrics") or loaded.get("summary")
        row["acc"] = _extract_acc(metrics)
        row["pib"] = _extract_pib(metrics)
        analysis = loaded.get("analysis")
        if analysis:
            row["pib"] = _extract_pib(analysis) or row.get("pib")
        pib = loaded.get("pib")
        if pib:
            row["pib"] = _extract_pib(pib) or row.get("pib")
        rows.append(row)
    return rows


def summarize_paper(paths: ProjectPaths) -> list[dict[str, Any]]:
    local_rows = {row["experiment"]: row for row in summarize(paths)}
    out: list[dict[str, Any]] = []
    for spec in PAPER_BUSI_ROWS:
        local = local_rows.get(spec.experiment, {})
        acc = local.get("acc")
        pib = local.get("pib")
        out.append(
            {
                "experiment": spec.experiment,
                "model": spec.model,
                "method": spec.method,
                "paper_acc": spec.paper_acc,
                "paper_pib": spec.paper_pib,
                "found_acc": acc,
                "found_pib": pib,
                "acc_delta": None if acc is None else float(acc) - spec.paper_acc,
                "pib_delta": None if pib is None else float(pib) - spec.paper_pib,
                "source": "outputs" if acc is not None or pib is not None else "missing",
            }
        )
    return out


def write_summary(rows: list[dict[str, Any]], out_csv: Path, out_json: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
