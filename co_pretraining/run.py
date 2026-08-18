from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

from .collect import summarize, summarize_paper, write_summary
from .experiments import build_registry, format_command
from .paths import ProjectPaths, REPO_ROOT


def _print_command(argv: list[str], cwd: Path) -> None:
    print(f"cwd: {cwd}")
    print("cmd:", " ".join(shlex.quote(x) for x in argv))


def _consume_launcher_extra(extra: list[str], values: dict[str, str], dry_run: bool) -> tuple[list[str], dict[str, str], bool]:
    passthrough: list[str] = []
    i = 0
    while i < len(extra):
        item = extra[i]
        if item == "--dry-run":
            dry_run = True
            i += 1
            continue
        if item in {"--gpu", "--workers", "--batch-size", "--epochs"}:
            if i + 1 >= len(extra):
                raise SystemExit(f"{item} requires a value")
            key = item[2:].replace("-", "_")
            values[key] = extra[i + 1]
            i += 2
            continue
        passthrough.append(item)
        i += 1
    return passthrough, values, dry_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate BUSI experiments.")
    parser.add_argument("--root", default=str(REPO_ROOT), help="Repository root.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List registered experiments.")

    p_show = sub.add_parser("show", help="Show one experiment spec.")
    p_show.add_argument("experiment")

    p_run = sub.add_parser("run", help="Run one registered stage.")
    p_run.add_argument("experiment")
    p_run.add_argument("stage")
    p_run.add_argument("--gpu", default="0")
    p_run.add_argument("--workers", default="8")
    p_run.add_argument("--batch-size", default="32")
    p_run.add_argument("--epochs", default="50")
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args appended to the command.")

    p_collect = sub.add_parser("collect", help="Collect existing outputs into a summary table.")
    p_collect.add_argument("--out-dir", default=str(REPO_ROOT / "outputs"))

    args = parser.parse_args()
    paths = ProjectPaths.from_root(args.root)
    registry = build_registry(paths)

    if args.cmd == "list":
        for name, exp in registry.items():
            stages = ",".join(exp.commands)
            flag = "paper" if exp.paper else "extra"
            print(f"{name:42s} {exp.family:12s} {flag:5s} stages=[{stages}]")
        return

    if args.cmd == "show":
        exp = registry[args.experiment]
        payload = {
            "name": exp.name,
            "family": exp.family,
            "paper": exp.paper,
            "description": exp.description,
            "outputs": exp.outputs,
            "stages": list(exp.commands),
            "notes": exp.notes,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        for stage, cmd in exp.commands.items():
            print()
            print(f"[{stage}] {cmd.description}")
            _print_command(format_command(cmd, {"gpu": 0, "workers": 8, "batch_size": 32, "epochs": 50}), cmd.cwd)
        return

    if args.cmd == "run":
        exp = registry[args.experiment]
        if args.stage not in exp.commands:
            raise SystemExit(f"Unknown stage {args.stage!r}. Available: {sorted(exp.commands)}")
        cmd = exp.commands[args.stage]
        values = {
            "gpu": args.gpu,
            "workers": args.workers,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
        }
        extra, values, dry_run = _consume_launcher_extra(list(args.extra), values, args.dry_run)
        argv = format_command(cmd, values) + extra
        _print_command(argv, cmd.cwd)
        if not dry_run:
            subprocess.run(argv, cwd=cmd.cwd, check=True)
        return

    if args.cmd == "collect":
        rows = summarize(paths)
        paper_rows = summarize_paper(paths)
        out_dir = Path(args.out_dir)
        write_summary(rows, out_dir / "summary.csv", out_dir / "summary.json")
        write_summary(paper_rows, out_dir / "paper_busi_compare.csv", out_dir / "paper_busi_compare.json")
        print(f"Wrote {out_dir / 'summary.csv'}")
        print(f"Wrote {out_dir / 'paper_busi_compare.csv'}")
        print()
        print(f"{'experiment':24s} {'method':14s} {'paper_acc':>9s} {'found_acc':>9s} {'paper_pib':>9s} {'found_pib':>9s} source")
        for row in paper_rows:
            acc = "   n/a   " if row["found_acc"] is None else f"{row['found_acc']:.3f}"
            pib = "   n/a   " if row["found_pib"] is None else f"{row['found_pib']:.3f}"
            print(
                f"{row['experiment']:24s} {row['method']:14s} "
                f"{row['paper_acc']:9.3f} {acc:>9s} {row['paper_pib']:9.3f} {pib:>9s} {row['source']}"
            )
        return


if __name__ == "__main__":
    main()
