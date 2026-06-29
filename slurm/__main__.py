"""CLI for Vulcan SLURM experiment grids.

Usage (from repo root):
  python -m slurm list
  python -m slurm submit EXPERIMENT [--dry-run] [--include-active]
  python -m slurm submit EXPERIMENT --account ACCT

Set SLURM_ACCOUNT or pass --account for sbatch.
"""

import argparse
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from slurm.experiments import EXPERIMENTS
from slurm.grid import (
    REPO_ROOT,
    SBATCH_SCRIPT,
    SCRIPT_DIR,
    Experiment,
    job_name,
    tasks_to_submit,
)
from slurm.queue import array_spec


def _sbatch_args(
    exp: Experiment,
    label: str,
    to_submit: list[int],
    account: str | None,
) -> list[str]:
    name = job_name(exp, label)
    args = [
        f"--job-name={name}",
        f"--export=ALL,REPO_ROOT={REPO_ROOT},EXPERIMENT={exp.name}",
        f"--array={array_spec(to_submit)}",
        str(SBATCH_SCRIPT),
    ]
    if account:
        args.insert(1, f"--account={account}")
    return args


def _tasks_by_label(exp: Experiment, task_ids: list[int]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for task_id in task_ids:
        groups[exp.configs[task_id].label].append(task_id)
    return dict(groups)


def _cmd_list(experiments: dict[str, Experiment]) -> int:
    for name in sorted(experiments):
        print(f"{name}: {experiments[name].grid_line()}")
    return 0


def _cmd_submit(args: argparse.Namespace, experiments: dict[str, Experiment]) -> int:
    exp = experiments[args.experiment]

    if not (REPO_ROOT / ".venv").is_dir():
        print(
            f"No .venv in {REPO_ROOT} — run {SCRIPT_DIR / 'setup.sh'}", file=sys.stderr
        )
        return 1

    to_submit, complete, in_progress = tasks_to_submit(
        exp, skip_active=not args.include_active
    )
    by_label = _tasks_by_label(exp, to_submit)

    if args.dry_run:
        print(f"Experiment: {exp.name}")
        print(f"Grid: {exp.grid_line()}")
        print(
            f"Complete: {complete} | In progress: {in_progress} | "
            f"To submit: {len(to_submit)}"
        )
        for label, task_ids in sorted(by_label.items()):
            sbatch_args = _sbatch_args(exp, label, task_ids, args.account)
            label_note = f" (label={label!r})" if label else ""
            print(f"Would run{label_note}: sbatch {' '.join(sbatch_args)}")
        return 0

    if not to_submit:
        print("Nothing to submit (all tasks complete or already queued/running).")
        return 0

    scratch = Path("/scratch") / os.environ.get("USER", "") / "logs" / "mue"
    for label, task_ids in sorted(by_label.items()):
        name = job_name(exp, label)
        (scratch / name).mkdir(parents=True, exist_ok=True)
        sbatch_args = _sbatch_args(exp, label, task_ids, args.account)
        result = subprocess.run(
            ["sbatch", *sbatch_args], capture_output=True, text=True, check=True
        )
        job_id = result.stdout.split()[-1]
        label_note = f", label={label!r}" if label else ""
        print(
            f"Submitted {name} array job {job_id} "
            f"({len(task_ids)} tasks, 1 full L40S x 3h each{label_note})"
        )
        print(f"Monitor: squeue -u $USER -j {job_id}")
        print(f"Logs:    /scratch/$USER/logs/mue/{name}/{job_id}_*.out")
    return 0


def main() -> int:
    names = sorted(EXPERIMENTS)
    parser = argparse.ArgumentParser(description="Vulcan SLURM experiment grids.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list registered experiments").set_defaults(
        func=lambda args: _cmd_list(EXPERIMENTS)
    )

    submit = sub.add_parser("submit", help="submit pending tasks to SLURM")
    submit.add_argument("experiment", choices=names)
    submit.add_argument("--dry-run", action="store_true")
    submit.add_argument("--include-active", action="store_true")
    submit.add_argument(
        "--account",
        default=os.environ.get("SLURM_ACCOUNT"),
        help="SLURM account (default: $SLURM_ACCOUNT)",
    )
    submit.set_defaults(func=lambda args: _cmd_submit(args, EXPERIMENTS))

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
