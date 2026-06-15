#!/usr/bin/env python3
"""Classic-control SLURM grid: the single source of truth for task logic.

This one file owns the whole pipeline:
  * the grid definition (env x reward-weight x predict-mode),
  * decode_task: array id -> run config (used by classic_control.sbatch),
  * completion / queue checks and submission (replaces the old submit.sh).

Seeds are vmapped *within* a task (--num_seeds), not spread across array
elements, so the array axis is just env x reward-weight x predict-mode.

Usage:
  submit.py submit [--dry-run] [--include-active] [--account ACCT]
  submit.py env TASK_ID      # emit shell assignments for one task (sbatch)
  submit.py plan [--format array|summary] [--include-active]
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

JOB_NAME = "mue-classic"
DEFAULT_ACCOUNT = "aip-amw8"

# submit.py lives at <repo>/slurm/vulcan/submit.py.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SBATCH_SCRIPT = SCRIPT_DIR / "classic_control.sbatch"

# --- grid definition (the one place task logic is described) ---------------
BASE_SEED = 0
NUM_SEEDS = 30  # vmapped within each task, not array elements

ENVS = [
    "Pendulum-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "CartPole-v1",
    "Acrobot-v1",
]
# (alpha, beta): exploit-only, explore-only, both.
REWARD_WEIGHTS = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
PREDICT_MODES = ["mean", "sample"]

COMBO_PER_ENV = len(REWARD_WEIGHTS) * len(PREDICT_MODES)
NUM_TASKS = len(ENVS) * COMBO_PER_ENV


def decode_task(task_id: int) -> dict[str, str | int | float]:
    """Map an array task id -> the run config for that task."""
    if not 0 <= task_id < NUM_TASKS:
        raise ValueError(f"task_id {task_id} out of range [0, {NUM_TASKS})")
    env_idx, rem = divmod(task_id, COMBO_PER_ENV)
    reward_idx, predict_idx = divmod(rem, len(PREDICT_MODES))
    alpha, beta = REWARD_WEIGHTS[reward_idx]
    return {
        "task_id": task_id,
        "env": ENVS[env_idx],
        "alpha": alpha,
        "beta": beta,
        "mode": PREDICT_MODES[predict_idx],
        "base_seed": BASE_SEED,
        "num_seeds": NUM_SEEDS,
    }


def log_dir(task: dict[str, str | int | float]) -> Path:
    """Parent dir for a task; main.py writes per-seed subdirs underneath."""
    alpha_tag = str(task["alpha"]).replace(".", "p")
    beta_tag = str(task["beta"]).replace(".", "p")
    return (
        REPO_ROOT
        / "runs"
        / "classic_grid"
        / str(task["env"])
        / f"a{alpha_tag}_b{beta_tag}"
        / str(task["mode"])
    )


def is_complete(task: dict[str, str | int | float]) -> bool:
    """A task is done once main.py has written its COMPLETE sentinel."""
    return (log_dir(task) / "COMPLETE").is_file()


def _parse_squeue_array_line(line: str) -> int | None:
    """Parse array task id from squeue %i output (e.g. '5216383_184')."""
    line = line.strip()
    if not line or line == "N/A":
        return None
    for sep in ("_", "."):
        if sep in line:
            suffix = line.rsplit(sep, 1)[-1]
            if suffix.isdigit():
                return int(suffix)
    return None


def active_task_ids(*, user: str | None = None) -> set[int]:
    """Array task ids currently queued or running in SLURM."""
    # %i = jobid_taskid for array jobs. (%a is account, not array index.)
    cmd = [
        "squeue",
        "-u",
        user or os.environ.get("USER", ""),
        "-n",
        JOB_NAME,
        "-h",
        "-o",
        "%i",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return set()

    active: set[int] = set()
    for line in result.stdout.splitlines():
        task_id = _parse_squeue_array_line(line)
        if task_id is not None:
            active.add(task_id)
    return active


def tasks_to_submit(
    *,
    skip_active: bool = True,
    user: str | None = None,
) -> tuple[list[int], int, int]:
    active = active_task_ids(user=user) if skip_active else set()
    complete = 0
    in_progress = 0
    to_submit: list[int] = []

    for task_id in range(NUM_TASKS):
        task = decode_task(task_id)
        if is_complete(task):
            complete += 1
            continue
        if task_id in active:
            in_progress += 1
            continue
        to_submit.append(task_id)

    return to_submit, complete, in_progress


def _array_spec(task_ids: list[int]) -> str:
    return ",".join(str(t) for t in task_ids)


def _grid_line() -> str:
    return (
        f"{NUM_TASKS} tasks = {len(ENVS)} envs x {len(REWARD_WEIGHTS)} reward "
        f"x {len(PREDICT_MODES)} predict ({NUM_SEEDS} seeds vmapped per task)"
    )


# --- subcommands -----------------------------------------------------------
def _cmd_env(args: argparse.Namespace) -> int:
    """Emit shell assignments for one task (eval'd by classic_control.sbatch)."""
    task = decode_task(args.task_id)
    fields = {
        "ENV": task["env"],
        "ALPHA": task["alpha"],
        "BETA": task["beta"],
        "MODE": task["mode"],
        "BASE_SEED": task["base_seed"],
        "NUM_SEEDS": task["num_seeds"],
        "LOG_DIR": str(log_dir(task)),
    }
    for key, value in fields.items():
        print(f"{key}={shlex.quote(str(value))}")
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    to_submit, complete, in_progress = tasks_to_submit(
        skip_active=not args.include_active
    )
    if args.format == "array":
        print(_array_spec(to_submit))
        return 0
    print(f"Grid: {_grid_line()}")
    print(f"Complete: {complete}/{NUM_TASKS}")
    print(f"In progress (skipped): {in_progress}/{NUM_TASKS}")
    print(f"To submit: {len(to_submit)}/{NUM_TASKS}")
    if to_submit:
        preview = to_submit[:8]
        suffix = "..." if len(to_submit) > len(preview) else ""
        print(f"Would submit task ids: {preview}{suffix}")
    return 0


def _cmd_submit(args: argparse.Namespace) -> int:
    if not (REPO_ROOT / ".venv").is_dir():
        print(
            f"No .venv in {REPO_ROOT} — run {SCRIPT_DIR / 'setup.sh'}", file=sys.stderr
        )
        return 1

    to_submit, complete, in_progress = tasks_to_submit(
        skip_active=not args.include_active
    )
    sbatch_args = [
        f"--account={args.account}",
        f"--export=ALL,REPO_ROOT={REPO_ROOT}",
        f"--array={_array_spec(to_submit)}",
        str(SBATCH_SCRIPT),
    ]

    if args.dry_run:
        print(f"Grid: {_grid_line()}")
        print(
            f"Complete: {complete} | In progress: {in_progress} | "
            f"To submit: {len(to_submit)}"
        )
        print(f"Would run: sbatch {' '.join(sbatch_args)}")
        return 0

    if not to_submit:
        print("Nothing to submit (all tasks complete or already queued/running).")
        return 0

    (Path("/scratch") / os.environ.get("USER", "") / "logs").mkdir(
        parents=True, exist_ok=True
    )
    result = subprocess.run(
        ["sbatch", *sbatch_args], capture_output=True, text=True, check=True
    )
    job_id = result.stdout.split()[-1]
    print(
        f"Submitted array job {job_id} ({len(to_submit)} tasks, "
        "1 full L40S x 3h each)"
    )
    print(f"Monitor: squeue -u $USER -j {job_id}")
    print(f"Logs:    /scratch/$USER/logs/{JOB_NAME}-{job_id}_*.out")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_submit = sub.add_parser("submit", help="submit pending tasks to SLURM")
    p_submit.add_argument("--dry-run", action="store_true")
    p_submit.add_argument(
        "--include-active",
        action="store_true",
        help="resubmit tasks already queued/running in SLURM",
    )
    p_submit.add_argument(
        "--account",
        default=os.environ.get("SLURM_ACCOUNT", DEFAULT_ACCOUNT),
    )
    p_submit.set_defaults(func=_cmd_submit)

    p_env = sub.add_parser("env", help="emit shell assignments for one task")
    p_env.add_argument("task_id", type=int)
    p_env.set_defaults(func=_cmd_env)

    p_plan = sub.add_parser("plan", help="list/summarize tasks needing submission")
    p_plan.add_argument("--format", choices=("array", "summary"), default="summary")
    p_plan.add_argument("--include-active", action="store_true")
    p_plan.set_defaults(func=_cmd_plan)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
