#!/usr/bin/env python3
"""Grid task mapping and completion checks for classic-control SLURM jobs."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import gymnax

JOB_NAME = "mue-classic"

ENVS = [
    "Pendulum-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "CartPole-v1",
    "Acrobot-v1",
]
SEEDS = list(range(10))
PREDICT_MODES = ["mean", "sample"]

# Task ids 0-199 keep the original exploit/explore grid layout.
LEGACY_REWARD_WEIGHTS = [(1.0, 0.0), (0.0, 1.0)]
LEGACY_COMBO_PER_ENV = len(LEGACY_REWARD_WEIGHTS) * len(PREDICT_MODES)
LEGACY_COMBO_PER_SEED = len(ENVS) * LEGACY_COMBO_PER_ENV
LEGACY_NUM_TASKS = len(SEEDS) * LEGACY_COMBO_PER_SEED

# Task ids 200-299 append a1b1 without shifting the legacy mapping.
BOTH_COMBO_PER_SEED = len(ENVS) * len(PREDICT_MODES)
BOTH_NUM_TASKS = len(SEEDS) * BOTH_COMBO_PER_SEED

NUM_TASKS = LEGACY_NUM_TASKS + BOTH_NUM_TASKS


def decode_task(task_id: int) -> dict[str, str | int | float]:
    if task_id < LEGACY_NUM_TASKS:
        seed = task_id // LEGACY_COMBO_PER_SEED
        rem = task_id % LEGACY_COMBO_PER_SEED
        env_idx = rem // LEGACY_COMBO_PER_ENV
        rem2 = rem % LEGACY_COMBO_PER_ENV
        reward_idx = rem2 // len(PREDICT_MODES)
        predict_idx = rem2 % len(PREDICT_MODES)
        alpha, beta = LEGACY_REWARD_WEIGHTS[reward_idx]
    else:
        sub = task_id - LEGACY_NUM_TASKS
        seed = sub // BOTH_COMBO_PER_SEED
        rem = sub % BOTH_COMBO_PER_SEED
        env_idx = rem // len(PREDICT_MODES)
        predict_idx = rem % len(PREDICT_MODES)
        alpha, beta = 1.0, 1.0
    return {
        "task_id": task_id,
        "seed": SEEDS[seed],
        "env": ENVS[env_idx],
        "alpha": alpha,
        "beta": beta,
        "mode": PREDICT_MODES[predict_idx],
    }


def log_dir(root: Path, task: dict[str, str | int | float]) -> Path:
    alpha_tag = str(task["alpha"]).replace(".", "p")
    beta_tag = str(task["beta"]).replace(".", "p")
    return (
        root
        / "runs"
        / "classic_grid"
        / str(task["env"])
        / f"a{alpha_tag}_b{beta_tag}"
        / str(task["mode"])
        / f"seed{task['seed']}"
    )


def expected_rollouts(env_name: str) -> int:
    _, env_params = gymnax.make(env_name)
    total_timesteps = env_params.max_steps_in_episode * 10
    num_steps = env_params.max_steps_in_episode // 10
    return total_timesteps // num_steps // 1


def is_complete(run_dir: Path, env_name: str) -> bool:
    if not run_dir.is_dir():
        return False
    if not any(run_dir.glob("events.out.tfevents*")):
        return False
    last_iter = expected_rollouts(env_name) - 1
    return (run_dir / f"uncertainty_{last_iter:04d}.png").is_file()


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


def active_task_ids(
    *,
    user: str | None = None,
    job_name: str = JOB_NAME,
) -> set[int]:
    """Array task ids currently queued or running in SLURM."""
    # %i = jobid_taskid for array jobs. (%a is account, not array index.)
    cmd = [
        "squeue",
        "-u",
        user or os.environ.get("USER", ""),
        "-n",
        job_name,
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
    repo_root: Path,
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
        if is_complete(log_dir(repo_root, task), str(task["env"])):
            complete += 1
            continue
        if task_id in active:
            in_progress += 1
            continue
        to_submit.append(task_id)

    return to_submit, complete, in_progress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repo_root",
        type=Path,
        help="Repository root (runs/classic_grid lives here)",
    )
    parser.add_argument(
        "--format",
        choices=("array", "summary"),
        default="array",
        help="array: comma-separated task ids; summary: human-readable counts",
    )
    parser.add_argument(
        "--include-active",
        action="store_true",
        help="submit tasks even if they are already queued/running in SLURM",
    )
    args = parser.parse_args()

    to_submit, complete, in_progress = tasks_to_submit(
        args.repo_root,
        skip_active=not args.include_active,
    )

    if args.format == "array":
        print(",".join(str(task_id) for task_id in to_submit))
        return

    print(f"Complete: {complete}/{NUM_TASKS}")
    print(f"In progress (skipped): {in_progress}/{NUM_TASKS}")
    print(f"To submit: {len(to_submit)}/{NUM_TASKS}")
    if to_submit:
        preview = to_submit[:8]
        suffix = "..." if len(to_submit) > len(preview) else ""
        print(f"Would submit task ids: {preview}{suffix}")


if __name__ == "__main__":
    main()
