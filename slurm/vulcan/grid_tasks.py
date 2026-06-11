#!/usr/bin/env python3
"""Grid task mapping and completion checks for classic-control SLURM jobs."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnax

ENVS = [
    "Pendulum-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "CartPole-v1",
    "Acrobot-v1",
]
SEEDS = list(range(10))
EXPLORE_ALPHAS = [1.0, 0.0]
EXPLORE_BETAS = [0.0, 1.0]
PREDICT_MODES = ["mean", "sample"]

COMBO_PER_SEED = 20
COMBO_PER_ENV = 4
NUM_TASKS = len(SEEDS) * len(ENVS) * len(EXPLORE_ALPHAS) * len(PREDICT_MODES)


def decode_task(task_id: int) -> dict[str, str | int | float]:
    seed = task_id // COMBO_PER_SEED
    rem = task_id % COMBO_PER_SEED
    env_idx = rem // COMBO_PER_ENV
    rem2 = rem % COMBO_PER_ENV
    explore_idx = rem2 // 2
    predict_idx = rem2 % 2
    return {
        "task_id": task_id,
        "seed": SEEDS[seed],
        "env": ENVS[env_idx],
        "alpha": EXPLORE_ALPHAS[explore_idx],
        "beta": EXPLORE_BETAS[explore_idx],
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


def incomplete_task_ids(repo_root: Path) -> list[int]:
    incomplete = []
    for task_id in range(NUM_TASKS):
        task = decode_task(task_id)
        if not is_complete(log_dir(repo_root, task), str(task["env"])):
            incomplete.append(task_id)
    return incomplete


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
    args = parser.parse_args()

    incomplete = incomplete_task_ids(args.repo_root)
    complete = NUM_TASKS - len(incomplete)

    if args.format == "array":
        print(",".join(str(task_id) for task_id in incomplete))
        return

    print(f"Complete: {complete}/{NUM_TASKS}")
    print(f"Incomplete: {len(incomplete)}/{NUM_TASKS}")
    if incomplete:
        preview = incomplete[:8]
        suffix = "..." if len(incomplete) > len(preview) else ""
        print(f"Would submit task ids: {preview}{suffix}")


if __name__ == "__main__":
    main()
