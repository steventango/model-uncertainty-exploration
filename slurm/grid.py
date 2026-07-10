import json
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path

from slurm.queue import active_task_ids

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SBATCH_SCRIPT = SCRIPT_DIR / "run.sbatch"
CONFIG_NAME = "config.json"

CLASSIC_ENVS = (
    "Pendulum-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "CartPole-v1",
    "Acrobot-v1",
)


@dataclass(frozen=True)
class RunConfig:
    env: str
    alpha: float
    beta: float
    mode: str
    bonus: str = "std"
    predict_reward_terminated: bool = False


@dataclass(frozen=True)
class Experiment:
    name: str
    configs: tuple[RunConfig, ...]
    base_seed: int = 0
    num_seeds: int = 30
    description: str = ""

    @property
    def num_tasks(self) -> int:
        return len(self.configs)

    def task_dir_name(self, task_id: int) -> str:
        return f"task_{task_id:04d}"

    def log_dir(self, task_id: int) -> Path:
        return REPO_ROOT / "runs" / self.name / self.task_dir_name(task_id)

    def task_config(self, task_id: int) -> dict[str, str | int | float | bool]:
        cfg = self.configs[task_id]
        return {
            "experiment": self.name,
            "task_id": task_id,
            "base_seed": self.base_seed,
            "num_seeds": self.num_seeds,
            **asdict(cfg),
        }

    def write_task_config(self, task_id: int) -> Path:
        log_dir = self.log_dir(task_id)
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / CONFIG_NAME
        path.write_text(json.dumps(self.task_config(task_id), indent=2) + "\n")
        return path

    def is_complete(self, task_id: int) -> bool:
        return (self.log_dir(task_id) / "COMPLETE").is_file()

    def grid_line(self) -> str:
        envs = {cfg.env for cfg in self.configs}
        per_env = len(self.configs) // len(envs) if envs else 0
        shape = f"{len(envs)} envs x {per_env} configs"
        if self.description:
            return (
                f"{self.num_tasks} tasks = {shape} ({self.description}; "
                f"{self.num_seeds} seeds vmapped per task)"
            )
        return (
            f"{self.num_tasks} tasks = {shape} "
            f"({self.num_seeds} seeds vmapped per task)"
        )


def tasks_to_submit(
    exp: Experiment,
    *,
    skip_active: bool = True,
    user: str | None = None,
) -> tuple[list[int], int, int]:
    active = active_task_ids(exp.name, user=user) if skip_active else set()
    complete = 0
    in_progress = 0
    to_submit: list[int] = []

    for task_id in range(exp.num_tasks):
        if exp.is_complete(task_id):
            complete += 1
            continue
        if task_id in active:
            in_progress += 1
            continue
        to_submit.append(task_id)

    return to_submit, complete, in_progress


def _main_argv(exp: Experiment, task_id: int) -> list[str]:
    cfg = exp.configs[task_id]
    argv = [
        "main.py",
        "--env",
        cfg.env,
        "--seed",
        str(exp.base_seed),
        "--num_seeds",
        str(exp.num_seeds),
        "--alpha",
        str(cfg.alpha),
        "--beta",
        str(cfg.beta),
        "--model_env_mode",
        cfg.mode,
        "--explore_bonus",
        cfg.bonus,
        "--log_dir",
        str(exp.log_dir(task_id)),
    ]
    if cfg.predict_reward_terminated:
        argv.append("--predict_reward_terminated")
    return argv


def prepare_task(exp: Experiment, task_id: int) -> None:
    """Write task config.json and print MAIN_ARGS for run.sbatch eval."""
    if not 0 <= task_id < exp.num_tasks:
        raise ValueError(f"task_id {task_id} out of range [0, {exp.num_tasks})")
    exp.write_task_config(task_id)
    print(f"MAIN_ARGS={shlex.quote(shlex.join(_main_argv(exp, task_id)))}")
