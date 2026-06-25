import json
import shlex
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Sequence

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
    model: str = "enn"
    label: str = ""
    overrides: tuple[tuple[str, str | int | float | bool], ...] = ()


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

    def task_config(self, task_id: int) -> dict[str, str | int | float | bool | dict]:
        cfg = self.configs[task_id]
        base = asdict(cfg)
        # Flatten overrides from list-of-pairs to a readable dict in the JSON output.
        base["overrides"] = {k: v for k, v in cfg.overrides}
        return {
            "experiment": self.name,
            "task_id": task_id,
            "base_seed": self.base_seed,
            "num_seeds": self.num_seeds,
            **base,
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


def job_name(exp: Experiment, label: str = "") -> str:
    """SLURM job name for a label group (empty label → experiment name only)."""
    return f"{exp.name}-{label}" if label else exp.name


def job_names(exp: Experiment) -> set[str]:
    return {job_name(exp, cfg.label) for cfg in exp.configs}


def active_task_ids_for_experiment(
    exp: Experiment, *, user: str | None = None
) -> set[int]:
    active: set[int] = set()
    for name in job_names(exp):
        active |= active_task_ids(name, user=user)
    return active


def tasks_to_submit(
    exp: Experiment,
    *,
    skip_active: bool = True,
    user: str | None = None,
) -> tuple[list[int], int, int]:
    active = active_task_ids_for_experiment(exp, user=user) if skip_active else set()
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


def _flag_values(value: str | int | float | bool | tuple) -> list[str]:
    """Serialize a scalar or tuple override value to a list of CLI tokens.

    A tuple becomes multiple space-separated tokens so tyro reads it as a
    variadic tuple field, e.g. (0.5, 1.0, 3.0) → ["0.5", "1.0", "3.0"].
    """
    if isinstance(value, tuple):
        return [str(v) for v in value]
    return [str(value)]


def _flag(dotted_key: str) -> str:
    """Convert a dotted override key to a tyro CLI flag.

    Examples:
        "model.length_scale" -> "--model.length-scale"
        "ppo.ent_coef"       -> "--ppo.ent-coef"
    """
    section, field = dotted_key.split(".", 1)
    return f"--{section}.{field.lower().replace('_', '-')}"


def _main_argv(exp: Experiment, task_id: int) -> list[str]:
    cfg = exp.configs[task_id]

    argv = [
        "main.py",
        "--env",
        cfg.env,
        "--seed",
        str(exp.base_seed),
        "--num-seeds",
        str(exp.num_seeds),
        "--alpha",
        str(cfg.alpha),
        "--beta",
        str(cfg.beta),
        "--model-env-mode",
        cfg.mode,
        "--explore-bonus",
        cfg.bonus,
        "--log-dir",
        str(exp.log_dir(task_id)),
    ]
    if cfg.predict_reward_terminated:
        argv.append("--predict-reward-terminated")

    for key, value in cfg.overrides:
        if key.startswith("ppo."):
            argv += [_flag(key)] + _flag_values(value)

    argv.append(f"model:{cfg.model}")
    for key, value in cfg.overrides:
        if key.startswith("model."):
            argv += [_flag(key)] + _flag_values(value)

    return argv


def sweep(
    *,
    env: str | Sequence[str],
    alpha: float | Sequence[float] = 0.0,
    beta: float | Sequence[float] = 1.0,
    mode: str | Sequence[str] = "sample",
    bonus: str | Sequence[str] = "std",
    predict_reward_terminated: bool | Sequence[bool] = False,
    model: str | Sequence[str] = "enn",
    label: str | Sequence[str] = "",
    **override_axes: float | int | str | bool | Sequence[float | int | str | bool],
) -> tuple[RunConfig, ...]:
    """Build a Cartesian-product grid of RunConfigs.

    Any argument can be a scalar (pinned) or a sequence (swept axis).
    Hyperparameter overrides use double-underscore notation for the dotted
    field name, e.g. ``model__length_scale=(0.5, 1.0, 3.0)`` or
    ``ppo__ent_coef=(0.0, 0.01)``.
    """

    def _axis(v: object) -> tuple:
        return tuple(v) if isinstance(v, (list, tuple)) else (v,)  # type: ignore[arg-type]

    base_axes: dict[str, tuple] = {
        "env": _axis(env),
        "alpha": _axis(alpha),
        "beta": _axis(beta),
        "mode": _axis(mode),
        "bonus": _axis(bonus),
        "predict_reward_terminated": _axis(predict_reward_terminated),
        "model": _axis(model),
        "label": _axis(label),
    }

    ov_keys: list[str] = [k.replace("__", ".", 1) for k in override_axes]
    ov_axes: list[tuple] = [_axis(v) for v in override_axes.values()]

    configs: list[RunConfig] = []
    n_base = len(base_axes)
    for combo in product(*base_axes.values(), *ov_axes):
        base_vals = dict(zip(base_axes.keys(), combo[:n_base]))
        ov_pairs = tuple(zip(ov_keys, combo[n_base:]))
        configs.append(RunConfig(**base_vals, overrides=ov_pairs))  # type: ignore[arg-type]

    return tuple(configs)


def prepare_task(exp: Experiment, task_id: int) -> None:
    """Write task config.json and print MAIN_ARGS for run.sbatch eval."""
    if not 0 <= task_id < exp.num_tasks:
        raise ValueError(f"task_id {task_id} out of range [0, {exp.num_tasks})")
    exp.write_task_config(task_id)
    print(f"MAIN_ARGS={shlex.quote(shlex.join(_main_argv(exp, task_id)))}")
