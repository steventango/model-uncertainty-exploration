import dataclasses
from dataclasses import dataclass, field
from itertools import product
from typing import Annotated, ClassVar, Literal

import tyro


@dataclass(frozen=True)
class ENNConfig:
    """Epistemic Neural Network"""

    name: ClassVar[str] = "enn"
    lr: float | tuple[float, ...] = 1e-3
    hidden_dim: int | tuple[int, ...] = 64
    learnable_hidden_dim: int | tuple[int, ...] = 15
    prior_hidden_dim: int | tuple[int, ...] = 5
    index_dim: int | tuple[int, ...] = 8
    activation: str = "tanh"
    update_steps: int | tuple[int, ...] = 10000


@dataclass(frozen=True)
class BLRConfig:
    """Bayesian Linear Regression with RBF/RFF features."""

    name: ClassVar[str] = "blr"
    feature_type: Literal["rbf", "rff"] = "rbf"
    num_features: int | tuple[int, ...] = 256
    length_scale: float | tuple[float, ...] = 1.0
    num_samples: int | tuple[int, ...] = 10
    lam: float | tuple[float, ...] = 0.01
    a0: float | tuple[float, ...] = 1.0
    b0: float | tuple[float, ...] = 1.0
    update_steps: int | tuple[int, ...] = 1


@dataclass(frozen=True)
class PPOConfig:
    """PPO hyperparameters."""

    lr: float = 3e-4
    num_envs: int = 2048
    num_steps: int = 10
    total_timesteps: float = 1e7
    update_epochs: int = 4
    num_minibatches: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_dim: int = 64
    activation: str = "tanh"
    use_layer_norm: bool = False
    anneal_lr: bool = False
    normalize_env: bool = True
    debug: bool = False


ModelConfig = (
    Annotated[ENNConfig, tyro.conf.subcommand("enn", default=ENNConfig())]
    | Annotated[BLRConfig, tyro.conf.subcommand("blr")]
)


@dataclass(frozen=True)
class Args:
    env: str = "Pendulum-v1"
    seed: int = 0
    num_seeds: int = 1
    alpha: float = 0.0
    beta: float = 1.0
    model_env_mode: Literal["mean", "sample"] = "sample"
    explore_bonus: Literal["std", "eig"] = "eig"
    predict_reward_terminated: bool = False
    reset_source: Literal["env", "buffer", "init"] = "env"
    rollout_length: int | None = None
    steps_per_rollout: int | None = None
    uncertainty_threshold: float | None = None
    num_rollouts: int | None = None
    offline: bool = False
    dataset: str = "plant-data/visu-v27"
    debug: bool = False
    log_dir: str | None = None
    label: str = ""
    ppo: PPOConfig = field(default_factory=PPOConfig)
    model: ModelConfig = field(default_factory=ENNConfig)


def candidate_configs(m):
    """Expand tuple fields into all scalar candidate configs (Cartesian product).

    Returns a tuple of configs where every field is a single committed value.
    If no field is a tuple, returns a 1-tuple containing the original config unchanged.

    Example:
        candidate_configs(BLRConfig(length_scale=(0.5, 1.0, 3.0), num_features=64))
        # → (BLRConfig(length_scale=0.5, num_features=64),
        #    BLRConfig(length_scale=1.0, num_features=64),
        #    BLRConfig(length_scale=3.0, num_features=64))
    """
    axes = {
        f.name: getattr(m, f.name)
        if isinstance(getattr(m, f.name), tuple)
        else (getattr(m, f.name),)
        for f in dataclasses.fields(m)
    }
    return tuple(
        dataclasses.replace(m, **dict(zip(axes.keys(), combo)))
        for combo in product(*axes.values())
    )


def model_config_dict(
    m: ENNConfig | BLRConfig,
    *,
    max_data: int | float,
    minibatch_size: int | float,
) -> dict:
    """Convert a committed (scalar) model config to the UPPERCASE dict the registry expects.

    Injects the runtime-derived key MINIBATCH_SIZE. For BLR with RBF features,
    NUM_FEATURES is overridden to max_data (the dataset buffer size) because the
    RBF bank places one center per stored datapoint.
    """
    d: dict = {k.upper(): v for k, v in dataclasses.asdict(m).items()}
    d["MINIBATCH_SIZE"] = minibatch_size
    if isinstance(m, BLRConfig) and m.feature_type == "rbf":
        d["NUM_FEATURES"] = max_data
    return d


def ppo_config_dict(
    p: PPOConfig,
    *,
    env_name: str,
    seed: int,
    offline: bool,
) -> dict:
    """Convert PPO config to the UPPERCASE dict ppo.py expects.

    Computes derived keys NUM_UPDATES and MINIBATCH_SIZE from the tunable base
    keys so that overriding e.g. --ppo.num-envs automatically re-derives them.
    """
    d: dict = {k.upper(): v for k, v in dataclasses.asdict(p).items()}
    if not offline:
        d["ENV_NAME"] = env_name
    d["SEED"] = seed
    d["NUM_UPDATES"] = int(d["TOTAL_TIMESTEPS"]) // d["NUM_STEPS"] // d["NUM_ENVS"]
    d["MINIBATCH_SIZE"] = d["NUM_ENVS"] * d["NUM_STEPS"] // d["NUM_MINIBATCHES"]
    return d
