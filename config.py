import dataclasses
from dataclasses import dataclass, field
from itertools import product
from typing import Annotated, ClassVar, Literal

import tyro


@dataclass(frozen=True)
class EpinetConfig:
    learnable_hidden_dim: int | tuple[int, ...] = 15
    prior_hidden_dim: int | tuple[int, ...] = 5
    index_dim: int | tuple[int, ...] = 8
    num_layers: int | tuple[int, ...] = 2
    activation: str = "silu"


@dataclass(frozen=True)
class ENNConfig:
    """Epistemic Neural Network"""

    name: ClassVar[str] = "enn"
    lr: float | tuple[float, ...] = 1e-3
    hidden_dim: int | tuple[int, ...] = 200
    activation: str = "silu"
    num_layers: int | tuple[int, ...] = 4
    l2: float | tuple[float, ...] = 0.0
    weight_decay: float | None = 1e-2
    label_smoothing: float = 0.01
    update_steps: int | tuple[int, ...] = 10000
    use_layer_norm: bool = False
    head: Literal["deterministic", "gaussian", "categorical"] = "deterministic"
    prior_scale: float = 1.0
    min_scale: float = 1e-3
    beta_nll: float = 0.5
    bootstrap: Literal["none", "bernoulli", "gaussian"] = "none"
    bootstrap_noise: float = 1.0
    epinet: EpinetConfig = field(default_factory=EpinetConfig)


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
    num_layers: int = 2
    activation: str = "tanh"
    use_layer_norm: bool = False
    anneal_lr: bool = False
    normalize_obs: bool = True
    normalize_reward: bool = True
    clip_obs: float | None = 10.0
    clip_reward: float | None = 10.0
    debug: bool = False


@dataclass(frozen=True)
class RNDConfig:
    """Random Network Distillation bonus (--explore-bonus rnd)."""

    lr: float = 1e-3
    hidden_dim: int = 64
    out_dim: int = 32
    batch_size: int = 256
    update_steps: int = 100



@dataclass(frozen=True)
class OpaxConfig:
    """Opax agent (--opax) and its iCEM planner hyperparameters"""
    #iCEM configuration
    num_samples: int = 500
    num_elites: int = 50
    init_std: float = 0.5
    alpha: float = 0.0
    elite_set_fraction: float = 0.3
    horizon: int = 25 #depends on env: 20 for CC, 30 for mujoco
    train_steps_per_model_update: int = 35
    transitions_per_update: int = 500
    normalize: bool = True
    #model configuration
    num_ensembles: int = 5
    num_layers: int = 2
    hidden_dim: int = 256
    reset_model: bool = True
    pred_diff: bool = True
    beta: float = 2.0
    deterministic_env: bool = True
    n_particles: int = 5
    num_steps: int = 20
    eval_envs: int = 10


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
    solver: Literal["ppo"] = "ppo"
    agent: Literal["opax", "model-based", "model-free"] = "model-based"
    model_env_mode: Literal["mean", "sample"] = "sample"
    explore_bonus: Literal["std", "eig", "rnd"] = "eig"
    predict_reward_terminated: bool = False
    reset_source: Literal["env", "buffer", "init"] = "env"
    rollout_length: int | None = None
    steps_per_rollout: int | None = None
    eval_every: int | None = None
    checkpoint_every: int | None = None
    uncertainty_threshold: float | None = None
    max_steps: int | None = None
    offline: bool = False
    dataset: str = "plant-data/visu-v27"
    debug: bool = False
    log_dir: str | None = None
    label: str = ""
    ppo: PPOConfig = field(default_factory=PPOConfig)
    rnd: RNDConfig = field(default_factory=RNDConfig)
    model: ModelConfig = field(default_factory=ENNConfig)
    opax: OpaxConfig = field(default_factory=OpaxConfig)

    def __post_init__(self):
        if (self.agent != "model-based") and self.offline:
            raise ValueError(f"--agent {self.agent} needs an online env")
        if (self.agent != "model-based") and self.explore_bonus == "rnd":
            raise ValueError(
                "--explore-bonus rnd is a model-env bonus (--agent model-based only)"
            )
        if (self.agent == "opax") and self.predict_reward_terminated:
            raise ValueError(
                "--predict-reward-terminated cannot be used with OPAX "
                "(it plans with the env's reward oracle)"
            )
        if getattr(self.model, "head", None) == "categorical":
            if self.offline:
                raise ValueError(
                    "--model.head categorical needs an online Discrete env"
                )
            if self.explore_bonus == "rnd":
                raise ValueError("--explore-bonus rnd does not support Discrete obs")


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
    max_data: int,
    minibatch_size: int,
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
    discrete_obs: bool = False,
) -> dict:
    """Convert PPO config to the UPPERCASE dict ppo.py expects.

    Computes derived keys NUM_UPDATES and MINIBATCH_SIZE from the tunable base
    keys so that overriding e.g. --ppo.num-envs automatically re-derives them.
    Observation normalization is disabled for Discrete (index) observations.
    """
    d: dict = {k.upper(): v for k, v in dataclasses.asdict(p).items()}
    if not offline:
        d["ENV_NAME"] = env_name
    d["SEED"] = seed
    d["NUM_UPDATES"] = int(d["TOTAL_TIMESTEPS"]) // d["NUM_STEPS"] // d["NUM_ENVS"]
    d["MINIBATCH_SIZE"] = d["NUM_ENVS"] * d["NUM_STEPS"] // d["NUM_MINIBATCHES"]
    d["NORMALIZE_OBS"] = d["NORMALIZE_OBS"] and not discrete_obs
    return d
