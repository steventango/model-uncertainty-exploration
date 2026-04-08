from dataclasses import asdict, dataclass, fields


@dataclass
class ExperimentConfig:
    # Environment
    env_id: str = "Pendulum-v1"
    seed: int = 42
    max_steps: int = 10_000
    eval_episodes: int = 5

    # Model
    model_type: str = "gp"  # "gp" | "lcm_gp" | "sparse_gp" | "blr" | "ensemble" | "bnn" | "mc_dropout" | "snn"

    # GP / LCM GP / Sparse GP
    gp_latent_dim: int = 4
    gp_num_inducing: int = 500
    gp_learning_rate: float = 0.01
    gp_num_iters: int = 500
    gp_batch_size: int = 1024

    # BLR
    blr_num_features: int = 256
    blr_prior_variance: float = 1.0
    blr_noise_variance: float = 0.1

    # Ensemble
    ensemble_size: int = 5
    ensemble_hidden_dims: tuple[int, ...] = (64, 64)
    ensemble_learning_rate: float = 1e-3
    ensemble_train_epochs: int = 50
    ensemble_batch_size: int = 64
    ensemble_bootstrap: bool = True

    # BNN (variational inference / Bayes by Backprop)
    bnn_hidden_dims: tuple[int, ...] = (64, 64)
    bnn_learning_rate: float = 1e-3
    bnn_train_epochs: int = 50
    bnn_batch_size: int = 64
    bnn_num_mc_samples: int = 20
    bnn_kl_weight: float = 1.0

    # MC Dropout
    mc_dropout_hidden_dims: tuple[int, ...] = (256, 256)
    mc_dropout_learning_rate: float = 1e-3
    mc_dropout_train_epochs: int = 100
    mc_dropout_rate: float = 0.1
    mc_dropout_num_mc_samples: int = 20

    # Spectral Normalized Network
    snn_hidden_dims: tuple[int, ...] = (256, 256)
    snn_learning_rate: float = 1e-3
    snn_train_epochs: int = 100
    snn_spectral_norm_coeff: float = 0.95

    # Agent
    agent_type: str = "random"  # "random" | "q_iteration" | "ppo" | "sac"
    q_function_type: str = "linear"  # "tabular" | "linear" | "nn"

    # Intrinsic reward
    intrinsic_reward: str = "trace"  # "trace" | "det" | "epistemic_trace" | "epistemic_det" | "aleatoric_trace" | "epistemic_ratio"
    alpha: float = 1.0  # extrinsic reward weight
    beta: float = 0.1  # intrinsic reward weight

    # Planning
    planning_schedule: str = (
        "every_k"  # "every_step" | "every_k" | "end_of_episode" | "marginal_ig"
    )
    planning_k: int = 10
    marginal_ig_threshold: float = 0.01

    # Buffer
    buffer_size: int = 100_000
    batch_size: int = 64

    # Agent updates per planning step
    agent_update_steps: int = 1

    # Q-iteration
    gamma: float = 0.99
    q_learning_rate: float = 0.01
    q_iterations: int = 50

    # Strategy
    strategy_type: str = "agent"  # "agent" | "greedy_acquisition" | "mpc" | "rnd" | "ids" | "ids_mpc"

    # Action optimizer
    optimizer_type: str = "grid"  # "grid" | "cem" | "random_shooting"

    # Acquisition function
    acquisition_fn: str = "trace"  # "trace" | "det" | "ucb" | "joint_eig" | "disagreement" | "epistemic_trace" | "epistemic_det" | "aleatoric_trace" | "epistemic_ratio"
    ucb_beta: float = 2.0

    # CEM parameters (shared by optimizer and sequence optimizer)
    cem_samples: int = 1000
    cem_elite_frac: float = 0.05
    cem_iterations: int = 10

    # MPC
    mpc_horizon: int = 10

    # RND
    rnd_feature_dim: int = 128

    # TD-MPC2
    tdmpc2_latent_dim: int = 50
    tdmpc2_hidden_dim: int = 256
    tdmpc2_horizon: int = 5
    tdmpc2_n_ensemble: int = 5
    tdmpc2_cem_samples: int = 512
    tdmpc2_cem_elite_frac: float = 0.05
    tdmpc2_cem_iterations: int = 6
    tdmpc2_cem_temperature: float = 0.5
    tdmpc2_cem_momentum: float = 0.1
    tdmpc2_learning_rate: float = 3e-4
    tdmpc2_tau: float = 0.01
    tdmpc2_consistency_coef: float = 2.0
    tdmpc2_reward_coef: float = 0.5
    tdmpc2_value_coef: float = 0.1
    tdmpc2_exploration_noise: float = 0.2

    # PPO
    ppo_hidden_dim: int = 64
    ppo_learning_rate: float = 3e-4
    ppo_clip_eps: float = 0.2
    ppo_entropy_coef: float = 0.0
    ppo_vf_coef: float = 0.5
    ppo_gae_lambda: float = 0.95
    ppo_n_epochs: int = 10
    ppo_n_minibatches: int = 32
    ppo_max_grad_norm: float = 0.5
    ppo_clip_vloss: bool = True

    # SAC
    sac_hidden_dim: int = 256
    sac_learning_rate: float = 3e-4
    sac_tau: float = 0.005
    sac_init_alpha: float = 0.2

    # Visualization
    visualize: bool = False

    # Logging
    log_dir: str = "runs"

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)


def apply_overrides(config: ExperimentConfig, overrides: list[str]) -> ExperimentConfig:
    """Apply ``--key=value`` CLI overrides to a config, coercing types."""
    d = asdict(config)
    defaults = {f.name: f.default for f in fields(ExperimentConfig)}
    for token in overrides:
        key, value = token.lstrip("-").split("=", 1)
        if key not in defaults:
            raise ValueError(f"Unknown config key: {key}")
        ref = defaults[key]
        if isinstance(ref, bool):
            d[key] = value.lower() in ("true", "1", "yes")
        elif isinstance(ref, int):
            d[key] = int(value)
        elif isinstance(ref, float):
            d[key] = float(value)
        else:
            d[key] = value
    return ExperimentConfig(**d)
