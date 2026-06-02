from dataclasses import dataclass

from mue.agents.base import BaseAgentConfig


@dataclass
class PPOConfig(BaseAgentConfig):
    lr: float = 3e-4
    num_envs: int = 2048
    num_steps: int = 10
    total_timesteps: int = int(5e7)
    update_epochs: int = 4
    num_minibatches: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    activation: str = "tanh"
    layer_norm: bool = True
    anneal_lr: bool = False
    normalize_obs: bool = True
    normalize_reward: bool = True
