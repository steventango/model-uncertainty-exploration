import gymnasium as gym

from mue.config import ExperimentConfig
from mue.models.gp import GPModel


def build_model(config: ExperimentConfig, env: gym.Env):
    assert env.observation_space.shape is not None
    assert env.action_space.shape is not None
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    match config.model_type:
        case "gp":
            return GPModel(obs_dim=obs_dim, act_dim=act_dim, seed=config.seed)
        case _:
            raise ValueError(f"Unknown model type: {config.model_type}")
