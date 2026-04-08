import gymnasium as gym

from mue.config import ExperimentConfig
from mue.models.gp import GPModel


def build_model(config: ExperimentConfig, env: gym.Env):
    obs_space = env.observation_space
    act_space = env.action_space

    assert isinstance(obs_space, gym.spaces.Box), (
        f"Expected Box obs space, got {type(obs_space)}"
    )
    obs_dim = obs_space.shape[0]

    if isinstance(act_space, gym.spaces.Discrete):
        act_dim = 1
    elif isinstance(act_space, gym.spaces.Box):
        act_dim = act_space.shape[0]
    else:
        raise ValueError(f"Unsupported action space: {type(act_space)}")

    match config.model_type:
        case "gp":
            return GPModel(obs_dim=obs_dim, act_dim=act_dim, seed=config.seed)
        case _:
            raise ValueError(f"Unknown model type: {config.model_type}")
