from dataclasses import dataclass

import gymnasium as gym

from mue.data.buffer import ReplayBuffer


@dataclass
class BufferConfig:
    buffer_size: int = 100_000


def build_buffer(config: BufferConfig, env: gym.Env) -> ReplayBuffer:
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
    return ReplayBuffer(
        max_length=config.buffer_size,
        obs_dim=obs_dim,
        act_dim=act_dim,
    )
