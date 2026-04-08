from dataclasses import dataclass

import flashbax as fbx
import gymnasium as gym
import jax
import jax.numpy as jnp
from flashbax.buffers.trajectory_buffer import TrajectoryBuffer, TrajectoryBufferState


@dataclass
class BufferConfig:
    buffer_size: int = 100_000
    batch_size: int = 64


def build_buffer(
    config: BufferConfig, env: gym.Env
) -> tuple[TrajectoryBuffer, TrajectoryBufferState]:
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
    buffer = fbx.make_flat_buffer(
        max_length=config.buffer_size,
        min_length=1,
        sample_batch_size=config.batch_size,
        add_batch_size=1,
    )
    buffer = buffer.replace(  # type: ignore[attr-defined]
        init=jax.jit(buffer.init),
        add=jax.jit(buffer.add, donate_argnums=0),
        sample=jax.jit(buffer.sample),
        can_sample=jax.jit(buffer.can_sample),
    )
    example = {
        "obs": jnp.zeros(obs_dim),
        "action": jnp.zeros(act_dim),
        "reward": jnp.array(0.0),
        "done": jnp.array(0.0),
    }
    state = buffer.init(example)
    return buffer, state
