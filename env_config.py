"""Per-environment configuration for validation and visualization."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp


SUPPORTED_ENVS = ("Pendulum-v1", "MountainCarContinuous-v0")


@dataclass(frozen=True)
class EnvConfig:
    name: str
    dynamics_dim: int
    delta_obs_labels: tuple[str, ...]
    representative_actions: tuple[float, ...]
    x_label: str
    y_label: str
    dynamics_label: str


def get_env_config(env_name: str) -> EnvConfig:
    configs = {
        "Pendulum-v1": EnvConfig(
            name="Pendulum-v1",
            dynamics_dim=2,
            delta_obs_labels=(
                r"$\Delta \cos(\theta)$",
                r"$\Delta \sin(\theta)$",
                r"$\Delta \dot{\theta}$",
            ),
            representative_actions=(-2.0, 0.0, 2.0),
            x_label="Theta (rad)",
            y_label="Theta Dot (rad/s)",
            dynamics_label="Delta Theta Dot",
        ),
        "MountainCarContinuous-v0": EnvConfig(
            name="MountainCarContinuous-v0",
            dynamics_dim=1,
            delta_obs_labels=(r"$\Delta x$", r"$\Delta \dot{x}$"),
            representative_actions=(-1.0, 0.0, 1.0),
            x_label="Position",
            y_label="Velocity",
            dynamics_label="Delta Velocity",
        ),
    }
    if env_name not in configs:
        raise ValueError(
            f"Unsupported env: {env_name}. Supported: {list(configs)}"
        )
    return configs[env_name]


def make_state(env_name: str, s1, s2):
    if env_name == "Pendulum-v1":
        from gymnax.environments.classic_control.pendulum import EnvState

        return EnvState(theta=s1, theta_dot=s2, last_u=jnp.zeros(()), time=0)
    if env_name == "MountainCarContinuous-v0":
        from gymnax.environments.classic_control.continuous_mountain_car import (
            EnvState,
        )

        return EnvState(position=s1, velocity=s2, time=0)
    raise ValueError(env_name)


def obs_from_coords(env_name: str, s1, s2):
    if env_name == "Pendulum-v1":
        return jnp.stack([jnp.cos(s1), jnp.sin(s1), s2], axis=-1)
    if env_name == "MountainCarContinuous-v0":
        return jnp.stack([s1, s2], axis=-1)
    raise ValueError(env_name)


def grid_coords(env_name: str, env_params, num_grid: int = 100):
    if env_name == "Pendulum-v1":
        s1 = jnp.linspace(-jnp.pi, jnp.pi, num_grid)
        s2 = jnp.linspace(-env_params.max_speed, env_params.max_speed, num_grid)
    elif env_name == "MountainCarContinuous-v0":
        s1 = jnp.linspace(env_params.min_position, env_params.max_position, num_grid)
        s2 = jnp.linspace(-env_params.max_speed, env_params.max_speed, num_grid)
    else:
        raise ValueError(env_name)
    return s1, s2


def sample_validation_batch(rng, env_name: str, env_params, num_samples: int):
    if env_name == "Pendulum-v1":
        rng, key_s1, key_s2, key_act = jax.random.split(rng, 4)
        s1 = jax.random.uniform(key_s1, (num_samples,), minval=-jnp.pi, maxval=jnp.pi)
        s2 = jax.random.uniform(
            key_s2, (num_samples,), minval=-env_params.max_speed, maxval=env_params.max_speed
        )
        act = jax.random.uniform(
            key_act, (num_samples, 1), minval=-env_params.max_torque, maxval=env_params.max_torque
        )
    elif env_name == "MountainCarContinuous-v0":
        rng, key_s1, key_s2, key_act = jax.random.split(rng, 4)
        s1 = jax.random.uniform(
            key_s1, (num_samples,), minval=env_params.min_position, maxval=env_params.max_position
        )
        s2 = jax.random.uniform(
            key_s2, (num_samples,), minval=-env_params.max_speed, maxval=env_params.max_speed
        )
        act = jax.random.uniform(
            key_act, (num_samples, 1), minval=env_params.min_action, maxval=env_params.max_action
        )
    else:
        raise ValueError(env_name)

    obs = obs_from_coords(env_name, s1, s2)
    x = jnp.concatenate([obs, act], axis=-1)
    return rng, x, obs, s1, s2, act


def action_visit_mask(env_name: str, visited_actions, act: float):
    visited = visited_actions[:, 0] if visited_actions.ndim > 1 else visited_actions
    low, _, high = get_env_config(env_name).representative_actions
    threshold = (high - low) / 6
    if act == low:
        return visited <= low + threshold
    if act == high:
        return visited >= high - threshold
    return (visited > low + threshold) & (visited < high - threshold)


def visited_coords_from_obs(env_name: str, obs):
    if env_name == "Pendulum-v1":
        return jnp.arctan2(obs[:, 1], obs[:, 0]), obs[:, 2]
    if env_name == "MountainCarContinuous-v0":
        return obs[:, 0], obs[:, 1]
    raise ValueError(env_name)


def action_title(env_name: str, act: float) -> str:
    low, mid, high = get_env_config(env_name).representative_actions
    threshold = (high - low) / 6
    if act == low:
        return rf"Action $u \approx {act}$ (binned $u \leq {low + threshold:.2f}$)"
    if act == high:
        return rf"Action $u \approx {act}$ (binned $u \geq {high - threshold:.2f}$)"
    return rf"Action $u \approx {act}$ (binned ${low + threshold:.2f} < u < {high - threshold:.2f}$)"
