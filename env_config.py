"""Per-environment configuration for validation and visualization."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from gymnax.environments import spaces


SUPPORTED_ENVS = (
    "Pendulum-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "CartPole-v1",
    "Acrobot-v1",
)


@dataclass(frozen=True)
class EnvConfig:
    name: str
    dynamics_dim: int
    delta_obs_labels: tuple[str, ...]
    representative_actions: tuple[float, ...]
    x_label: str
    y_label: str
    dynamics_label: str


def is_discrete(env, env_params) -> bool:
    return isinstance(env.action_space(env_params), spaces.Discrete)


def num_actions(env, env_params) -> int | None:
    if is_discrete(env, env_params):
        return env.action_space(env_params).n
    return None


def policy_action_dim(env, env_params) -> int:
    if is_discrete(env, env_params):
        return env.action_space(env_params).n
    return env.action_space(env_params).shape[0]


def model_action_dim(env, env_params) -> int:
    n = num_actions(env, env_params)
    if n is not None:
        return n
    return env.action_space(env_params).shape[0]


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
        "MountainCar-v0": EnvConfig(
            name="MountainCar-v0",
            dynamics_dim=1,
            delta_obs_labels=(r"$\Delta x$", r"$\Delta \dot{x}$"),
            representative_actions=(0.0, 1.0, 2.0),
            x_label="Position",
            y_label="Velocity",
            dynamics_label="Delta Velocity",
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
        "CartPole-v1": EnvConfig(
            name="CartPole-v1",
            dynamics_dim=3,
            delta_obs_labels=(
                r"$\Delta x$",
                r"$\Delta \dot{x}$",
                r"$\Delta \theta$",
                r"$\Delta \dot{\theta}$",
            ),
            representative_actions=(0.0, 1.0),
            x_label="Cart Position",
            y_label="Pole Angle (rad)",
            dynamics_label="Delta Theta Dot",
        ),
        "Acrobot-v1": EnvConfig(
            name="Acrobot-v1",
            dynamics_dim=5,
            delta_obs_labels=(
                r"$\Delta \cos(\theta_1)$",
                r"$\Delta \sin(\theta_1)$",
                r"$\Delta \cos(\theta_2)$",
                r"$\Delta \sin(\theta_2)$",
                r"$\Delta \dot{\theta}_1$",
                r"$\Delta \dot{\theta}_2$",
            ),
            representative_actions=(0.0, 1.0, 2.0),
            x_label=r"$\theta_1$ (rad)",
            y_label=r"$\theta_2$ (rad)",
            dynamics_label=r"Delta $\dot{\theta}_2$",
        ),
    }
    if env_name not in configs:
        raise ValueError(f"Unsupported env: {env_name}. Supported: {list(configs)}")
    return configs[env_name]


def make_state(env_name: str, s1, s2):
    if env_name == "Pendulum-v1":
        from gymnax.environments.classic_control.pendulum import EnvState

        return EnvState(theta=s1, theta_dot=s2, last_u=jnp.zeros(()), time=0)
    if env_name in ("MountainCar-v0", "MountainCarContinuous-v0"):
        if env_name == "MountainCar-v0":
            from gymnax.environments.classic_control.mountain_car import EnvState
        else:
            from gymnax.environments.classic_control.continuous_mountain_car import (
                EnvState,
            )

        return EnvState(position=s1, velocity=s2, time=0)
    if env_name == "CartPole-v1":
        from gymnax.environments.classic_control.cartpole import EnvState

        return EnvState(
            x=s1,
            x_dot=jnp.zeros_like(s1),
            theta=s2,
            theta_dot=jnp.zeros_like(s2),
            time=0,
        )
    if env_name == "Acrobot-v1":
        from gymnax.environments.classic_control.acrobot import EnvState

        return EnvState(
            joint_angle1=s1,
            joint_angle2=s2,
            velocity_1=jnp.zeros_like(s1),
            velocity_2=jnp.zeros_like(s2),
            time=0,
        )
    raise ValueError(env_name)


def obs_from_coords(env_name: str, s1, s2):
    if env_name == "Pendulum-v1":
        return jnp.stack([jnp.cos(s1), jnp.sin(s1), s2], axis=-1)
    if env_name in ("MountainCar-v0", "MountainCarContinuous-v0"):
        return jnp.stack([s1, s2], axis=-1)
    if env_name == "CartPole-v1":
        return jnp.stack([s1, jnp.zeros_like(s1), s2, jnp.zeros_like(s2)], axis=-1)
    if env_name == "Acrobot-v1":
        return jnp.stack(
            [
                jnp.cos(s1),
                jnp.sin(s1),
                jnp.cos(s2),
                jnp.sin(s2),
                jnp.zeros_like(s1),
                jnp.zeros_like(s2),
            ],
            axis=-1,
        )
    raise ValueError(env_name)


def grid_coords(env_name: str, env_params, num_grid: int = 100):
    if env_name == "Pendulum-v1":
        s1 = jnp.linspace(-jnp.pi, jnp.pi, num_grid)
        s2 = jnp.linspace(-env_params.max_speed, env_params.max_speed, num_grid)
    elif env_name in ("MountainCar-v0", "MountainCarContinuous-v0"):
        s1 = jnp.linspace(env_params.min_position, env_params.max_position, num_grid)
        s2 = jnp.linspace(-env_params.max_speed, env_params.max_speed, num_grid)
    elif env_name == "CartPole-v1":
        # Extend slightly past failure thresholds so termination heatmaps show
        # the boundary band (gymnax uses strict >, so the legal box alone is all zeros).
        s1 = jnp.linspace(
            -1.1 * env_params.x_threshold,
            1.1 * env_params.x_threshold,
            num_grid,
        )
        s2 = jnp.linspace(
            -1.1 * env_params.theta_threshold_radians,
            1.1 * env_params.theta_threshold_radians,
            num_grid,
        )
    elif env_name == "Acrobot-v1":
        s1 = jnp.linspace(-jnp.pi, jnp.pi, num_grid)
        s2 = jnp.linspace(-jnp.pi, jnp.pi, num_grid)
    else:
        raise ValueError(env_name)
    return s1, s2


def sample_validation_batch(rng, env, env_params, env_name: str, num_samples: int):
    n_act = num_actions(env, env_params)
    if env_name == "Pendulum-v1":
        rng, key_s1, key_s2, key_act = jax.random.split(rng, 4)
        s1 = jax.random.uniform(key_s1, (num_samples,), minval=-jnp.pi, maxval=jnp.pi)
        s2 = jax.random.uniform(
            key_s2,
            (num_samples,),
            minval=-env_params.max_speed,
            maxval=env_params.max_speed,
        )
        act = jax.random.uniform(
            key_act,
            (num_samples, 1),
            minval=-env_params.max_torque,
            maxval=env_params.max_torque,
        )
    elif env_name == "MountainCar-v0":
        rng, key_s1, key_s2, key_act = jax.random.split(rng, 4)
        s1 = jax.random.uniform(
            key_s1,
            (num_samples,),
            minval=env_params.min_position,
            maxval=env_params.max_position,
        )
        s2 = jax.random.uniform(
            key_s2,
            (num_samples,),
            minval=-env_params.max_speed,
            maxval=env_params.max_speed,
        )
        act = jax.random.randint(key_act, (num_samples, 1), 0, n_act)
    elif env_name == "MountainCarContinuous-v0":
        rng, key_s1, key_s2, key_act = jax.random.split(rng, 4)
        s1 = jax.random.uniform(
            key_s1,
            (num_samples,),
            minval=env_params.min_position,
            maxval=env_params.max_position,
        )
        s2 = jax.random.uniform(
            key_s2,
            (num_samples,),
            minval=-env_params.max_speed,
            maxval=env_params.max_speed,
        )
        act = jax.random.uniform(
            key_act,
            (num_samples, 1),
            minval=env_params.min_action,
            maxval=env_params.max_action,
        )
    elif env_name == "CartPole-v1":
        rng, key_s1, key_s2, key_act = jax.random.split(rng, 4)
        s1 = jax.random.uniform(
            key_s1,
            (num_samples,),
            minval=-env_params.x_threshold,
            maxval=env_params.x_threshold,
        )
        s2 = jax.random.uniform(
            key_s2,
            (num_samples,),
            minval=-env_params.theta_threshold_radians,
            maxval=env_params.theta_threshold_radians,
        )
        act = jax.random.randint(key_act, (num_samples, 1), 0, n_act)
    elif env_name == "Acrobot-v1":
        rng, key_s1, key_s2, key_act = jax.random.split(rng, 4)
        s1 = jax.random.uniform(key_s1, (num_samples,), minval=-jnp.pi, maxval=jnp.pi)
        s2 = jax.random.uniform(key_s2, (num_samples,), minval=-jnp.pi, maxval=jnp.pi)
        act = jax.random.randint(key_act, (num_samples, 1), 0, n_act)
    else:
        raise ValueError(env_name)

    obs = obs_from_coords(env_name, s1, s2)
    return rng, obs, s1, s2, act


def action_visit_mask(env_name: str, visited_actions, act: float, *, discrete: bool):
    env_config = get_env_config(env_name)
    visited = visited_actions[:, 0] if visited_actions.ndim > 1 else visited_actions
    if discrete:
        return jnp.isclose(visited, act)
    low, _, high = env_config.representative_actions
    threshold = (high - low) / 6
    if act == low:
        return visited <= low + threshold
    if act == high:
        return visited >= high - threshold
    return (visited > low + threshold) & (visited < high - threshold)


def visited_coords_from_obs(env_name: str, obs):
    if env_name == "Pendulum-v1":
        return jnp.arctan2(obs[:, 1], obs[:, 0]), obs[:, 2]
    if env_name in ("MountainCar-v0", "MountainCarContinuous-v0"):
        return obs[:, 0], obs[:, 1]
    if env_name == "CartPole-v1":
        return obs[:, 0], obs[:, 2]
    if env_name == "Acrobot-v1":
        return jnp.arctan2(obs[:, 1], obs[:, 0]), jnp.arctan2(obs[:, 3], obs[:, 2])
    raise ValueError(env_name)


def action_title(env_name: str, act: float, *, discrete: bool) -> str:
    env_config = get_env_config(env_name)
    if discrete:
        return rf"Action $a = {int(act)}$"
    low, mid, high = env_config.representative_actions
    threshold = (high - low) / 6
    if act == low:
        return rf"Action $u \approx {act}$ (binned $u \leq {low + threshold:.2f}$)"
    if act == high:
        return rf"Action $u \approx {act}$ (binned $u \geq {high - threshold:.2f}$)"
    return rf"Action $u \approx {act}$ (binned ${low + threshold:.2f} < u < {high - threshold:.2f}$)"
