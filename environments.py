import jax.numpy as jnp

from wrappers import StateReconstructionWrapper


class PendulumWrapper(StateReconstructionWrapper):
    def get_state(self, obs, last_action, time, next_obs=None):
        from gymnax.environments.classic_control.pendulum import EnvState

        return EnvState(
            theta=jnp.arctan2(obs[1], obs[0]),
            theta_dot=obs[2],
            last_u=jnp.reshape(last_action, ()),
            time=time,
        )


class MountainCarWrapper(StateReconstructionWrapper):
    def get_state(self, obs, last_action, time, next_obs=None):
        from gymnax.environments.classic_control.mountain_car import EnvState

        return EnvState(position=obs[0], velocity=obs[1], time=time)


class MountainCarContinuousWrapper(StateReconstructionWrapper):
    def get_state(self, obs, last_action, time, next_obs=None):
        from gymnax.environments.classic_control.continuous_mountain_car import EnvState

        return EnvState(position=obs[0], velocity=obs[1], time=time)


class CartPoleWrapper(StateReconstructionWrapper):
    def get_state(self, obs, last_action, time, next_obs=None):
        from gymnax.environments.classic_control.cartpole import EnvState

        return EnvState(
            x=obs[0], x_dot=obs[1], theta=obs[2], theta_dot=obs[3], time=time
        )


class AcrobotWrapper(StateReconstructionWrapper):
    def get_state(self, obs, last_action, time, next_obs=None):
        from gymnax.environments.classic_control.acrobot import EnvState

        return EnvState(
            joint_angle1=jnp.arctan2(obs[1], obs[0]),
            joint_angle2=jnp.arctan2(obs[3], obs[2]),
            velocity_1=obs[4],
            velocity_2=obs[5],
            time=time,
        )


# ---------------------------------------------------------------------------
# Brax wrappers
# ---------------------------------------------------------------------------

class BraxStateReconstructionWrapper(StateReconstructionWrapper):
    """Base for brax envs: state reconstruction from obs is not possible."""

    def get_state(self, obs, last_action, time, next_obs=None):
        raise NotImplementedError(
            "Brax pipeline states cannot be reconstructed from observations. "
            "Use predict_reward_terminated=True (model predictions) or ensure "
            "the wrapper implements obs_to_reward_terminated()."
        )


class InvertedPendulumBraxWrapper(BraxStateReconstructionWrapper):
    """obs = [cart_pos, pole_angle, cart_vel, pole_ang_vel]"""

    def obs_to_reward_terminated(self, obs, action, next_obs):
        terminated = jnp.abs(next_obs[1]) > 0.2
        reward = jnp.ones(())
        return reward, terminated


class HalfCheetahBraxWrapper(BraxStateReconstructionWrapper):
    """obs = [q[1:], qd]; qd[0] = next_obs[8] ≈ x_velocity."""

    def obs_to_reward_terminated(self, obs, action, next_obs):
        forward_reward = next_obs[8]  # qd[0] ≈ rootx velocity
        ctrl_cost = 0.1 * jnp.sum(jnp.square(action))
        reward = forward_reward - ctrl_cost
        terminated = jnp.bool_(False)
        return reward, terminated


class SwimmerBraxWrapper(BraxStateReconstructionWrapper):
    """obs = [q[2:], qd]; qd[0] = next_obs[3] ≈ x_velocity."""

    def obs_to_reward_terminated(self, obs, action, next_obs):
        forward_reward = next_obs[3]  # qd[0] ≈ rootx velocity
        ctrl_cost = 1e-4 * jnp.sum(jnp.square(action))
        reward = forward_reward - ctrl_cost
        terminated = jnp.bool_(False)
        return reward, terminated


class HopperBraxWrapper(BraxStateReconstructionWrapper):
    """obs = [rootz, rooty, thigh, leg, foot, rootx_vel, ...]; 11D.

    healthy_z_range=(0.7, inf), healthy_angle_range=(-0.2, 0.2).
    """

    def obs_to_reward_terminated(self, obs, action, next_obs):
        z = next_obs[0]       # q[rootz] ≈ Cartesian z height
        angle = next_obs[1]   # q[rooty] = trunk lean (brax uses this directly)
        is_healthy = (z > 0.7) & (angle > -0.2) & (angle < 0.2)
        terminated = ~is_healthy
        x_vel = next_obs[5]   # qd[rootx] ≈ forward velocity
        reward = x_vel + 1.0 - 1e-3 * jnp.sum(jnp.square(action))
        return reward, terminated


class Walker2dBraxWrapper(BraxStateReconstructionWrapper):
    """obs = [rootz, rooty, 6 leg angles, rootx_vel, ...]; 17D.

    healthy_z_range=(0.8, 2.0), healthy_angle_range=(-1.0, 1.0).
    """

    def obs_to_reward_terminated(self, obs, action, next_obs):
        z = next_obs[0]       # q[rootz]
        angle = next_obs[1]   # q[rooty]
        is_healthy = (z > 0.8) & (z < 2.0) & (angle > -1.0) & (angle < 1.0)
        terminated = ~is_healthy
        x_vel = next_obs[8]   # qd[rootx]
        reward = x_vel + 1.0 - 1e-3 * jnp.sum(jnp.square(action))
        return reward, terminated


class AntBraxWrapper(BraxStateReconstructionWrapper):
    """obs = [rootz, quat(4), 8 joints, vx, vy, vz, ...]; 27D.

    healthy_z_range=(0.2, 1.0). Ignores contact_cost (not in obs).
    """

    def obs_to_reward_terminated(self, obs, action, next_obs):
        z = next_obs[0]       # q[rootz]
        is_healthy = (z > 0.2) & (z < 1.0)
        terminated = ~is_healthy
        x_vel = next_obs[13]  # qd[vx] = rootx velocity
        reward = x_vel + 1.0 - 0.5 * jnp.sum(jnp.square(action))
        return reward, terminated


class ReacherBraxWrapper(BraxStateReconstructionWrapper):
    """obs[-3:] = fingertip-to-target vector; reward = -norm(vec) - sum(action^2)."""

    def obs_to_reward_terminated(self, obs, action, next_obs):
        reward = -jnp.linalg.norm(next_obs[-3:]) - jnp.sum(jnp.square(action))
        terminated = jnp.bool_(False)
        return reward, terminated


class PusherBraxWrapper(BraxStateReconstructionWrapper):
    """obs[14:17]=fingertip, obs[17:20]=object, obs[20:23]=goal.

    reward = -norm(obj-goal) - 0.1*sum(action^2) - 0.5*norm(finger-obj).
    """

    def obs_to_reward_terminated(self, obs, action, next_obs):
        fingertip = next_obs[14:17]
        obj = next_obs[17:20]
        goal = next_obs[20:23]
        reward = (
            -jnp.linalg.norm(obj - goal)
            - 0.1 * jnp.sum(jnp.square(action))
            - 0.5 * jnp.linalg.norm(fingertip - obj)
        )
        terminated = jnp.bool_(False)
        return reward, terminated


class InvertedDoublePendulumBraxWrapper(BraxStateReconstructionWrapper):
    """obs = [x, sin(q1), sin(q2), cos(q1), cos(q2), xd, q1d, q2d]; 8D.

    Tip is 0.6m from body[2] (pole2 joint) along pole2's axis.
    Link lengths: L1 = L2 = 0.6m.  y = L1*cos(q1) + L2*cos(q1+q2).
    Termination: y <= 1.  Maximum upright height is ~1.2m.
    """

    def obs_to_reward_terminated(self, obs, action, next_obs):
        sq1, sq2 = next_obs[1], next_obs[2]
        cq1, cq2 = next_obs[3], next_obs[4]
        sq1q2 = sq1 * cq2 + cq1 * sq2   # sin(q1+q2)
        cq1q2 = cq1 * cq2 - sq1 * sq2   # cos(q1+q2)
        L = 0.6
        x_tip = next_obs[0] + L * sq1 + L * sq1q2
        y_tip = L * cq1 + L * cq1q2
        terminated = y_tip <= 1.0
        v1, v2 = next_obs[6], next_obs[7]
        dist_penalty = 0.01 * x_tip ** 2 + (y_tip - 2.0) ** 2
        vel_penalty = 1e-3 * v1 ** 2 + 5e-3 * v2 ** 2
        reward = jnp.where(~terminated, 10.0, 0.0) - dist_penalty - vel_penalty
        return reward, terminated


class GenericBraxWrapper(BraxStateReconstructionWrapper):
    """Brax envs where reward/termination cannot be derived from obs alone."""


_WRAPPER_MAP = {
    "Pendulum-v1": PendulumWrapper,
    "MountainCar-v0": MountainCarWrapper,
    "MountainCarContinuous-v0": MountainCarContinuousWrapper,
    "CartPole-v1": CartPoleWrapper,
    "Acrobot-v1": AcrobotWrapper,
    # Brax environments
    "inverted_pendulum": InvertedPendulumBraxWrapper,
    "halfcheetah": HalfCheetahBraxWrapper,
    "swimmer": SwimmerBraxWrapper,
    "hopper": HopperBraxWrapper,
    "walker2d": Walker2dBraxWrapper,
    "ant": AntBraxWrapper,
    "reacher": ReacherBraxWrapper,
    "pusher": PusherBraxWrapper,
    "inverted_double_pendulum": InvertedDoublePendulumBraxWrapper,
    "humanoid": GenericBraxWrapper,
    "humanoidstandup": GenericBraxWrapper,
}

BRAX_ENVS = frozenset({
    "inverted_pendulum", "halfcheetah", "swimmer",
    "inverted_double_pendulum", "hopper", "walker2d",
    "ant", "humanoid", "humanoidstandup", "pusher", "reacher",
})

BRAX_BACKENDS = {"swimmer": "generalized"}


def make_state_reconstruction_wrapper(env, env_name: str) -> StateReconstructionWrapper:
    cls = _WRAPPER_MAP.get(env_name)
    if cls is None:
        raise ValueError(
            f"No state reconstruction wrapper for env: {env_name}. Supported: {list(_WRAPPER_MAP)}"
        )
    return cls(env)
