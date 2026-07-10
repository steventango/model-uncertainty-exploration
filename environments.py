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


_WRAPPER_MAP = {
    "Pendulum-v1": PendulumWrapper,
    "MountainCar-v0": MountainCarWrapper,
    "MountainCarContinuous-v0": MountainCarContinuousWrapper,
    "CartPole-v1": CartPoleWrapper,
    "Acrobot-v1": AcrobotWrapper,
}


def make_state_reconstruction_wrapper(env, env_name: str) -> StateReconstructionWrapper:
    cls = _WRAPPER_MAP.get(env_name)
    if cls is None:
        raise ValueError(
            f"No state reconstruction wrapper for env: {env_name}. Supported: {list(_WRAPPER_MAP)}"
        )
    return cls(env)
