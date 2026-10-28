"""Gymnasium MuJoCo v5 tasks ported to JAX via MJX.

Each env loads Gymnasium's own XML and reproduces its ``*_v5.py`` observation,
reward and reset distribution, so the task matches ``gymnasium.make(...)``
rather than brax's re-tuned versions. MJX matches MuJoCo C to machine
precision in float64; under the default float32 trajectories drift faster
(most visibly through half_cheetah's ground contacts).

Exposes the same interface as :class:`wrappers.BraxGymnaxWrapper`.
"""

import os

import gymnasium
import jax
import jax.numpy as jnp
import mujoco
from flax import struct
from gymnax.environments import environment, spaces
from mujoco import mjx

_ASSETS = os.path.join(os.path.dirname(gymnasium.__file__), "envs", "mujoco", "assets")


@struct.dataclass
class GymnasiumMJXState:
    data: mjx.Data
    time: jnp.ndarray


class GymnasiumMJXEnv:
    xml_file: str
    frame_skip: int
    episode_length: int

    def __init__(self):
        self._mj_model = mujoco.MjModel.from_xml_path(os.path.join(_ASSETS, self.xml_file))
        self._model = mjx.put_model(self._mj_model)
        self._init_qpos = jnp.asarray(self._mj_model.qpos0)
        self._init_qvel = jnp.zeros(self._mj_model.nv)
        self.dt = self._mj_model.opt.timestep * self.frame_skip

    # Per-env hooks, ported from gymnasium/envs/mujoco/<env>_v5.py.
    def _reset_qpos_qvel(self, key):
        raise NotImplementedError

    def _get_obs(self, data):
        raise NotImplementedError

    def _get_reward(self, data_before, data_after, action):
        raise NotImplementedError

    @property
    def default_params(self):
        return environment.EnvParams(max_steps_in_episode=self.episode_length)

    def reset(self, key, params=None):
        qpos, qvel = self._reset_qpos_qvel(key)
        data = mjx.make_data(self._model).replace(qpos=qpos, qvel=qvel)
        # Gymnasium's set_state runs mj_forward so body positions match qpos.
        data = mjx.forward(self._model, data)
        state = GymnasiumMJXState(data=data, time=jnp.zeros((), jnp.int32))
        return self._get_obs(data), state

    reset_env = reset

    def _physics_step(self, data, action):
        data = data.replace(ctrl=action)
        return jax.lax.fori_loop(
            0, self.frame_skip, lambda _, d: mjx.step(self._model, d), data
        )

    def step_env(self, key, state, action, params=None):
        data = self._physics_step(state.data, action)
        next_state = GymnasiumMJXState(data=data, time=state.time + 1)
        obs = self._get_obs(data)
        reward = self._get_reward(state.data, data, action)
        # None of the ported tasks terminate; episodes end by TimeLimit only.
        terminated = jnp.bool_(False)
        return obs, next_state, reward, terminated, {}

    def step(self, key, state, action, params=None):
        next_obs, next_state, reward, terminated, _ = self.step_env(
            key, state, action, params
        )
        truncated = (next_state.time >= self.episode_length) & ~terminated
        done = terminated | truncated
        info = {"next_obs": next_obs}
        reset_obs, reset_state = self.reset(key, params)
        state_out = jax.tree_util.tree_map(
            lambda r, s: jnp.where(done, r, s), reset_state, next_state
        )
        obs_out = jnp.where(done, reset_obs, next_obs)
        return obs_out, state_out, reward, terminated, truncated, info

    def observation_space(self, params):
        obs = jax.eval_shape(lambda: self._get_obs(mjx.make_data(self._model)))
        return spaces.Box(low=-jnp.inf, high=jnp.inf, shape=obs.shape)

    def action_space(self, params):
        ctrlrange = self._mj_model.actuator_ctrlrange
        return spaces.Box(
            low=jnp.asarray(ctrlrange[:, 0], jnp.float32),
            high=jnp.asarray(ctrlrange[:, 1], jnp.float32),
            shape=(self._mj_model.nu,),
        )


class ReacherV5(GymnasiumMJXEnv):
    xml_file = "reacher.xml"
    frame_skip = 2
    episode_length = 50

    def __init__(self):
        super().__init__()
        self._fingertip = self._mj_model.body("fingertip").id
        self._target = self._mj_model.body("target").id

    def _reset_qpos_qvel(self, key):
        k_qpos, k_goal, k_qvel = jax.random.split(key, 3)
        qpos = self._init_qpos + jax.random.uniform(
            k_qpos, self._init_qpos.shape, minval=-0.1, maxval=0.1
        )
        # Uniform on the radius-0.2 disk, as Gymnasium's rejection sampling gives.
        r = 0.2 * jnp.sqrt(jax.random.uniform(k_goal, ()))
        ang = 2 * jnp.pi * jax.random.uniform(jax.random.fold_in(k_goal, 1), ())
        qpos = qpos.at[-2:].set(r * jnp.array([jnp.cos(ang), jnp.sin(ang)]))
        qvel = self._init_qvel + jax.random.uniform(
            k_qvel, self._init_qvel.shape, minval=-0.005, maxval=0.005
        )
        qvel = qvel.at[-2:].set(0.0)
        return qpos, qvel

    def _fingertip_to_target(self, data):
        # Like Gymnasium, reads xpos straight after mj_step, i.e. from the last
        # substep's forward pass (one substep stale), not recomputed.
        return data.xpos[self._fingertip] - data.xpos[self._target]

    def _get_obs(self, data):
        theta = data.qpos[:2]
        return jnp.concatenate([
            jnp.cos(theta),
            jnp.sin(theta),
            data.qpos[2:],
            data.qvel[:2],
            self._fingertip_to_target(data)[:2],
        ])

    def _get_reward(self, data_before, data_after, action):
        reward_dist = -jnp.linalg.norm(self._fingertip_to_target(data_after))
        reward_ctrl = -jnp.sum(jnp.square(action))
        return reward_dist + reward_ctrl


class _ForwardLocomotionV5(GymnasiumMJXEnv):
    """Swimmer/HalfCheetah: reward = forward_weight * x_vel - ctrl_weight * |a|^2."""

    episode_length = 1000
    forward_reward_weight = 1.0
    ctrl_cost_weight: float
    reset_noise_scale = 0.1
    num_excluded_qpos: int

    def _get_obs(self, data):
        return jnp.concatenate([data.qpos[self.num_excluded_qpos :], data.qvel])

    def _get_reward(self, data_before, data_after, action):
        x_velocity = (data_after.qpos[0] - data_before.qpos[0]) / self.dt
        ctrl_cost = self.ctrl_cost_weight * jnp.sum(jnp.square(action))
        return self.forward_reward_weight * x_velocity - ctrl_cost


class SwimmerV5(_ForwardLocomotionV5):
    xml_file = "swimmer.xml"
    frame_skip = 4
    ctrl_cost_weight = 1e-4
    num_excluded_qpos = 2

    def _reset_qpos_qvel(self, key):
        k_qpos, k_qvel = jax.random.split(key)
        s = self.reset_noise_scale
        qpos = self._init_qpos + jax.random.uniform(
            k_qpos, self._init_qpos.shape, minval=-s, maxval=s
        )
        qvel = self._init_qvel + jax.random.uniform(
            k_qvel, self._init_qvel.shape, minval=-s, maxval=s
        )
        return qpos, qvel


class HalfCheetahV5(_ForwardLocomotionV5):
    xml_file = "half_cheetah.xml"
    frame_skip = 5
    ctrl_cost_weight = 0.1
    num_excluded_qpos = 1

    def _reset_qpos_qvel(self, key):
        k_qpos, k_qvel = jax.random.split(key)
        s = self.reset_noise_scale
        qpos = self._init_qpos + jax.random.uniform(
            k_qpos, self._init_qpos.shape, minval=-s, maxval=s
        )
        qvel = self._init_qvel + s * jax.random.normal(k_qvel, self._init_qvel.shape)
        return qpos, qvel


GYMNASIUM_MJX_ENVS = {
    "Reacher-v5": ReacherV5,
    "Swimmer-v5": SwimmerV5,
    "HalfCheetah-v5": HalfCheetahV5,
}


def make_gymnasium_mjx_env(env_name: str) -> GymnasiumMJXEnv:
    return GYMNASIUM_MJX_ENVS[env_name]()
