import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnax.environments.classic_control.mountain_car import EnvState as BaseEnvState

import ground_truth
from env_config import sample_validation_batch
from environments import make_gymnax_env, make_state_reconstruction_wrapper
from noisy_mountain_car import EnvState

ENV_NAME = "NoisyMountainCar-v0"


@pytest.fixture(scope="module")
def envs():
    env, params = make_gymnax_env(ENV_NAME)
    base, base_params = gymnax.make("MountainCar-v0")
    return env, params, base, base_params


def _states(n=64, seed=0):
    k1, k2 = jax.random.split(jax.random.key(seed))
    position = jax.random.uniform(k1, (n,), minval=-1.2, maxval=0.6)
    velocity = jax.random.uniform(k2, (n,), minval=-0.07, maxval=0.07)
    return position, velocity


def test_spaces_and_reset(envs):
    env, params, _, _ = envs
    assert env.action_space(params).n == 4
    assert env.observation_space(params).shape == (3,)
    obs, state = env.reset(jax.random.key(0), params)
    assert obs.shape == (3,)
    assert obs[2] == 0.0 and state.noise == 0.0


@pytest.mark.parametrize("action", [0, 1, 2, 3])
def test_step_matches_mountain_car(envs, action):
    env, params, base, base_params = envs
    base_action = 1 if action == 3 else action
    position, velocity = _states()
    keys = jax.random.split(jax.random.key(1), position.shape[0])

    def one(key, p, v):
        state = EnvState(position=p, velocity=v, time=0, noise=jnp.zeros(()))
        base_state = BaseEnvState(position=p, velocity=v, time=0)
        obs, _, r, d, _ = env.step_env(key, state, action, params)
        base_obs, _, base_r, base_d, _ = base.step_env(
            key, base_state, base_action, base_params
        )
        return obs, r, d, base_obs, base_r, base_d

    obs, r, d, base_obs, base_r, base_d = jax.vmap(one)(keys, position, velocity)
    np.testing.assert_array_equal(obs[:, :2], base_obs)
    np.testing.assert_array_equal(r, base_r)
    np.testing.assert_array_equal(d, base_d)
    if action == 3:
        assert np.all(obs[:, 2] != 0.0)
        assert np.std(np.asarray(obs[:, 2])) > 0.5
    else:
        np.testing.assert_array_equal(obs[:, 2], 0.0)


def test_terminates_at_goal(envs):
    env, params, _, _ = envs
    state = EnvState(
        position=jnp.array(0.49), velocity=jnp.array(0.02), time=0, noise=jnp.zeros(())
    )
    _, _, reward, done, _ = env.step_env(jax.random.key(0), state, 3, params)
    assert reward == -1.0 and done


def test_state_reconstruction(envs):
    env, *_ = envs
    wrapper = make_state_reconstruction_wrapper(env, ENV_NAME)
    obs = jnp.array([-0.3, 0.01, 0.7])
    state = wrapper.get_state(obs, 3, 5)
    np.testing.assert_array_equal(env.get_obs(state), obs)
    assert state.time == 5


def test_ground_truth_is_noise_free_mean(envs):
    env, params, _, _ = envs
    _, obs, s1, s2, act = sample_validation_batch(
        jax.random.key(0), env, params, ENV_NAME, 256
    )
    np.testing.assert_array_equal(obs[:, 2], 0.0)
    assert set(np.unique(np.asarray(act))) == {0, 1, 2, 3}
    delta = ground_truth.true_delta_obs(env, params, ENV_NAME, s1, s2, act[:, 0])
    np.testing.assert_array_equal(delta[:, 2], 0.0)
