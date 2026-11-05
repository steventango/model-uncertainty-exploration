import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drifter_distractor import (
    CONSTANT,
    DISTRACTOR,
    DRIFTER,
    DrifterDistractor,
    EnvParams,
    action_targets,
)

ENV = DrifterDistractor()
PARAMS = ENV.default_params


def _run(actions, params=PARAMS, seed=0):
    """Step through ``actions``; returns obs (T,) and info mu (T, 4)."""
    reset_key, step_key = jax.random.split(jax.random.key(seed))
    _, state = ENV.reset(reset_key, params)

    def body(state, xs):
        key, action = xs
        obs, state, reward, terminated, truncated, info = ENV.step(
            key, state, action, params
        )
        return state, (obs[0], info["mu"], reward, terminated | truncated)

    keys = jax.random.split(step_key, len(actions))
    _, (obs, mu, reward, done) = jax.lax.scan(body, state, (keys, jnp.asarray(actions)))
    assert not np.any(np.asarray(done))
    assert np.all(np.asarray(reward) == 0.0)
    return np.asarray(obs), np.asarray(mu)


def test_reset():
    obs, state = ENV.reset(jax.random.key(0), PARAMS)
    np.testing.assert_array_equal(obs, [0.0])
    mu = np.asarray(state.mu)
    np.testing.assert_array_equal(mu[:3], 0.0)
    assert -50.0 <= mu[3] <= 50.0 and mu[3] != 0.0


def test_action_target_mapping():
    assert PARAMS.target_types == (DISTRACTOR, DISTRACTOR, DRIFTER, CONSTANT)
    np.testing.assert_array_equal(action_targets(0, PARAMS), [0, 1, 2, 3])
    switched = EnvParams(switch_step=5)
    np.testing.assert_array_equal(action_targets(4, switched), [0, 1, 2, 3])
    np.testing.assert_array_equal(action_targets(5, switched), [2, 3, 0, 1])


@pytest.mark.parametrize("action", [0, 1])
def test_distractor(action):
    obs, mu = _run([action] * 20_000)
    np.testing.assert_array_equal(mu[:, action], 0.0)
    assert abs(obs.mean()) < 0.05
    assert obs.var() == pytest.approx(1.0, rel=0.05)


def test_constant():
    _, state0 = ENV.reset(jax.random.split(jax.random.key(0))[0], PARAMS)
    obs, mu = _run([3] * 1000)
    np.testing.assert_array_equal(obs, float(state0.mu[3]))
    np.testing.assert_array_equal(mu[:, 3], float(state0.mu[3]))


def test_drifter_random_walk():
    obs, mu = _run([2] * 20_000)
    np.testing.assert_array_equal(obs, mu[:, 2])
    steps = np.diff(mu[:, 2])
    unclipped = (np.abs(mu[1:, 2]) < 50.0) & (np.abs(mu[:-1, 2]) < 50.0)
    assert steps[unclipped].var() == pytest.approx(0.1, rel=0.05)
    assert np.all(np.abs(mu[:, 2]) <= 50.0)
    # Drift continues: the mean keeps moving late in the run.
    assert np.abs(mu[-1000:, 2] - mu[-1001:-1, 2]).max() > 0.0


def test_drifter_clipped():
    params = EnvParams(drifter_step_var=100.0)
    _, mu = _run([0] * 2000, params)
    assert np.all(np.abs(mu[:, 2]) <= 50.0)
    assert np.any(np.abs(mu[:, 2]) == 50.0)


def test_drift_independent_of_action():
    _, mu_sampled = _run([2] * 500)
    _, mu_ignored = _run([0] * 500)
    np.testing.assert_allclose(mu_sampled, mu_ignored)


def test_switch():
    params = EnvParams(switch_step=100)
    obs, mu = _run([3] * 100 + [1] * 100, params)
    np.testing.assert_array_equal(obs[:100], mu[0, 3])
    np.testing.assert_array_equal(obs[100:], mu[0, 3])
    np.testing.assert_array_equal(mu[100:, 1], mu[0, 3])
