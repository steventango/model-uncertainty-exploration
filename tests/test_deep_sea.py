"""Tests for what environments/bsuite/deep_sea.py adds on top of gymnax's DeepSea: the action map
carried by params (one per seed), the one-hot observation over reachable cells,
and the reward/termination oracle used by the model env."""

import jax
import jax.numpy as jnp
import pytest

from environments.bsuite.deep_sea import KeyedDeepSea
from wrappers import OneHotObservationWrapper

SIZE = 5
SEEDS = jnp.arange(4)


@pytest.fixture(scope="module")
def env():
    return OneHotObservationWrapper(KeyedDeepSea(SIZE))


@pytest.fixture(scope="module")
def params(env):
    return jax.vmap(env.params_for_seed)(SEEDS)


def _reset(env, params, key=0):
    keys = jax.random.split(jax.random.key(key), len(SEEDS))
    return jax.vmap(env.reset)(keys, params)


def test_action_map_differs_per_seed_and_is_in_state(env, params):
    maps = jax.vmap(env._env.action_map)(params)
    assert len({m.tobytes() for m in maps}) == len(SEEDS)
    _, state = _reset(env, params)
    assert (state.action_mapping == maps).all()


def test_action_map_is_a_function_of_the_seed_only(env, params):
    _, a = _reset(env, params, key=1)
    _, b = _reset(env, params, key=2)
    assert (a.action_mapping == b.action_mapping).all()
    same_seed = env.params_for_seed(SEEDS[0])
    assert (env._env.action_map(same_seed) == a.action_mapping[0]).all()


def test_action_map_survives_autoreset(env, params):
    def run(p, key):
        _, state = env.reset(key, p)

        def body(carry, k):
            state = carry
            action = jax.random.randint(k, (), 0, 2)
            _, state, _, terminated, _, _ = env.step(k, state, action, p)
            return state, (state.action_mapping, terminated)

        return jax.lax.scan(body, state, jax.random.split(key, 4 * SIZE))[1]

    keys = jax.random.split(jax.random.key(0), len(SEEDS))
    maps, terminated = jax.vmap(run)(params, keys)
    assert terminated.sum() >= len(SEEDS)  # several episodes per seed
    assert (maps == maps[:, :1]).all()


def test_observation_is_one_hot_over_reachable_cells(env, params):
    obs, state = _reset(env, params)
    assert obs.shape == (len(SEEDS), env.num_cells)
    assert env.num_cells == SIZE * (SIZE + 1) // 2
    assert (obs.argmax(-1) == 0).all() and (obs.sum(-1) == 1).all()

    # Walking down a diagonal visits cells 0, 1, 3, 6, ... (row * (row + 1) / 2 + col).
    p, key = jax.tree.map(lambda x: x[0], params), jax.random.key(0)
    obs, state = env.reset(key, p)
    a_map = env._env.action_map(p)
    cells = [int(obs.argmax())]
    for _ in range(SIZE - 1):
        action = a_map[state.row, state.column]
        obs, state, _, _, _, _ = env.step(key, state, action, p)
        cells.append(int(obs.argmax()))
    assert cells == [r * (r + 1) // 2 + r for r in range(SIZE)]


def test_observation_after_the_bottom_row_is_the_reset_cell(env, params):
    p = jax.tree.map(lambda x: x[0], params)
    key = jax.random.key(0)
    obs, state = env.reset(key, p)
    a_map = env._env.action_map(p)
    for _ in range(SIZE):
        obs, state, _, terminated, _, info = env.step(
            key, state, a_map[state.row, state.column], p
        )
    assert terminated
    assert obs.argmax() == 0  # autoreset
    assert info["final_observation"].argmax() == 0  # VBE: cell 0, not all zeros


def test_optimal_policy_reaches_the_goal_on_every_seed(env, params):
    def run(p, key):
        obs, state = env.reset(key, p)
        a_map = env._env.action_map(p)

        def body(carry, k):
            obs, state = carry
            obs, state, reward, _, _, _ = env.step(
                k, state, a_map[state.row, state.column], p
            )
            return (obs, state), reward

        return jax.lax.scan(body, (obs, state), jax.random.split(key, SIZE))[1].sum()

    keys = jax.random.split(jax.random.key(0), len(SEEDS))
    returns = jax.vmap(run)(params, keys)
    assert jnp.allclose(returns, 1 - 0.01)  # goal reward minus the move cost


def test_oracle_matches_env_step(env, params):
    def run(p, key):
        obs, state = env.reset(key, p)

        def body(carry, k):
            obs, state = carry
            action = jax.random.randint(k, (), 0, 2)
            next_obs, state, reward, terminated, _, info = env.step(k, state, action, p)
            oracle_reward, oracle_terminated = env.obs_to_reward_terminated(
                obs, action, info["final_observation"], p
            )
            return (next_obs, state), (
                reward,
                terminated,
                oracle_reward,
                oracle_terminated,
            )

        return jax.lax.scan(body, (obs, state), jax.random.split(key, 10 * SIZE))[1]

    keys = jax.random.split(jax.random.key(0), len(SEEDS))
    reward, terminated, oracle_reward, oracle_terminated = jax.vmap(run)(params, keys)
    assert terminated.any()
    assert jnp.allclose(reward, oracle_reward, atol=1e-6)
    assert (terminated == oracle_terminated).all()


def test_oracle_is_linear_in_a_soft_observation(env, params):
    """The model env feeds predicted (soft) one-hot vectors: the oracle must give
    the probability-weighted reward."""
    p = jax.tree.map(lambda x: x[0], params)
    goal = env.num_cells - 1
    right = env._env.action_map(p)[env.cell_rows, env.cell_cols][goal]
    at_goal, elsewhere = (
        jax.nn.one_hot(goal, env.num_cells),
        jax.nn.one_hot(0, env.num_cells),
    )
    r_goal, _ = env.obs_to_reward_terminated(at_goal, right, at_goal, p)
    r_else, _ = env.obs_to_reward_terminated(elsewhere, right, at_goal, p)
    soft = 0.3 * at_goal + 0.7 * elsewhere
    r_soft, _ = env.obs_to_reward_terminated(soft, right, at_goal, p)
    assert jnp.allclose(r_soft, 0.3 * r_goal + 0.7 * r_else, atol=1e-6)


def test_inner_observation_is_the_cell_index(env, params):
    inner = env._env
    p = jax.tree.map(lambda x: x[0], params)
    key = jax.random.key(0)
    obs, state = inner.reset(key, p)
    assert obs.dtype == jnp.int32 and obs == 0
    assert inner.observation_space(p).n == env.num_cells
    obs, state, *_ = inner.step(key, state, inner.action_map(p)[0, 0], p)
    assert obs in (1, 2)


def test_one_hot_wrapper_space(env):
    space = env.observation_space(env.default_params)
    assert space.shape == (env.num_cells,)
    assert jnp.all(space.low == 0) and jnp.all(space.high == 1)
