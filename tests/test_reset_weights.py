import jax
import jax.numpy as jnp

from model_env import inverse_count_weights, inverse_density_weights, mix_with_uniform


def test_inverse_count_is_uniform_over_distinct_states():
    index = jnp.array([0, 0, 0, 1, 2, 2, 9])
    valid = jnp.arange(7) < 6  # the last row is unfilled
    w = inverse_count_weights(index, valid, 4)
    w = w / w.sum()
    mass = jnp.zeros(4).at[index].add(w)
    assert jnp.allclose(mass, jnp.array([1 / 3, 1 / 3, 1 / 3, 0.0]))
    assert w[-1] == 0


def test_inverse_density_upweights_sparse_region_and_ignores_invalid_rows():
    key = jax.random.key(0)
    dense = jax.random.normal(key, (900, 2)) * 0.1
    sparse = jax.random.normal(jax.random.key(1), (100, 2)) * 0.1 + 5.0
    obs = jnp.concatenate([dense, sparse, jnp.full((50, 2), 1e6)])
    valid = jnp.arange(1050) < 1000
    w = inverse_density_weights(obs, valid, jax.random.key(2))
    w = w / w.sum()
    assert float(w[1000:].sum()) == 0.0
    # Uniform over the covered region puts about half the mass on each cluster,
    # where uniform over rows would put 10% on the sparse one.
    assert 0.3 < float(w[900:1000].sum()) < 0.7


def test_inverse_density_is_near_uniform_for_uniform_data():
    obs = jax.random.uniform(jax.random.key(0), (1000, 3))
    w = inverse_density_weights(obs, jnp.ones(1000, bool), jax.random.key(1))
    w = w / w.sum()
    assert float(w.max() * 1000) < 100.0  # floored: no row exceeds 100x average


def test_mix_with_uniform():
    valid = jnp.arange(5) < 4
    w = mix_with_uniform(jnp.array([0.0, 0.0, 3.0, 1.0, 99.0]), valid, 0.2)
    assert jnp.isclose(w.sum(), 1.0)
    assert w[4] == 0.0
    assert jnp.allclose(w[:2], 0.2 / 4)
    assert w[2] > w[3] > w[0]
    # No uncertainty anywhere: falls back to uniform rather than dividing by zero.
    flat = mix_with_uniform(jnp.zeros(5), valid, 0.2)
    assert jnp.allclose(flat[:4] / flat.sum(), 0.25)
