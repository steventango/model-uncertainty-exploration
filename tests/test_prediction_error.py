import jax
import jax.numpy as jnp
from flax import nnx

from config import PredictionErrorConfig
from prediction_error import make_prediction_error, make_train_prediction_error


def test_error_predictor_hooked_by_noise():
    """Noisy-TV data: action 2 writes N(0, 1) into the second obs dim, the other
    actions are deterministic. e(s, a) should learn the irreducible error."""
    n, act_dim = 2048, 3
    k_obs, k_act, k_noise, k_init, k_train = jax.random.split(jax.random.key(0), 5)
    obs = jnp.stack([jax.random.uniform(k_obs, (n,), minval=-1.0), jnp.zeros(n)], -1)
    action = jax.random.randint(k_act, (n, 1), 0, act_dim)
    noisy = action[:, 0] == 2
    next_obs = jnp.stack(
        [
            obs[:, 0] + 0.1 * (action[:, 0] - 1),
            jnp.where(noisy, jax.random.normal(k_noise, (n,)), 0.0),
        ],
        -1,
    )

    config = PredictionErrorConfig(update_steps=1000)
    pe, optimizer = make_prediction_error(
        config, 2 + act_dim, 2, act_dim, nnx.Rngs(params=k_init)
    )
    forward_losses, error_losses = make_train_prediction_error(config)(
        pe, optimizer, obs, action, next_obs, n, k_train
    )
    assert forward_losses.shape == error_losses.shape == (config.update_steps,)

    bonus = jax.vmap(pe)(obs, action)
    assert jnp.all(bonus >= 0.0)
    noisy_bonus = bonus[noisy].mean()
    clean_bonus = bonus[~noisy].mean()
    assert noisy_bonus > 0.5
    assert noisy_bonus > 10 * clean_bonus
