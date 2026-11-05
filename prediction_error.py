"""Prediction-error ("curiosity") exploration bonus.

The bonus must be available inside imagined rollouts, where there is no real
next observation, so the forward model's error is amortized: an error predictor
regresses the forward model's squared error on real transitions and is queried
at (obs, action). Under irreducible noise the forward model converges to the
mean and the residual stays at the noise variance, so the bonus is hooked by
aleatoric noise (a noisy TV), unlike the world model's epistemic bonuses.
"""

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from config import PredictionErrorConfig
from rnd import MLP


class PredictionError(nnx.Module):
    """Forward model f(s, a) → standardized Δobs and error predictor
    e(s, a) → mean squared error of f, both over standardized (obs, action).

    Discrete actions (``act_dim`` set) are one-hot encoded and not standardized.
    """

    def __init__(self, in_features, obs_dim, act_dim, hidden_dim, *, rngs):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.forward_model = MLP(in_features, hidden_dim, obs_dim, rngs=rngs)
        self.error_predictor = MLP(in_features, hidden_dim, 1, rngs=rngs)
        self.input_mean = nnx.Variable(jnp.zeros(in_features))
        self.input_std = nnx.Variable(jnp.ones(in_features))
        self.delta_mean = nnx.Variable(jnp.zeros(obs_dim))
        self.delta_std = nnx.Variable(jnp.ones(obs_dim))

    def _encode(self, obs, action):
        """obs (N, obs_dim), action (N, a_dim); discrete uses a_dim=1 index column."""
        if self.act_dim is not None:
            action = jax.nn.one_hot(action[:, 0], self.act_dim)
        return jnp.concatenate([obs, action], axis=-1)

    def _input(self, obs, action):
        x = (self._encode(obs, action) - self.input_mean) / self.input_std
        return jnp.clip(x, -5.0, 5.0)

    def __call__(self, obs, action):
        """Predicted error of a single transition, clipped at 0."""
        x = self._input(jnp.reshape(obs, (1, -1)), jnp.reshape(action, (1, -1)))
        return jnp.maximum(self.error_predictor(x)[0, 0], 0.0)

    def update_stats(self, obs, action, next_obs, count):
        mask = (jnp.arange(obs.shape[0]) < count)[:, None]
        x = self._encode(obs, action)
        self.input_mean[...] = jnp.mean(x, axis=0, where=mask)
        self.input_std[...] = jnp.std(x, axis=0, where=mask) + 1e-8
        if self.act_dim is not None:
            self.input_mean[self.obs_dim :] = 0.0
            self.input_std[self.obs_dim :] = 1.0
        delta = next_obs - obs
        self.delta_mean[...] = jnp.mean(delta, axis=0, where=mask)
        self.delta_std[...] = jnp.std(delta, axis=0, where=mask) + 1e-8


def make_prediction_error(
    config: PredictionErrorConfig, in_features, obs_dim, act_dim, rngs
):
    pe = PredictionError(in_features, obs_dim, act_dim, config.hidden_dim, rngs=rngs)
    optimizer = nnx.Optimizer(pe, optax.adam(config.lr), wrt=nnx.Param)
    return pe, optimizer


def make_train_prediction_error(config: PredictionErrorConfig):
    """Fit f and e on transitions sampled from the first ``count`` rows.

    Returns the per-step forward and error losses, each (update_steps,).
    """

    def train(pe, optimizer, obs, action, next_obs, count, rng):
        pe.update_stats(obs, action, next_obs, count)

        def loss_fn(pe, obs, action, next_obs):
            x = pe._input(obs, action)
            target = (next_obs - obs - pe.delta_mean) / pe.delta_std
            sq_err = jnp.mean(jnp.square(pe.forward_model(x) - target), axis=-1)
            forward_loss = jnp.mean(sq_err)
            error = pe.error_predictor(x)[:, 0]
            error_loss = jnp.mean(jnp.square(error - jax.lax.stop_gradient(sq_err)))
            return forward_loss + error_loss, (forward_loss, error_loss)

        def step(carry, key):
            pe, optimizer = carry
            idx = jax.random.randint(key, (config.batch_size,), 0, count)
            (_, losses), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
                pe, obs[idx], action[idx], next_obs[idx]
            )
            optimizer.update(pe, grads)
            return (pe, optimizer), losses

        keys = jax.random.split(rng, config.update_steps)
        _, (forward_losses, error_losses) = nnx.scan(step)((pe, optimizer), keys)
        return forward_losses, error_losses

    return train
