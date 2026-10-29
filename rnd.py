"""Random Network Distillation (Burda et al., 2018) exploration bonus."""

import jax
import jax.numpy as jnp
import optax
from flax import nnx


class MLP(nnx.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, *, rngs):
        self.dense1 = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.dense2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.dense3 = nnx.Linear(hidden_dim, out_dim, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.dense1(x))
        x = nnx.relu(self.dense2(x))
        return self.dense3(x)


class RND(nnx.Module):
    """Bonus = squared error of a trained predictor against a fixed random target.

    Observations are standardized with the replay-buffer statistics and clipped
    to [-5, 5] before both networks, as in the paper.
    """

    def __init__(self, obs_dim, hidden_dim, out_dim, *, rngs):
        self.target = MLP(obs_dim, hidden_dim, out_dim, rngs=rngs)
        self.predictor = MLP(obs_dim, hidden_dim, out_dim, rngs=rngs)
        self.obs_mean = nnx.Variable(jnp.zeros(obs_dim))
        self.obs_std = nnx.Variable(jnp.ones(obs_dim))

    def _features(self, obs):
        x = jnp.clip((obs - self.obs_mean) / self.obs_std, -5.0, 5.0)
        return self.predictor(x), jax.lax.stop_gradient(self.target(x))

    def __call__(self, obs):
        pred, target = self._features(obs)
        return jnp.mean(jnp.square(pred - target), axis=-1)

    def update_stats(self, obs, count):
        mask = (jnp.arange(obs.shape[0]) < count)[:, None]
        self.obs_mean[...] = jnp.mean(obs, axis=0, where=mask)
        self.obs_std[...] = jnp.std(obs, axis=0, where=mask) + 1e-8


def make_rnd(config, obs_dim, rngs):
    rnd = RND(obs_dim, config["HIDDEN_DIM"], config["OUT_DIM"], rngs=rngs)
    optimizer = nnx.Optimizer(rnd.predictor, optax.adam(config["LR"]), wrt=nnx.Param)
    return rnd, optimizer


def make_train_rnd(config):
    """Fit the predictor on next observations sampled from the first ``count`` rows."""

    def train(rnd, optimizer, next_obs, count, rng):
        rnd.update_stats(next_obs, count)

        def loss_fn(predictor, x):
            x = jnp.clip((x - rnd.obs_mean) / rnd.obs_std, -5.0, 5.0)
            target = jax.lax.stop_gradient(rnd.target(x))
            return jnp.mean(jnp.square(predictor(x) - target))

        def step(carry, key):
            rnd, optimizer = carry
            idx = jax.random.randint(key, (config["BATCH_SIZE"],), 0, count)
            loss, grads = nnx.value_and_grad(loss_fn)(rnd.predictor, next_obs[idx])
            optimizer.update(rnd.predictor, grads)
            return (rnd, optimizer), loss

        keys = jax.random.split(rng, config["UPDATE_STEPS"])
        _, losses = nnx.scan(step)((rnd, optimizer), keys)
        return losses

    return train
