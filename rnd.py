"""Random Network Distillation (Burda et al., 2018) exploration bonus."""

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from config import RNDConfig


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

    def __init__(self, obs_dim, hidden_dim, out_dim, lr, *, rngs):
        self.target = MLP(obs_dim, hidden_dim, out_dim, rngs=rngs)
        self.predictor = MLP(obs_dim, hidden_dim, out_dim, rngs=rngs)
        self.optimizer = nnx.Optimizer(self.predictor, optax.adam(lr), wrt=nnx.Param)
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


def make_rnd(config: RNDConfig, obs_dim, rngs):
    return RND(obs_dim, config.hidden_dim, config.out_dim, config.lr, rngs=rngs)


def make_train_rnd(config: RNDConfig):
    """Fit the predictor on next observations sampled from the first ``count`` rows."""

    def train(rnd, next_obs, count, rng):
        rnd.update_stats(next_obs, count)

        def loss_fn(predictor, x):
            x = jnp.clip((x - rnd.obs_mean) / rnd.obs_std, -5.0, 5.0)
            target = jax.lax.stop_gradient(rnd.target(x))
            return jnp.mean(jnp.square(predictor(x) - target))

        def step(rnd, key):
            idx = jax.random.randint(key, (config.batch_size,), 0, count)
            loss, grads = nnx.value_and_grad(loss_fn)(rnd.predictor, next_obs[idx])
            rnd.optimizer.update(rnd.predictor, grads)
            return rnd, loss

        keys = jax.random.split(rng, config.update_steps)
        _, losses = nnx.scan(step)(rnd, keys)
        return losses

    return train


class RNDBonus:
    """Model-env exploration bonus: RND novelty of the predicted next observation.

    The bonus state is a seed-batched RND, trained on the replay buffer's next
    observations each update and passed to the model env as ``params.bonus``.
    """

    def __init__(self, config: RNDConfig, obs_dim: int):
        self.config = config
        self.obs_dim = obs_dim
        self._train = nnx.jit(nnx.vmap(make_train_rnd(config), in_axes=(0, 0, None, 0)))

    def init(self, key: jax.Array) -> RND:
        @nnx.vmap
        def build(key):
            return make_rnd(self.config, self.obs_dim, nnx.Rngs(params=key))

        return build(key)

    def update(self, rnd: RND, experience, key: jax.Array) -> dict[str, jax.Array]:
        """Train in place; return per-seed metrics to log."""
        losses = self._train(
            rnd,
            experience.buffer.info["next_obs"],
            jnp.asarray(experience.size),
            key,
        )
        return {"rnd/loss": losses[:, -1]}
