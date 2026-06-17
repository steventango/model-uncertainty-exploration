from typing import Literal

import jax
import jax.numpy as jnp
from flax import nnx


class WorldModel(nnx.Module):
    """Base class for learned dynamics models with shared normalization."""

    def __init__(
        self,
        in_features: int,
        obs_dim: int,
        act_dim: int | None = None,
        eps: float = 1e-8,
        predict_reward_terminated: bool = True,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.eps = eps
        self.predict_reward_terminated = predict_reward_terminated
        self.input_mean = nnx.Variable(jnp.zeros(in_features))
        self.input_std = nnx.Variable(jnp.ones(in_features))
        self.delta_obs_mean = nnx.Variable(jnp.zeros(obs_dim))
        self.delta_obs_std = nnx.Variable(jnp.ones(obs_dim))
        self.reward_mean = nnx.Variable(jnp.zeros(()))
        self.reward_std = nnx.Variable(jnp.ones(()))

    # --- Abstract per-model primitives ---

    def predict_sample(self, x, index):
        """Single input x, single index → sample output."""
        raise NotImplementedError

    def sample_index(self, key, num_samples: int):
        """Draw ``num_samples`` posterior indices from the prior."""
        raise NotImplementedError

    def predict_mean(self, x):
        """Deterministic posterior mean for a single normalized input."""
        raise NotImplementedError

    # --- Concrete uncertainty helpers ---

    def predict_samples(self, x, index):
        """x (in_features,), index (S, index_dim) → (S, out_features). Maps over S indices."""
        return jax.vmap(self.predict_sample, in_axes=(None, 0))(x, index)

    def batch_predict_sample(self, x, index):
        """x (N, in_features), index (index_dim,) → (N, out_features). Maps over N inputs with a single index."""
        return jax.vmap(self.predict_sample, in_axes=(0, None))(x, index)

    def batch_predict_samples(self, x, index):
        """x (N, in_features), index (S, index_dim) → (S, N, out_features). Maps over N inputs and S indices."""
        return jax.vmap(self.predict_samples, in_axes=(0, None), out_axes=1)(x, index)

    def uncertainty(
        self,
        samples,
        kind: Literal["std", "eig"] = "std",
        reduce_output: bool = True,
    ):
        """Scalar (or per-output) uncertainty from posterior samples (S, N, out_features) or (S, out_features).

        reduce_output=True  → mean over output dims (scalar per input point)
        reduce_output=False → per-output-dim uncertainty (shape N, out_features or out_features)
        """
        if kind == "eig":
            u = 0.5 * jnp.log(1.0 + samples.var(axis=0))
        else:
            u = samples.std(axis=0)
        if reduce_output:
            return u.mean(axis=-1)
        return u

    # --- Shared input/normalization helpers ---

    def encode_action(self, action):
        """action: (batch, a_dim); discrete uses a_dim=1 index column."""
        if self.act_dim is not None:
            return jax.nn.one_hot(action[:, 0], self.act_dim)
        return action

    def build_input(self, obs, action):
        """obs: (batch, obs_dim), action: (batch, a_dim)."""
        return jnp.concatenate([obs, self.encode_action(action)], axis=-1)

    def single_input(self, obs, action):
        obs = jnp.asarray(obs)
        if obs.ndim == 1:
            obs = obs[None]
        action = jnp.atleast_2d(jnp.asarray(action))
        x = self.normalize_input(self.build_input(obs, action))
        return jnp.reshape(x, (-1,))

    def update_stats(self, dataset, pointer):
        n_samples = dataset.obs.shape[0]
        mask = jnp.arange(n_samples) < pointer
        mask2d = mask[:, None]

        delta_obs = dataset.info["next_obs"] - dataset.obs

        self.input_mean[: self.obs_dim] = jnp.mean(dataset.obs, axis=0, where=mask2d)
        self.input_std[: self.obs_dim] = jnp.maximum(
            jnp.std(dataset.obs, axis=0, where=mask2d), self.eps
        )

        if self.act_dim is None:
            self.input_mean[self.obs_dim :] = jnp.mean(
                dataset.action, axis=0, where=mask2d
            )
            self.input_std[self.obs_dim :] = jnp.maximum(
                jnp.std(dataset.action, axis=0, where=mask2d), self.eps
            )
        else:
            self.input_mean[self.obs_dim :] = 0.0
            self.input_std[self.obs_dim :] = 1.0

        self.delta_obs_mean[...] = jnp.mean(delta_obs, axis=0, where=mask2d)
        self.delta_obs_std[...] = jnp.maximum(
            jnp.std(delta_obs, axis=0, where=mask2d), self.eps
        )
        self.reward_mean[...] = jnp.mean(dataset.reward, axis=0, where=mask)
        self.reward_std[...] = jnp.maximum(
            jnp.std(dataset.reward, axis=0, where=mask), self.eps
        )

    def normalize_input(self, x):
        return (x - self.input_mean) / self.input_std

    def normalize_delta_obs(self, delta):
        return (delta - self.delta_obs_mean) / self.delta_obs_std

    def normalize_reward(self, reward):
        return (reward - self.reward_mean) / self.reward_std

    def denormalize_delta_obs(self, delta_norm):
        return delta_norm * self.delta_obs_std + self.delta_obs_mean

    def denormalize_reward(self, reward_norm):
        return reward_norm * self.reward_std + self.reward_mean


# --- Registry and dispatching factories ---

_REGISTRY: dict[str, dict] = {}


def register_model(name: str):
    """Decorator to register a model's build/train factories under a string key."""

    def decorator(factories: dict):
        _REGISTRY[name] = factories
        return factories

    return decorator


def make_batched_model(model_type: str, *args, **kwargs):
    return _REGISTRY[model_type]["make_batched_model"](*args, **kwargs)


def make_batched_train_model(model_type: str, *args, **kwargs):
    return _REGISTRY[model_type]["make_batched_train_model"](*args, **kwargs)


def make_batched_rngs(keys):
    """Build a seed-batched nnx.Rngs: one independent stream per seed."""

    @nnx.vmap(in_axes=0, out_axes=0)
    def build(key):
        return nnx.Rngs(key)

    return build(keys)
