from typing import Protocol

import jax
import jax.numpy as jnp
from flax import nnx


class Features(Protocol):
    num_features: int

    def update(self, X: jax.Array, pointer) -> None: ...
    def __call__(self, x: jax.Array) -> jax.Array: ...


class RBFFeatures(nnx.Module):
    """Lazy RBF features: centers are set to training inputs at fit time."""

    def __init__(self, num_features: int, in_features: int, length_scale: float):
        self.num_features = num_features
        self.length_scale = nnx.Variable(jnp.asarray(length_scale, dtype=jnp.float32))
        self.C = nnx.Variable(jnp.zeros((num_features, in_features), dtype=jnp.float32))
        self.n_valid = nnx.Variable(jnp.zeros((), dtype=jnp.int32))

    def update(self, X: jax.Array, pointer) -> None:
        self.C.value = X
        self.n_valid.value = pointer.astype(jnp.int32)

    def __call__(self, x: jax.Array) -> jax.Array:
        diff = jnp.expand_dims(x, axis=-2) - self.C.value
        sq_dist = jnp.sum(diff**2, axis=-1)
        phi = jnp.exp(-sq_dist / (2 * self.length_scale.value**2))
        valid = jnp.arange(self.num_features) < self.n_valid.value
        rbf = phi * valid
        bias = jnp.ones(x.shape[:-1] + (1,))
        return jnp.concatenate([bias, rbf], axis=-1)


class RFFFeatures(nnx.Module):
    """Fixed random Fourier features."""

    def __init__(
        self, key: jax.Array, in_features: int, num_features: int, length_scale: float
    ):
        self.num_features = num_features
        key_w, key_b = jax.random.split(key)
        W = jax.random.normal(key_w, (num_features, in_features)) / length_scale
        b = jax.random.uniform(key_b, (num_features,), minval=0.0, maxval=2.0 * jnp.pi)
        self.W = nnx.Variable(W)
        self.b = nnx.Variable(b)

    def update(self, X: jax.Array, pointer) -> None:  # noqa: ARG002
        pass

    def __call__(self, x: jax.Array) -> jax.Array:
        y = x @ self.W.value.T + self.b.value
        phi = jnp.sqrt(2.0 / self.num_features) * jnp.cos(y)
        bias = jnp.ones(x.shape[:-1] + (1,))
        return jnp.concatenate([bias, phi], axis=-1)
