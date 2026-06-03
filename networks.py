import distrax
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.nnx.nn.initializers import constant, orthogonal


class Actor(nnx.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        activation: str = "tanh",
        *,
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        if activation == "relu":
            self.activation = nnx.relu
        else:
            self.activation = nnx.tanh
        self.dense1 = nnx.Linear(
            state_dim,
            256,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.dense2 = nnx.Linear(
            256,
            256,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.dense3 = nnx.Linear(
            256,
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.log_std = nnx.Param(jnp.zeros(self.action_dim))

    def __call__(self, x: jax.Array):
        actor_mean = self.dense1(x)
        actor_mean = self.activation(actor_mean)
        actor_mean = self.dense2(actor_mean)
        actor_mean = self.activation(actor_mean)
        actor_mean = self.dense3(actor_mean)
        pi = distrax.MultivariateNormalDiag(
            actor_mean, jnp.exp(self.log_std.get_value())
        )
        return pi


class Critic(nnx.Module):
    def __init__(self, state_dim: int, activation: str = "tanh", *, rngs: nnx.Rngs):
        if activation == "relu":
            self.activation = nnx.relu
        else:
            self.activation = nnx.tanh
        self.dense1 = nnx.Linear(
            state_dim,
            256,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.dense2 = nnx.Linear(
            256,
            256,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            rngs=rngs,
        )
        self.dense3 = nnx.Linear(
            256,
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array):
        critic = self.dense1(x)
        critic = self.activation(critic)
        critic = self.dense2(critic)
        critic = self.activation(critic)
        critic = self.dense3(critic)
        return jnp.squeeze(critic, axis=-1)


class ActorCritic(nnx.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        activation: str = "tanh",
        *,
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        self.activation = activation
        self.actor = Actor(state_dim, action_dim, activation, rngs=rngs)
        self.critic = Critic(state_dim, activation, rngs=rngs)

    def __call__(self, x):
        pi = self.actor(x)
        critic = self.critic(x)
        return pi, critic
