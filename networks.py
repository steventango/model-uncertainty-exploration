from itertools import pairwise

import distrax
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.nnx.nn.initializers import orthogonal
from flax.nnx.nn.linear import default_kernel_init


def onehot_rows_init(n, num_blocks):
    def init(key, shape, dtype=jnp.float32):
        k1, k2 = jax.random.split(key)
        w = default_kernel_init(k2, shape, dtype)
        rows = jax.random.normal(k1, (n, shape[1]), dtype) / np.sqrt(num_blocks)
        return w.at[:n].set(rows)

    return init


class Block(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str,
        use_layer_norm: bool,
        kernel_init,
        *,
        rngs: nnx.Rngs,
    ):
        self.linear = nnx.Linear(
            in_features, out_features, kernel_init=kernel_init, rngs=rngs
        )
        self.norm = nnx.LayerNorm(out_features, rngs=rngs) if use_layer_norm else None
        self.activation = getattr(nnx, activation)

    def __call__(self, x):
        x = self.linear(x)
        if self.norm is not None:
            x = self.norm(x)
        return self.activation(x)


class MLP(nnx.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        rngs: nnx.Rngs,
        num_layers: int = 1,
        activation: str = "tanh",
        use_layer_norm: bool = False,
        kernel_init=default_kernel_init,
        out_kernel_init=default_kernel_init,
        first_kernel_init=None,
    ):
        sizes = [in_features] + [hidden_features] * num_layers
        inits = [first_kernel_init or kernel_init] + [kernel_init] * (num_layers - 1)
        self.blocks = nnx.List(
            [
                Block(i, o, activation, use_layer_norm, init, rngs=rngs)
                for (i, o), init in zip(pairwise(sizes), inits)
            ]
        )
        self.out = nnx.Linear(
            hidden_features, out_features, kernel_init=out_kernel_init, rngs=rngs
        )

    def __call__(self, x, rngs: nnx.Rngs | None = None):
        for block in self.blocks:
            x = block(x)
        return self.out(x), x


class Actor(nnx.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        activation: str = "tanh",
        discrete: bool = False,
        use_layer_norm: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.discrete = discrete
        self.mlp = MLP(
            state_dim,
            hidden_dim,
            action_dim,
            rngs=rngs,
            num_layers=2,
            activation=activation,
            use_layer_norm=use_layer_norm,
            kernel_init=orthogonal(np.sqrt(2)),
            out_kernel_init=orthogonal(0.01),
        )
        if not discrete:
            self.log_std = nnx.Param(jnp.zeros(action_dim))

    def __call__(self, x: jax.Array):
        actor_mean, _ = self.mlp(x)
        if self.discrete:
            return distrax.Categorical(logits=actor_mean)
        return distrax.MultivariateNormalDiag(
            actor_mean, jnp.exp(self.log_std.get_value())
        )


class Critic(nnx.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        activation: str = "tanh",
        use_layer_norm: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.mlp = MLP(
            state_dim,
            hidden_dim,
            1,
            rngs=rngs,
            num_layers=2,
            activation=activation,
            use_layer_norm=use_layer_norm,
            kernel_init=orthogonal(np.sqrt(2)),
            out_kernel_init=orthogonal(1.0),
        )

    def __call__(self, x: jax.Array):
        critic, _ = self.mlp(x)
        return jnp.squeeze(critic, axis=-1)


class ActorCritic(nnx.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        activation: str = "tanh",
        discrete: bool = False,
        use_layer_norm: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.actor = Actor(
            state_dim,
            action_dim,
            hidden_dim,
            activation,
            discrete,
            use_layer_norm,
            rngs=rngs,
        )
        self.critic = Critic(
            state_dim, hidden_dim, activation, use_layer_norm, rngs=rngs
        )

    def __call__(self, x):
        pi = self.actor(x)
        critic = self.critic(x)
        return pi, critic


class ProjectedMLP(nnx.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        rngs: nnx.Rngs,
        num_layers: int = 1,
        activation: str = "tanh",
        first_kernel_init=None,
    ):
        self.mlp = MLP(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            rngs=rngs,
            out_kernel_init=nnx.initializers.zeros,
            first_kernel_init=first_kernel_init,
            use_layer_norm=False,
            num_layers=num_layers,
            activation=activation,
        )

    def __call__(self, x, z, rngs: nnx.Rngs | None = None):
        xz = jnp.concatenate([x, z], axis=-1)
        y, _ = self.mlp(xz, rngs=rngs)
        y1 = y.reshape(-1, z.shape[-1])
        return (y1 * z).sum(axis=-1)


class MLPEnsemble(nnx.Module):
    def __init__(
        self,
        num_models: int,
        in_features: int,
        hidden_features: int,
        out_features: int,
        rngs: nnx.Rngs,
        num_layers: int = 1,
        activation: str = "tanh",
        first_kernel_init=None,
    ):
        keys = jax.random.split(rngs.params(), num_models)

        @nnx.vmap
        def create_model(key):
            return MLP(
                in_features,
                hidden_features,
                out_features,
                use_layer_norm=False,
                num_layers=num_layers,
                activation=activation,
                first_kernel_init=first_kernel_init,
                rngs=nnx.Rngs(params=key),
            )

        self.models = create_model(keys)

    def __call__(self, x, z, rngs: nnx.Rngs | None = None):
        graphdef, states = nnx.split(self.models)

        def forward(state, inputs):
            model = nnx.merge(graphdef, state)
            y, _ = model(inputs, rngs=rngs)
            return y

        y = jax.vmap(forward, in_axes=(0, None))(states, x)
        return jnp.einsum("no,n->o", y, z)


class EpiNet(nnx.Module):
    def __init__(
        self,
        in_features: int,
        learnable_hidden_features: int,
        prior_hidden_features: int,
        base_features: int,
        out_features: int,
        index_dim: int,
        *,
        num_layers: int = 1,
        activation: str = "tanh",
        prior_scale: float = 1.0,
        first_kernel_init=None,
        rngs: nnx.Rngs,
    ):
        self.learnable = ProjectedMLP(
            in_features=in_features + base_features + index_dim,
            hidden_features=learnable_hidden_features,
            out_features=out_features * index_dim,
            rngs=rngs,
            num_layers=num_layers,
            activation=activation,
            first_kernel_init=first_kernel_init,
        )
        self.prior = MLPEnsemble(
            num_models=index_dim,
            in_features=in_features,
            hidden_features=prior_hidden_features,
            out_features=out_features,
            rngs=rngs,
            num_layers=num_layers,
            activation=activation,
            first_kernel_init=first_kernel_init,
        )
        self.prior_scale = prior_scale

    def __call__(self, phi, x, z, rngs: nnx.Rngs | None = None):
        return self.learnable(
            phi, z, rngs=rngs
        ) + self.prior_scale * jax.lax.stop_gradient(self.prior(x, z, rngs=rngs))


class Frozen(nnx.Variable):
    pass


class OneHotEmbed(nnx.Module):
    def __init__(self, blocks, mode, dim, trainable, *, rngs: nnx.Rngs):
        self.blocks = tuple(blocks)
        self.mode = mode
        self.dims = self.blocks
        if mode == "random":
            self.dims = (dim or self.blocks[0],) + self.blocks[1:]
            table = nnx.Param if trainable else Frozen
            self.tables = nnx.List(
                [
                    table(jax.random.normal(rngs.params(), (k, d)))
                    for k, d in zip(self.blocks, self.dims)
                ]
            )

    @property
    def out_features(self):
        return sum(self.dims)

    def __call__(self, x):
        if self.mode == "identity":
            return x
        parts = jnp.split(x, np.cumsum(self.blocks)[:-1], axis=-1)
        parts = [p @ t[...] for p, t in zip(parts, self.tables)]
        return jnp.concatenate(parts, axis=-1)


class ENN(nnx.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        learnable_hidden_features: int,
        prior_hidden_features: int,
        out_features: int,
        index_dim: int,
        *,
        use_layer_norm: bool = False,
        num_layers: int = 1,
        activation: str = "tanh",
        epinet_num_layers: int = 1,
        epinet_activation: str = "tanh",
        prior_scale: float = 1.0,
        embed_blocks: tuple[int, ...] | None = None,
        embed: str = "identity",
        embed_dim: int | None = None,
        embed_trainable: bool = False,
        onehot_init: str = "lecun",
        rngs: nnx.Rngs,
    ):
        self.embed = OneHotEmbed(
            embed_blocks or (in_features,),
            embed if embed_blocks else "identity",
            embed_dim,
            embed_trainable,
            rngs=rngs,
        )
        in_features = self.embed.out_features
        first_kernel_init = None
        if embed_blocks and self.embed.mode == "identity" and onehot_init == "normal":
            first_kernel_init = onehot_rows_init(in_features, len(embed_blocks))
        self.base = MLP(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            use_layer_norm=use_layer_norm,
            num_layers=num_layers,
            activation=activation,
            first_kernel_init=first_kernel_init,
            rngs=rngs,
        )
        self.epinet = EpiNet(
            in_features=in_features,
            learnable_hidden_features=learnable_hidden_features,
            prior_hidden_features=prior_hidden_features,
            base_features=hidden_features,
            out_features=out_features,
            index_dim=index_dim,
            num_layers=epinet_num_layers,
            activation=epinet_activation,
            prior_scale=prior_scale,
            first_kernel_init=first_kernel_init,
            rngs=rngs,
        )
        self.index_dim = index_dim

    def base_out(self, x, rngs: nnx.Rngs | None = None):
        return self.base(self.embed(x), rngs=rngs)

    def __call__(self, x, z, rngs: nnx.Rngs | None = None):
        x = self.embed(x)
        y, features = self.base(x, rngs=rngs)
        phi = jnp.concatenate([x, features], axis=-1)
        return y, y + self.epinet(jax.lax.stop_gradient(phi), x, z, rngs=rngs)
