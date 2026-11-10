from itertools import pairwise

import distrax
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.nnx.nn.initializers import orthogonal
from flax.nnx.nn.linear import default_kernel_init


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
    ):
        sizes = [in_features] + [hidden_features] * num_layers
        self.blocks = nnx.List(
            [
                Block(i, o, activation, use_layer_norm, kernel_init, rngs=rngs)
                for i, o in pairwise(sizes)
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
        z_input: bool = True,
    ):
        self.z_input = z_input
        self.mlp = MLP(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            rngs=rngs,
            out_kernel_init=nnx.initializers.zeros,
            use_layer_norm=False,
            num_layers=num_layers,
            activation=activation,
        )

    def __call__(self, x, z, rngs: nnx.Rngs | None = None):
        if self.z_input:
            x = jnp.concatenate([x, z], axis=-1)
        y, _ = self.mlp(x, rngs=rngs)
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


class RFFEnsemble(nnx.Module):
    def __init__(
        self,
        num_models: int,
        in_features: int,
        out_features: int,
        rngs: nnx.Rngs,
        num_features: int = 64,
        lengthscale: float = 1.0,
    ):
        k1, k2, k3 = jax.random.split(rngs.params(), 3)
        shape = (num_models, in_features, num_features)
        self.freq = nnx.Param(jax.random.normal(k1, shape) / lengthscale)
        self.phase = nnx.Param(
            jax.random.uniform(k2, (num_models, num_features), maxval=2 * jnp.pi)
        )
        self.weight = nnx.Param(
            jax.random.normal(k3, (num_models, num_features, out_features))
            / jnp.sqrt(num_models)
        )
        self.num_features = num_features

    def __call__(self, x, z, rngs: nnx.Rngs | None = None):
        features = jnp.cos(jnp.einsum("i,nif->nf", x, self.freq[...]) + self.phase[...])
        features = features * jnp.sqrt(2.0 / self.num_features)
        y = jnp.einsum("nf,nfo->no", features, self.weight[...])
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
        z_input: bool = True,
        prior: str = "mlp",
        rff_lengthscale: float = 1.0,
        rngs: nnx.Rngs,
    ):
        self.learnable = ProjectedMLP(
            in_features=in_features + base_features + index_dim * z_input,
            hidden_features=learnable_hidden_features,
            out_features=out_features * index_dim,
            rngs=rngs,
            num_layers=num_layers,
            activation=activation,
            z_input=z_input,
        )
        if prior == "rff":
            self.prior = RFFEnsemble(
                num_models=index_dim,
                in_features=in_features,
                out_features=out_features,
                rngs=rngs,
                lengthscale=rff_lengthscale,
            )
        else:
            self.prior = MLPEnsemble(
                num_models=index_dim,
                in_features=in_features,
                hidden_features=prior_hidden_features,
                out_features=out_features,
                rngs=rngs,
                num_layers=num_layers,
                activation=activation,
            )
        self.prior_scale = 1.0

    def __call__(self, phi, x, z, rngs: nnx.Rngs | None = None):
        return self.learnable(
            phi, z, rngs=rngs
        ) + self.prior_scale * jax.lax.stop_gradient(self.prior(x, z, rngs=rngs))


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
        epinet_z_input: bool = True,
        epinet_prior: str = "mlp",
        epinet_rff_lengthscale: float = 1.0,
        rngs: nnx.Rngs,
    ):
        self.base = MLP(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            use_layer_norm=use_layer_norm,
            num_layers=num_layers,
            activation=activation,
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
            z_input=epinet_z_input,
            prior=epinet_prior,
            rff_lengthscale=epinet_rff_lengthscale,
            rngs=rngs,
        )
        self.index_dim = index_dim

    def __call__(self, x, z, rngs: nnx.Rngs | None = None):
        y, features = self.base(x, rngs=rngs)
        phi = jnp.concatenate([x, features], axis=-1)
        return y, y + self.epinet(jax.lax.stop_gradient(phi), x, z, rngs=rngs)
