from typing import Literal

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax.scipy.stats import norm

from models.base import WorldModel, register_model
from networks import ENN


class ENNModel(WorldModel):
    """ENN-based dynamics model. Epistemic uncertainty via index-variable sampling."""

    def __init__(
        self,
        enn: ENN,
        in_features: int,
        obs_dim: int,
        act_dim: int | None = None,
        eps: float = 1e-8,
        predict_reward_terminated: bool = True,
        bootstrap: Literal["none", "bernoulli", "gaussian"] = "none",
        bootstrap_noise: float = 1.0,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(
            in_features=in_features,
            obs_dim=obs_dim,
            act_dim=act_dim,
            eps=eps,
            predict_reward_terminated=predict_reward_terminated,
        )
        self.enn = enn
        self.bootstrap = bootstrap
        self.bootstrap_noise = bootstrap_noise
        self.bootstrap_key = nnx.Variable(rngs.bootstrap())

    @property
    def index_dim(self):
        return self.enn.index_dim

    def predict_sample(self, x, index):
        """Single input x, single index vector → sample output."""
        return self.enn(x, index)[1]

    def sample_index(self, key, num_samples: int):
        return jax.random.normal(key, (num_samples, self.index_dim))

    def predict_mean(self, x):
        """Deterministic base-network output for a single normalized input."""
        y, _ = self.enn.base(x)
        return y

    def bootstrap_c(self, indices, num: int):
        key = self.bootstrap_key[...]
        c = jax.vmap(
            lambda i: jax.random.normal(
                jax.random.fold_in(key, i), (num, self.index_dim)
            )
        )(indices)
        return c / jnp.linalg.norm(c, axis=-1, keepdims=True)


def _bootstrap(model: ENNModel, indices, z, num_outputs: int, p: float = 0.5):
    n = indices.shape[0]
    keep = jnp.ones(n, dtype=bool)
    noise = jnp.zeros((n, num_outputs))
    if model.bootstrap == "bernoulli":
        cz = model.bootstrap_c(indices, 1)[:, 0] @ z
        keep = cz > norm.ppf(p)
    elif model.bootstrap == "gaussian":
        noise = model.bootstrap_noise * (model.bootstrap_c(indices, num_outputs) @ z)
    return keep, noise


def _loss_fn(model: ENNModel, batch, indices, rngs: nnx.Rngs):
    x = model.build_input(batch.obs, batch.action)
    x = model.normalize_input(x)
    z = jax.random.normal(rngs(), shape=(model.index_dim,))
    logits = model.batch_predict_sample(x, z)
    num_outputs = model.obs_dim + int(model.predict_reward_terminated)
    keep, noise = _bootstrap(model, indices, z, num_outputs)

    def masked_mean(loss):
        return jnp.where(keep.any(), jnp.mean(loss, where=keep), 0.0)

    delta_next_state = batch.info["next_obs"] - batch.obs
    delta_next_state = model.normalize_delta_obs(delta_next_state)
    delta_next_state_target = delta_next_state + noise[:, : model.obs_dim]
    delta_next_state_loss = (
        logits[..., : model.obs_dim] - delta_next_state_target
    ) ** 2
    delta_next_state_loss = masked_mean(delta_next_state_loss.mean(-1))

    if model.predict_reward_terminated:
        reward_target = model.normalize_reward(batch.reward) + noise[:, model.obs_dim]
        reward_loss = (logits[..., -2] - reward_target) ** 2
        reward_loss = masked_mean(reward_loss)

        terminated_target = batch.terminated.astype(jnp.float32)
        terminated_pred = logits[..., -1]
        terminated_loss = optax.sigmoid_binary_cross_entropy(
            terminated_pred, terminated_target
        )
        terminated_loss = masked_mean(terminated_loss)
    else:
        reward_loss = jnp.zeros(())
        terminated_loss = jnp.zeros(())

    loss = delta_next_state_loss + reward_loss + terminated_loss
    return loss, (delta_next_state_loss, reward_loss, terminated_loss)


@nnx.jit
def _train_step(
    model: ENNModel,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    rngs: nnx.Rngs,
    batch,
    indices,
):
    grad_fn = nnx.value_and_grad(_loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(model, batch, indices, rngs)
    delta_next_state_loss, reward_loss, terminated_loss = aux
    metrics.update(
        loss=loss,
        delta_next_state_loss=delta_next_state_loss,
        reward_loss=reward_loss,
        terminated_loss=terminated_loss,
    )
    optimizer.update(model, grads)


@nnx.jit
def _eval_step(
    model: ENNModel,
    metrics: nnx.MultiMetric,
    rngs: nnx.Rngs,
    batch,
    indices,
):
    loss, aux = _loss_fn(model, batch, indices, rngs)
    delta_next_state_loss, reward_loss, terminated_loss = aux
    metrics.update(
        loss=loss,
        delta_next_state_loss=delta_next_state_loss,
        reward_loss=reward_loss,
        terminated_loss=terminated_loss,
    )


def _train_model(
    model: ENNModel,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    dataset,
    update_steps: int,
    pointer: int,
    minibatch_size: int,
    rngs: nnx.Rngs,
):
    model.update_stats(dataset, pointer)

    def train_step_fn(train_state, _):
        model, optimizer, metrics, rngs = train_state
        indices = jax.random.randint(rngs(), (minibatch_size,), 0, pointer)
        minibatch = jax.tree_util.tree_map(
            lambda x: jnp.take(x, indices, axis=0), dataset
        )
        metrics.reset()
        _train_step(model, optimizer, metrics, rngs, minibatch, indices)
        return (model, optimizer, metrics, rngs), metrics.compute()

    train_state = (model, optimizer, metrics, rngs)
    _, history = nnx.scan(train_step_fn, length=update_steps)(train_state, None)
    return history


def _make_batched_model(
    model_config,
    in_features,
    obs_dim,
    out_features,
    act_dim,
    keys,
    predict_reward_terminated: bool = True,
):
    @nnx.vmap
    def build(key):
        rngs = nnx.Rngs(key)
        enn = ENN(
            in_features,
            model_config["HIDDEN_DIM"],
            model_config["LEARNABLE_HIDDEN_DIM"],
            model_config["PRIOR_HIDDEN_DIM"],
            out_features,
            model_config["INDEX_DIM"],
            use_layer_norm=model_config.get("USE_LAYER_NORM", False),
            num_layers=model_config["NUM_LAYERS"],
            activation=model_config["ACTIVATION"],
            rngs=rngs,
        )
        model = ENNModel(
            enn,
            in_features,
            obs_dim,
            act_dim=act_dim,
            predict_reward_terminated=predict_reward_terminated,
            bootstrap=model_config["BOOTSTRAP"],
            bootstrap_noise=model_config["BOOTSTRAP_NOISE"],
            rngs=rngs,
        )
        tx = optax.adamw(model_config["LR"], weight_decay=1e-4)
        not_prior_params = nnx.All(nnx.Param, nnx.Not(nnx.PathContains("prior")))
        optimizer = nnx.Optimizer(model, tx, wrt=not_prior_params)
        metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average("loss"),
            delta_next_state_loss=nnx.metrics.Average("delta_next_state_loss"),
            reward_loss=nnx.metrics.Average("reward_loss"),
            terminated_loss=nnx.metrics.Average("terminated_loss"),
        )
        return model, optimizer, metrics

    model, optimizer, metrics = build(keys)
    return model, (optimizer, metrics)


def _make_batched_train_model(update_steps, minibatch_size):
    def core(model, optimizer, metrics, dataset, pointer, rngs):
        return _train_model(
            model,
            optimizer,
            metrics,
            dataset,
            update_steps,
            pointer,
            minibatch_size,
            rngs,
        )

    batched = nnx.jit(nnx.vmap(core, in_axes=(0, 0, 0, 0, None, 0), out_axes=0))

    def train_fn(model, train_state, dataset, pointer, rngs):
        optimizer, metrics = train_state
        return batched(model, optimizer, metrics, dataset, pointer, rngs)

    return train_fn


register_model("enn")(
    {
        "make_batched_model": _make_batched_model,
        "make_batched_train_model": _make_batched_train_model,
    }
)
