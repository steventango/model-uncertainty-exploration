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
        bootstrap_key: jax.Array,
        in_features: int,
        obs_dim: int,
        act_dim: int | None = None,
        eps: float = 1e-8,
        predict_reward_terminated: bool = True,
        bootstrap: Literal["bernoulli", "gaussian"] = "bernoulli",
        bootstrap_noise: float = 1.0,
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
        self.bootstrap_key = nnx.Variable(jax.random.key_data(bootstrap_key))

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

    def bootstrap_c(self, indices):
        """Fixed unit vectors c_i of dataset rows for the delta, reward and terminated heads.  (3, N, index_dim)"""
        key = jax.random.wrap_key_data(self.bootstrap_key[...])
        c = jax.vmap(
            lambda i: jax.random.normal(
                jax.random.fold_in(key, i), (3, self.index_dim)
            ),
            out_axes=1,
        )(indices)
        return c / jnp.linalg.norm(c, axis=-1, keepdims=True)


def _loss_fn(model: ENNModel, batch, indices, rngs: nnx.Rngs):
    x = model.build_input(batch.obs, batch.action)
    x = model.normalize_input(x)
    z = jax.random.normal(rngs(), shape=(model.index_dim,))
    logits = model.batch_predict_sample(x, z)
    cz = model.bootstrap_c(indices) @ z
    p = 0.5
    mask = (cz > norm.ppf(p)).astype(jnp.float32)
    if model.bootstrap == "bernoulli":
        noise = jnp.zeros_like(cz)
        weight = mask
    else:
        noise = model.bootstrap_noise * cz
        weight = jnp.ones_like(cz)

    def weighted_mean(loss, w):
        return (loss * w).sum() / jnp.maximum(w.sum(), 1.0)

    delta_next_state = batch.info["next_obs"] - batch.obs
    delta_next_state = model.normalize_delta_obs(delta_next_state)
    delta_next_state_target = delta_next_state + noise[0, :, None]
    delta_next_state_loss = (
        logits[..., : model.obs_dim] - delta_next_state_target
    ) ** 2
    delta_next_state_loss = weighted_mean(delta_next_state_loss.mean(-1), weight[0])

    if model.predict_reward_terminated:
        reward_target = model.normalize_reward(batch.reward) + noise[1]
        reward_loss = (logits[..., -2] - reward_target) ** 2
        reward_loss = weighted_mean(reward_loss, weight[1])

        terminated_target = batch.terminated.astype(jnp.float32)
        terminated_pred = logits[..., -1]
        terminated_loss = optax.sigmoid_binary_cross_entropy(
            terminated_pred, terminated_target
        )
        terminated_loss = weighted_mean(terminated_loss, mask[2])
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
        enn = ENN(
            in_features,
            model_config["HIDDEN_DIM"],
            model_config["LEARNABLE_HIDDEN_DIM"],
            model_config["PRIOR_HIDDEN_DIM"],
            out_features,
            model_config["INDEX_DIM"],
            use_layer_norm=model_config.get("USE_LAYER_NORM", False),
            rngs=nnx.Rngs(params=key),
        )
        model = ENNModel(
            enn,
            jax.random.fold_in(key, 1),
            in_features,
            obs_dim,
            act_dim=act_dim,
            predict_reward_terminated=predict_reward_terminated,
            bootstrap=model_config["BOOTSTRAP"],
            bootstrap_noise=model_config["BOOTSTRAP_NOISE"],
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
