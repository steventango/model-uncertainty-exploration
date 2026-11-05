from functools import partial
from typing import Literal

import distrax
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax.scipy.stats import norm

from models.base import WorldModel, head_out_features, register_model
from networks import ENN


class ENNModel(WorldModel):
    """ENN-based dynamics model. Epistemic uncertainty via index-variable sampling.

    ``head`` selects the output:
      deterministic: point estimate of the normalized delta-obs (+ reward/terminated).
      gaussian: diagonal Gaussian over the normalized delta-obs (aleatoric noise).
    For the distributional head each epistemic index z yields a full predictive
    distribution; exploration uses only the disagreement across z (epistemic).
    """

    def __init__(
        self,
        enn: ENN,
        bootstrap_key: jax.Array,
        in_features: int,
        obs_dim: int,
        act_dim: int | None = None,
        eps: float = 1e-8,
        predict_reward_terminated: bool = True,
        head: str = "deterministic",
        min_scale: float = 1e-3,
        beta_nll: float = 0.5,
        eig_noise_floor: float = 0.0,
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
        head_out_features(head, obs_dim, predict_reward_terminated)  # validates
        self.head = head
        self.min_scale = min_scale
        self.beta_nll = beta_nll
        self.eig_noise_floor = eig_noise_floor
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
        """Expected normalized delta-obs from the base network for a single input.

        gaussian: the Gaussian loc.
        """
        y, _ = self.enn.base(x)
        if self.head == "gaussian":
            return y[..., : self.obs_dim]
        return y

    # --- Distributional head ---

    def _loc_scale(self, y):
        loc, raw_scale = y[..., : self.obs_dim], y[..., self.obs_dim :]
        return loc, jax.nn.softplus(raw_scale) + self.min_scale

    def dist_from_output(self, y) -> distrax.Distribution:
        """Raw network output → distrax predictive distribution."""
        if self.head == "gaussian":
            return distrax.MultivariateNormalDiag(*self._loc_scale(y))
        raise ValueError("deterministic head has no predictive distribution")

    def sample_delta(self, x, index, key):
        if self.head == "deterministic":
            return super().sample_delta(x, index, key)
        return self.dist_from_output(self.predict_sample(x, index)).sample(seed=key)

    def _targets(self, obs, next_obs):
        """Distribution-space targets: the normalized delta-obs."""
        return self.normalize_delta_obs(next_obs - obs)

    def nll(self, x, obs, next_obs):
        if self.head == "deterministic":
            return super().nll(x, obs, next_obs)
        y, _ = self.enn.base(x)
        return -self.dist_from_output(y).log_prob(self._targets(obs, next_obs))

    def uncertainty(self, x, z, kind="std", reduce_output=True):
        """Epistemic-only uncertainty at a single x over S indices z.

        Aleatoric noise is excluded: it is the spread *within* each index's
        predictive distribution, while these measure disagreement *across* z.
          eig: ½ log(1 + Var_z[μ] / E_z[σ²]) per obs dim (Gaussian information
            gain with the predicted aleatoric noise).
          std: sqrt(Var_z) of the mean μ.
        """
        if self.head == "deterministic":
            return super().uncertainty(x, z, kind, reduce_output)
        loc, scale = self._loc_scale(self.predict_samples(x, z))  # (S, obs_dim)
        if kind == "eig":
            noise_var = (scale**2).mean(axis=0) + self.eig_noise_floor**2
            u = 0.5 * jnp.log1p(loc.var(axis=0) / noise_var)
        else:
            u = loc.std(axis=0)
        return u.sum(axis=-1) if reduce_output else u

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


TRAINABLE = nnx.All(nnx.Param, nnx.Not(nnx.PathContains("prior")))


def _loss_fn(
    model: ENNModel,
    batch,
    indices,
    rngs: nnx.Rngs,
    num_index_samples: int = 1,
    l2: float = 0.0,
    num_data=1,
):
    x = model.build_input(batch.obs, batch.action)
    x = model.normalize_input(x)
    c = model.bootstrap_c(indices)
    zs = jax.random.normal(rngs(), shape=(num_index_samples, model.index_dim))
    losses = [_index_loss(model, batch, x, c, z) for z in zs]
    loss, aux = jax.tree.map(lambda *v: jnp.mean(jnp.stack(v)), *losses)
    if l2 > 0:
        sq = sum(jnp.sum(p**2) for p in jax.tree.leaves(nnx.state(model, TRAINABLE)))
        loss = loss + l2 * sq / num_data
    return loss, aux


def _index_loss(model: ENNModel, batch, x, c, z):
    logits = model.batch_predict_sample(x, z)
    cz = c @ z
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

    if model.head == "gaussian":
        nll = _beta_nll(model, logits, delta_next_state_target)
        nll = weighted_mean(nll, weight[0])
        return nll, (nll, jnp.zeros(()), jnp.zeros(()))

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


def _beta_nll(model: ENNModel, y, target):
    """Per-point Gaussian NLL, each dim weighted by sg(σ^2β) (Seitzer et al. 2022).

    β=0 is the NLL; β=1 gives the mean the MSE gradient. The per-point optimum
    (μ, σ) is unchanged; only the weighting across points differs.
    """
    loc, scale = model._loc_scale(y)
    nll = 0.5 * ((target - loc) / scale) ** 2 + jnp.log(scale)
    weight = jax.lax.stop_gradient(scale ** (2 * model.beta_nll))
    return (weight * nll).sum(-1)


@partial(nnx.jit, static_argnames=("num_index_samples", "l2"))
def _train_step(
    model: ENNModel,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    rngs: nnx.Rngs,
    batch,
    indices,
    num_index_samples: int = 1,
    l2: float = 0.0,
    num_data=1,
):
    grad_fn = nnx.value_and_grad(_loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(
        model, batch, indices, rngs, num_index_samples, l2, num_data
    )
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
    num_index_samples: int = 1,
    l2: float = 0.0,
):
    model.update_stats(dataset, pointer)

    def train_step_fn(train_state, _):
        model, optimizer, metrics, rngs = train_state
        indices = jax.random.randint(rngs(), (minibatch_size,), 0, pointer)
        minibatch = jax.tree_util.tree_map(
            lambda x: jnp.take(x, indices, axis=0), dataset
        )
        metrics.reset()
        _train_step(
            model,
            optimizer,
            metrics,
            rngs,
            minibatch,
            indices,
            num_index_samples=num_index_samples,
            l2=l2,
            num_data=pointer,
        )
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
    head: str = "deterministic",
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
            num_layers=model_config["NUM_LAYERS"],
            activation=model_config["ACTIVATION"],
            index_input=model_config["INDEX_INPUT"],
            prior=model_config["PRIOR"],
            prior_lengthscale=model_config["PRIOR_LENGTHSCALE"],
        )
        model = ENNModel(
            enn,
            jax.random.fold_in(key, 1),
            in_features,
            obs_dim,
            act_dim=act_dim,
            predict_reward_terminated=predict_reward_terminated,
            head=head,
            min_scale=model_config.get("MIN_SCALE", 1e-3),
            beta_nll=model_config.get("BETA_NLL", 0.5),
            eig_noise_floor=model_config.get("EIG_NOISE_FLOOR", 0.0),
            bootstrap=model_config["BOOTSTRAP"],
            bootstrap_noise=model_config["BOOTSTRAP_NOISE"],
        )
        tx = optax.adamw(model_config["LR"], weight_decay=1e-4)
        if model_config["GRAD_CLIP"] > 0:
            tx = optax.chain(optax.clip_by_global_norm(model_config["GRAD_CLIP"]), tx)
        optimizer = nnx.Optimizer(model, tx, wrt=TRAINABLE)
        metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average("loss"),
            delta_next_state_loss=nnx.metrics.Average("delta_next_state_loss"),
            reward_loss=nnx.metrics.Average("reward_loss"),
            terminated_loss=nnx.metrics.Average("terminated_loss"),
        )
        return model, optimizer, metrics

    model, optimizer, metrics = build(keys)
    return model, (optimizer, metrics)


def _make_batched_train_model(
    update_steps, minibatch_size, num_index_samples: int = 1, l2: float = 0.0
):
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
            num_index_samples,
            l2,
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
