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
      categorical: distribution over the next discrete state (one-hot obs).
    For the distributional heads each epistemic index z yields a full predictive
    distribution; exploration uses only the disagreement across z (epistemic).
    """

    def __init__(
        self,
        enn: ENN,
        in_features: int,
        obs_dim: int,
        act_dim: int | None = None,
        eps: float = 1e-8,
        predict_reward_terminated: bool = True,
        head: str = "deterministic",
        min_scale: float = 1e-3,
        bootstrap: Literal["none", "bernoulli", "gaussian"] = "none",
        bootstrap_noise: float = 1.0,
        l2: float = 0.0,
        weight_decay: float | None = None,
        label_smoothing: float = 0.0,
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
        head_out_features(head, obs_dim, predict_reward_terminated)  # validates
        self.head = head
        self.min_scale = min_scale
        self.enn = enn
        self.bootstrap = bootstrap
        self.bootstrap_noise = bootstrap_noise
        self.l2 = l2
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
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
        """Expected normalized delta-obs from the base network for a single input.

        gaussian: the Gaussian loc. categorical: E[one_hot(s')] - obs = probs - obs.
        """
        y, _ = self.enn.base_out(x)
        if self.head == "gaussian":
            return y[..., : self.obs_dim]
        if self.head == "categorical":
            return jax.nn.softmax(y, axis=-1) - self._obs(x)
        return y

    # --- Distributional heads ---

    def _obs(self, x):
        # Categorical inputs are left unnormalized (see update_stats), so the
        # one-hot obs is the first obs_dim entries of the model input.
        return x[..., : self.obs_dim]

    def _loc_scale(self, y):
        loc, raw_scale = y[..., : self.obs_dim], y[..., self.obs_dim :]
        return loc, jax.nn.softplus(raw_scale) + self.min_scale

    def dist_from_output(self, y) -> distrax.Distribution:
        """Raw network output → distrax predictive distribution."""
        if self.head == "gaussian":
            return distrax.MultivariateNormalDiag(*self._loc_scale(y))
        if self.head == "categorical":
            return distrax.Categorical(logits=y)
        raise ValueError("deterministic head has no predictive distribution")

    def sample_delta(self, x, index, key):
        if self.head == "deterministic":
            return super().sample_delta(x, index, key)
        sample = self.dist_from_output(self.predict_sample(x, index)).sample(seed=key)
        if self.head == "categorical":
            return jax.nn.one_hot(sample, self.obs_dim) - self._obs(x)
        return sample

    def _targets(self, obs, next_obs):
        """Distribution-space targets: next-state class or normalized delta-obs."""
        if self.head == "categorical":
            return jnp.argmax(next_obs, axis=-1)
        return self.normalize_delta_obs(next_obs - obs)

    def nll(self, x, obs, next_obs):
        if self.head == "deterministic":
            return super().nll(x, obs, next_obs)
        y, _ = self.enn.base_out(x)
        return -self.dist_from_output(y).log_prob(self._targets(obs, next_obs))

    def update_stats(self, dataset, pointer):
        super().update_stats(dataset, pointer)
        if self.head == "categorical":
            # One-hot obs: a per-column std is ~0 for unvisited states and would
            # blow up the input, and deltas are not regressed. Keep both raw.
            self.input_mean[: self.obs_dim] = 0.0
            self.input_std[: self.obs_dim] = 1.0
            self.delta_obs_mean[...] = 0.0
            self.delta_obs_std[...] = 1.0

    def uncertainty(self, x, z, kind="std", reduce_output=True):
        """Epistemic-only uncertainty at a single x over S indices z.

        Aleatoric noise is excluded: it is the spread *within* each index's
        predictive distribution, while these measure disagreement *across* z.
          categorical, eig: BALD mutual information H[E_z p] - E_z H[p], split
            into per-class terms that sum to it.
          normal, eig: ½ log(1 + Var_z[μ] / E_z[σ²]) per obs dim (Gaussian
            information gain with the predicted aleatoric noise).
          std (either head): sqrt(Var_z) of the mean parameters (probs or μ).
        """
        if self.head == "deterministic":
            return super().uncertainty(x, z, kind, reduce_output)
        y = self.predict_samples(x, z)  # (S, out_features)
        if self.head == "categorical":
            p = jax.nn.softmax(y, axis=-1)
            if kind == "eig":
                p_bar = p.mean(axis=0)
                u = -jax.scipy.special.xlogy(p_bar, p_bar) + jax.scipy.special.xlogy(
                    p, p
                ).mean(axis=0)
            else:
                u = jnp.sqrt(p.var(axis=0))
        else:
            loc, scale = self._loc_scale(y)
            if kind == "eig":
                u = 0.5 * jnp.log1p(loc.var(axis=0) / (scale**2).mean(axis=0))
            else:
                u = jnp.sqrt(loc.var(axis=0))
        return u.sum(axis=-1) if reduce_output else u

    def bootstrap_c(self, indices, num: int):
        key = self.bootstrap_key[...]
        c = jax.vmap(
            lambda i: jax.random.normal(
                jax.random.fold_in(key, i), (num, self.index_dim)
            )
        )(indices)
        return c / jnp.linalg.norm(c, axis=-1, keepdims=True)


def _projections(model: ENNModel, n, z, rngs: nnx.Rngs):
    """Per-datapoint ENN perturbation c_iᵀz with c_i uniform on the sphere (~N(0,1))."""
    c = jax.random.normal(rngs(), shape=(n, model.index_dim))
    c = c / jnp.linalg.norm(c, axis=-1, keepdims=True)
    return (c * z).sum(axis=-1)


def _distribution_loss(model: ENNModel, batch, rngs: nnx.Rngs):
    """NLL of the predictive distribution at one sampled epistemic index z."""
    x = model.normalize_input(model.build_input(batch.obs, batch.action))
    z = jax.random.normal(rngs(), shape=(model.index_dim,))
    y = model.batch_predict_sample(x, z)
    target = model._targets(batch.obs, batch.info["next_obs"])
    proj = _projections(model, batch.obs.shape[0], z, rngs)
    if model.head == "gaussian":
        # ENN target perturbation at the observation-noise level (predicted σ).
        loc, scale = model._loc_scale(y)
        target = target + jax.lax.stop_gradient(scale) * proj[:, None]
        return -distrax.MultivariateNormalDiag(loc, scale).log_prob(target).mean()
    soft = optax.smooth_labels(jax.nn.one_hot(target, y.shape[-1]), model.label_smoothing)
    nll = -(soft * jax.nn.log_softmax(y)).sum(-1)
    if model.bootstrap != "bernoulli":
        return nll.mean()
    mask = (proj > norm.ppf(0.5)).astype(jnp.float32)
    return (nll * mask).sum() / jnp.maximum(mask.sum(), 1.0)


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
    if model.head != "deterministic":
        nll = _distribution_loss(model, batch, rngs)
        return nll, (nll, jnp.zeros(()), jnp.zeros(()))
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
    decay = model.l2 / pointer if model.weight_decay is None else model.weight_decay
    optimizer.opt_state.hyperparams["l2"][...] = decay

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


# Default prior-function scale per head. The prior must perturb the outputs on the
# scale the data can drive them to, or the shared base network's extrapolation
# swamps it at unvisited inputs. Regression targets are normalized to unit std,
# so 1 matches. Categorical logits are unbounded under cross-entropy: on
# deterministic transitions the base learns margins of ~5-10 that transfer to
# unseen states through shared weights, so an O(1) prior leaves MI ~0 there.
_DEFAULT_PRIOR_SCALE = {"deterministic": 1.0, "gaussian": 1.0, "categorical": 10.0}


def _head_default(defaults, configured, head):
    return defaults[head] if configured is None else configured


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
        rngs = nnx.Rngs(key)
        epinet = model_config["EPINET"]
        enn = ENN(
            in_features,
            model_config["HIDDEN_DIM"],
            epinet["learnable_hidden_dim"],
            epinet["prior_hidden_dim"],
            out_features,
            epinet["index_dim"],
            use_layer_norm=model_config.get("USE_LAYER_NORM", False),
            num_layers=model_config["NUM_LAYERS"],
            activation=model_config["ACTIVATION"],
            epinet_num_layers=epinet["num_layers"],
            epinet_activation=epinet["activation"],
            prior_scale=_head_default(
                _DEFAULT_PRIOR_SCALE, model_config.get("PRIOR_SCALE"), head
            ),
            embed_blocks=(
                (obs_dim, in_features - obs_dim) if head == "categorical" else None
            ),
            embed=model_config.get("EMBED", "identity"),
            embed_dim=model_config.get("EMBED_DIM"),
            embed_trainable=model_config.get("EMBED_TRAINABLE", False),
            onehot_init=model_config.get("ONEHOT_INIT", "lecun"),
            rngs=rngs,
        )
        model = ENNModel(
            enn,
            in_features,
            obs_dim,
            act_dim=act_dim,
            predict_reward_terminated=predict_reward_terminated,
            head=head,
            min_scale=model_config.get("MIN_SCALE", 1e-3),
            bootstrap=model_config["BOOTSTRAP"],
            bootstrap_noise=model_config["BOOTSTRAP_NOISE"],
            l2=model_config["L2"],
            weight_decay=model_config.get("WEIGHT_DECAY"),
            label_smoothing=model_config.get("LABEL_SMOOTHING", 0.0),
            rngs=rngs,
        )
        tx = optax.inject_hyperparams(
            lambda l2: optax.adamw(model_config["LR"], weight_decay=l2)
        )(l2=0.0)
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
