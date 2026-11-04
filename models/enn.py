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
      normal: diagonal Gaussian over the normalized delta-obs (aleatoric noise).
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
        l2_reg: float = 0.0,
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
        self.l2_reg = l2_reg
        self.enn = enn

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

        normal: the Gaussian loc. categorical: E[one_hot(s')] - obs = probs - obs.
        """
        y, _ = self.enn.base(x)
        if self.head == "normal":
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
        if self.head == "normal":
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
        y, _ = self.enn.base(x)
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
    if model.head == "normal":
        # ENN target perturbation at the observation-noise level (predicted σ).
        loc, scale = model._loc_scale(y)
        target = target + jax.lax.stop_gradient(scale) * proj[:, None]
        return -distrax.MultivariateNormalDiag(loc, scale).log_prob(target).mean()
    # Categorical: discrete analogue of the perturbation (as for the terminated
    # head) — each index fits a random half of the data.
    mask = (proj > norm.ppf(0.5)).astype(jnp.float32)
    nll = -distrax.Categorical(logits=y).log_prob(target)
    return (nll * mask).sum() / jnp.maximum(mask.sum(), 1.0)


_TRAINABLE = nnx.All(nnx.Param, nnx.Not(nnx.PathContains("prior")))


def _l2_penalty(model: ENNModel, n_data):
    """(λ/N)·‖θ‖² over the trainable (non-prior) parameters.

    With the per-datapoint *mean* loss this is the MAP objective
    Σ_i ℓ_i + λ‖θ‖² divided by N: k samples at an input are weighed against a
    fixed prior strength λ regardless of the dataset size, so evidence (and
    hence epistemic uncertainty) scales with visit counts. Adam's normalized
    steps preserve this objective's stationary points, unlike AdamW's
    decoupled decay, which bounds weights at ~1/wd irrespective of the data.
    """
    sq = sum(jnp.sum(p**2) for p in jax.tree.leaves(nnx.state(model, _TRAINABLE)))
    return model.l2_reg * sq / jnp.maximum(n_data, 1)


def _loss_fn(model: ENNModel, batch, rngs: nnx.Rngs, n_data):
    if model.head != "deterministic":
        nll = _distribution_loss(model, batch, rngs)
        return nll + _l2_penalty(model, n_data), (nll, jnp.zeros(()), jnp.zeros(()))
    sigma = 1.0
    x = model.build_input(batch.obs, batch.action)
    x = model.normalize_input(x)
    z = jax.random.normal(rngs(), shape=(model.index_dim,))
    logits = model.batch_predict_sample(x, z)

    delta_next_state_c = jax.random.normal(
        rngs(), shape=(batch.obs.shape[0], model.index_dim)
    )
    delta_next_state_c = delta_next_state_c / jnp.linalg.norm(
        delta_next_state_c, axis=-1, keepdims=True
    )

    delta_next_state = batch.info["next_obs"] - batch.obs
    delta_next_state = model.normalize_delta_obs(delta_next_state)
    delta_next_state_target = delta_next_state + sigma * (delta_next_state_c * z).sum(
        axis=-1, keepdims=True
    )
    delta_next_state_loss = (
        logits[..., : model.obs_dim] - delta_next_state_target
    ) ** 2
    delta_next_state_loss = delta_next_state_loss.mean()

    if model.predict_reward_terminated:
        reward_c = jax.random.normal(
            rngs(), shape=(batch.obs.shape[0], model.index_dim)
        )
        reward_c = reward_c / jnp.linalg.norm(reward_c, axis=-1, keepdims=True)
        reward_target = model.normalize_reward(batch.reward) + sigma * (
            reward_c * z
        ).sum(axis=-1)
        reward_loss = (logits[..., -2] - reward_target) ** 2
        reward_loss = reward_loss.mean()

        terminated_c = jax.random.normal(
            rngs(), shape=(batch.obs.shape[0], model.index_dim)
        )
        terminated_c = terminated_c / jnp.linalg.norm(
            terminated_c, axis=-1, keepdims=True
        )
        p = 0.5
        mask = ((terminated_c * z).sum(axis=-1) > norm.ppf(p)).astype(jnp.float32)
        terminated_target = batch.terminated.astype(jnp.float32)
        terminated_pred = logits[..., -1]
        terminated_loss = optax.sigmoid_binary_cross_entropy(
            terminated_pred, terminated_target
        )
        terminated_loss = (terminated_loss * mask).sum() / jnp.maximum(mask.sum(), 1.0)
    else:
        reward_loss = jnp.zeros(())
        terminated_loss = jnp.zeros(())

    loss = delta_next_state_loss + reward_loss + terminated_loss
    loss = loss + _l2_penalty(model, n_data)
    return loss, (delta_next_state_loss, reward_loss, terminated_loss)


@nnx.jit
def _train_step(
    model: ENNModel,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    rngs: nnx.Rngs,
    batch,
    n_data,
):
    grad_fn = nnx.value_and_grad(_loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(model, batch, rngs, n_data)
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
    n_data,
):
    loss, aux = _loss_fn(model, batch, rngs, n_data)
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
        _train_step(model, optimizer, metrics, rngs, minibatch, pointer)
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
_DEFAULT_PRIOR_SCALE = {"deterministic": 1.0, "normal": 1.0, "categorical": 10.0}

# Default λ of the (λ/N)·‖θ‖² MAP penalty per head. Without it, cross-entropy
# training collapses the categorical posterior after ~1 visit (MI drops >90%),
# more so the longer it trains. λ=1 tracks a Dirichlet posterior's decay with
# visit counts on RiverSwim and is insensitive to training length. Other heads
# keep their previous (unregularized) objective.
_DEFAULT_L2_REG = {"deterministic": 0.0, "normal": 0.0, "categorical": 1.0}


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
        enn = ENN(
            in_features,
            model_config["HIDDEN_DIM"],
            model_config["LEARNABLE_HIDDEN_DIM"],
            model_config["PRIOR_HIDDEN_DIM"],
            out_features,
            model_config["INDEX_DIM"],
            use_layer_norm=model_config.get("USE_LAYER_NORM", False),
            prior_scale=_head_default(
                _DEFAULT_PRIOR_SCALE, model_config.get("PRIOR_SCALE"), head
            ),
            rngs=nnx.Rngs(params=key),
        )
        model = ENNModel(
            enn,
            in_features,
            obs_dim,
            act_dim=act_dim,
            predict_reward_terminated=predict_reward_terminated,
            head=head,
            min_scale=model_config.get("MIN_SCALE", 1e-3),
            l2_reg=_head_default(_DEFAULT_L2_REG, model_config.get("L2_REG"), head),
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
