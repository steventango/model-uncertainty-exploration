from functools import partial

from flax import nnx
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
import optax

from networks import ENN


class DynamicsModel(nnx.Module):
    """ENN with dataset normalization stats for inputs and targets."""

    def __init__(
        self,
        enn: ENN,
        in_features: int,
        obs_dim: int,
        act_dim: int | None = None,
        eps: float = 1e-8,
    ):
        self.enn = enn
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.eps = eps
        self.input_mean = nnx.Variable(jnp.zeros(in_features))
        self.input_std = nnx.Variable(jnp.ones(in_features))
        self.delta_obs_mean = nnx.Variable(jnp.zeros(obs_dim))
        self.delta_obs_std = nnx.Variable(jnp.ones(obs_dim))
        self.reward_mean = nnx.Variable(jnp.zeros(()))
        self.reward_std = nnx.Variable(jnp.ones(()))

    @property
    def index_dim(self):
        return self.enn.index_dim

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

    def __call__(self, x, z, rngs: nnx.Rngs | None = None):
        return self.enn(x, z, rngs=rngs)


def loss_fn(model: DynamicsModel, batch, rngs: nnx.Rngs):
    sigma = 1.0
    x = model.build_input(batch.obs, batch.action)
    x = model.normalize_input(x)
    z = jax.random.normal(rngs(), shape=(model.index_dim,))
    _, logits = jax.vmap(model.__call__, in_axes=(0, None))(x, z)

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
    delta_next_state_loss = (logits[..., :-2] - delta_next_state_target) ** 2
    delta_next_state_loss = delta_next_state_loss.mean()

    reward_c = jax.random.normal(rngs(), shape=(batch.obs.shape[0], model.index_dim))
    reward_c = reward_c / jnp.linalg.norm(reward_c, axis=-1, keepdims=True)
    reward_target = model.normalize_reward(batch.reward) + sigma * (reward_c * z).sum(
        axis=-1
    )
    reward_loss = (logits[..., -2] - reward_target) ** 2
    reward_loss = reward_loss.mean()

    terminated_c = jax.random.normal(
        rngs(), shape=(batch.obs.shape[0], model.index_dim)
    )
    terminated_c = terminated_c / jnp.linalg.norm(terminated_c, axis=-1, keepdims=True)
    p = 0.5
    mask = ((terminated_c * z).sum(axis=-1) > norm.ppf(p)).astype(jnp.float32)
    terminated_target = batch.terminated.astype(jnp.float32)
    terminated_pred = logits[..., -1]
    terminated_loss = optax.sigmoid_binary_cross_entropy(
        terminated_pred, terminated_target
    )
    terminated_loss = (terminated_loss * mask).sum() / jnp.maximum(mask.sum(), 1.0)

    loss = delta_next_state_loss + reward_loss + terminated_loss
    return loss, (delta_next_state_loss, reward_loss, terminated_loss)


@nnx.jit
def train_step(
    model: DynamicsModel,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    rngs: nnx.Rngs,
    batch,
):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(model, batch, rngs)
    delta_next_state_loss, reward_loss, terminated_loss = aux
    metrics.update(
        loss=loss,
        delta_next_state_loss=delta_next_state_loss,
        reward_loss=reward_loss,
        terminated_loss=terminated_loss,
    )
    optimizer.update(model, grads)


@nnx.jit
def eval_step(
    model: DynamicsModel,
    metrics: nnx.MultiMetric,
    rngs: nnx.Rngs,
    batch,
):
    """Eval for a single step."""
    loss, aux = loss_fn(model, batch, rngs)
    delta_next_state_loss, reward_loss, terminated_loss = aux
    metrics.update(
        loss=loss,
        delta_next_state_loss=delta_next_state_loss,
        reward_loss=reward_loss,
        terminated_loss=terminated_loss,
    )


@partial(nnx.jit, static_argnums=(4, 6))
def train_model(
    model: DynamicsModel,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    dataset,
    update_steps: int,
    pointer: int,
    minibatch_size: int,
    rngs: nnx.Rngs,
):
    """Train the model."""
    model.update_stats(dataset, pointer)

    def train_step_fn(train_state, _):
        model, optimizer, metrics, rngs = train_state
        indices = jax.random.randint(rngs(), (minibatch_size,), 0, pointer)
        minibatch = jax.tree_util.tree_map(lambda x: jnp.take(x, indices, axis=0), dataset)
        metrics.reset()
        train_step(model, optimizer, metrics, rngs, minibatch)
        return (model, optimizer, metrics, rngs), metrics.compute()

    train_state = (model, optimizer, metrics, rngs)
    _, history = nnx.scan(train_step_fn, length=update_steps)(train_state, None)
    return history
