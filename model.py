import jax
import jax.numpy as jnp
import optax
from flax import nnx

from networks import MLP


def loss_fn(model: MLP, batch, rngs: nnx.Rngs | None = None):
    x = jnp.concatenate([batch.obs, batch.action], axis=-1)
    logits = model(x)

    delta_next_state = batch.info["next_obs"] - batch.obs
    delta_next_state_loss = (logits[..., :-2] - delta_next_state) ** 2
    delta_next_state_loss = delta_next_state_loss.mean()

    reward_loss = (logits[..., -2] - batch.reward) ** 2
    reward_loss = reward_loss.mean()

    terminated_pred = logits[..., -1]
    terminated_loss = optax.sigmoid_binary_cross_entropy(
        terminated_pred, batch.terminated.astype(jnp.float32)
    )
    terminated_loss = terminated_loss.mean()

    loss = delta_next_state_loss + reward_loss + terminated_loss
    return loss, (delta_next_state_loss, reward_loss, terminated_loss)


@nnx.jit
def train_step(
    model: MLP,
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


def _train_step(
    train_state,
    batch,
):
    model, optimizer, metrics, rngs = train_state
    train_step(model, optimizer, metrics, rngs, batch)
    return train_state, None


@nnx.jit
def eval_step(
    model: MLP,
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


def make_train_epoch(minibatch_size: int, num_minibatches: int):
    def train_epoch(train_state, unused):
        """Train for a single epoch."""
        model, optimizer, metrics, batch, rngs = train_state
        batch_size = minibatch_size * num_minibatches
        permutation = jax.random.permutation(rngs.epoch(), batch_size)
        shuffled_batch = jax.tree_util.tree_map(
            lambda x: jnp.take(x, permutation, axis=0), batch
        )
        minibatches = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, [num_minibatches, -1] + list(x.shape[1:])),
            shuffled_batch,
        )
        metrics.reset()
        inner_train_state = (model, optimizer, metrics, rngs)
        nnx.scan(_train_step)(inner_train_state, minibatches)
        return train_state, metrics.compute()

    return train_epoch


def train_model(
    model: MLP,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    batch,
    epochs: int,
    batch_size: int,
    minibatch_size: int,
    rngs: nnx.Rngs,
):
    """Train the model."""
    num_minibatches = batch_size // minibatch_size
    train_epoch = make_train_epoch(minibatch_size, num_minibatches)
    train_state = (model, optimizer, metrics, batch, rngs)
    _, history = nnx.scan(train_epoch, length=epochs)(train_state, None)
    return history
