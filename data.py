import jax
import jax.numpy as jnp


def _batch_action(action):
    """RL rollout actions → (batch, a_dim) for DynamicsModel."""
    action = jnp.atleast_1d(action)
    if action.ndim == 1:
        return action[:, None]
    return action


def collate_rollout(traj_batch):
    """Flatten (T, E, …) PPO transition batch → (T*E, …) with model action layout."""
    traj_batch = jax.tree_util.tree_map(
        lambda x: x.reshape((-1,) + x.shape[2:]), traj_batch
    )
    return traj_batch._replace(action=_batch_action(traj_batch.action))
