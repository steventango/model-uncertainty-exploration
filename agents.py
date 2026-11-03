import os
from functools import partial

import jax
import orbax.checkpoint as ocp
from flax import nnx

from ppo import unstack_train_state


@partial(jax.jit, static_argnums=1)
def vsplit(keys, num=2):
    return jax.vmap(lambda k: jax.random.split(k, num), out_axes=1)(keys)


def seed_slice(tree, idx):
    """Index every leaf of a batched pytree, e.g. pick one seed (or (seed, config))."""
    return jax.tree_util.tree_map(lambda x: x[idx], tree)


def save_checkpoint(
    seed_dir, models, explore_train_state, eval_train_state, seed_idx, solver="ppo"
):
    checkpointer = ocp.StandardCheckpointer()
    abs_seed_dir = os.path.abspath(seed_dir)

    if models is not None:
        _, model_state = nnx.split(unstack_train_state(models, seed_idx))
        checkpointer.save(
            os.path.join(abs_seed_dir, "checkpoint", "model"), model_state
        )

    for tag, batched_ts in [
        (f"{solver}_explore", explore_train_state),
        (f"{solver}_eval", eval_train_state),
    ]:
        network, _, normalize_vec_obs, _ = unstack_train_state(batched_ts, seed_idx)
        _, network_state = nnx.split(network)
        _, obs_norm_state = nnx.split(normalize_vec_obs)
        base = os.path.join(abs_seed_dir, "checkpoint", tag)
        checkpointer.save(os.path.join(base, "network"), network_state)
        checkpointer.save(os.path.join(base, "obs_norm"), obs_norm_state)

    checkpointer.wait_until_finished()
