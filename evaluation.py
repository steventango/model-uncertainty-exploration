import jax
import jax.numpy as jnp
from flax import nnx


def evaluate_policy(eval_config, env, env_params, rollout_fn, eval_train_state, rng):
    """Roll out the policy and return the mean return over completed episodes.

    Uses masked-mean (not boolean indexing) so it is vmap-safe.
    """
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, eval_config["NUM_ENVS"])
    obsv, env_state = env.reset(reset_rng, env_params)
    rng, _rng = jax.random.split(rng)
    runner_state = (eval_train_state, env_state, obsv, _rng)
    _, traj_batch = rollout_fn(runner_state)
    done = traj_batch.info["returned_episode"]
    returns = traj_batch.info["returned_episode_returns"]
    # Guard against NaN when no episode completed (done all-False): the masked
    # mean is then 0/0 = NaN, so fall back to 0.0 in that case.
    mean_return = jnp.mean(returns, where=done)
    return jnp.where(jnp.any(done), mean_return, 0.0)


vevaluate_policy = nnx.jit(
    nnx.vmap(evaluate_policy, in_axes=(None, None, None, None, 0, 0), out_axes=0)
)
