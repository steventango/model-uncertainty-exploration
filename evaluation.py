import jax


def evaluate_policy(
    eval_config,
    eval_env,
    eval_env_params,
    eval_train_state,
    rng,
    rollout_fn,
):
    # INIT EVAL ENV
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, eval_config["NUM_ENVS"])
    eval_obsv, eval_env_state = eval_env.reset(reset_rng, eval_env_params)

    # ROLLOUT
    rng, _rng = jax.random.split(rng)
    eval_runner_state = (eval_train_state, eval_env_state, eval_obsv, _rng)
    eval_runner_state, traj_batch = rollout_fn(eval_runner_state)
    returned_episode = traj_batch.info["returned_episode"]
    returns = traj_batch.info["returned_episode_returns"][returned_episode]
    mean_return = returns.mean()
    std_return = returns.std()
    print(f"Mean evaluation return: {mean_return} +/- {std_return}")
    return mean_return
