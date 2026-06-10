import jax
import jax.numpy as jnp
import plotting
import ground_truth


def generate_validation_data(env, env_params, num_val=1000):
    val_rng = jax.random.PRNGKey(42)
    val_rng, val_subkey_theta, val_subkey_tdot, val_subkey_act = jax.random.split(
        val_rng, 4
    )
    val_theta = jax.random.uniform(
        val_subkey_theta, (num_val,), minval=-jnp.pi, maxval=jnp.pi
    )
    val_tdot = jax.random.uniform(val_subkey_tdot, (num_val,), minval=-8.0, maxval=8.0)
    val_act = jax.random.uniform(val_subkey_act, (num_val, 1), minval=-2.0, maxval=2.0)

    val_obs = jnp.stack([jnp.cos(val_theta), jnp.sin(val_theta), val_tdot], axis=-1)
    val_x = jnp.concatenate([val_obs, val_act], axis=-1)

    val_true_delta_obs = ground_truth.true_delta_obs(
        env, env_params, val_theta, val_tdot, val_act[:, 0]
    )
    val_true_reward = ground_truth.true_reward(
        env, env_params, val_theta, val_tdot, val_act[:, 0]
    )

    return val_x, val_true_delta_obs, val_true_reward


def evaluate_validation(
    model,
    env,
    env_params,
    val_x,
    val_true_delta_obs,
    val_true_reward,
    dataset,
    pointer,
    rng,
    j,
):
    batch = jax.tree_util.tree_map(lambda x: x[:pointer], dataset)

    # Training data predictions (mean model, i.e. zero epistemic index)
    x_data = jnp.concatenate([batch.obs, batch.action], axis=-1)
    dummy_z = jnp.zeros(model.index_dim)
    _, mean_y = jax.vmap(model.__call__, in_axes=(0, None))(x_data, dummy_z)
    pred_delta_obs = mean_y[..., :-2]
    pred_reward = mean_y[..., -2]

    true_delta_obs = batch.info["next_obs"] - batch.obs
    true_reward = batch.reward

    # Uniformly sampled state-space validation points
    num_rand = 1000
    rng, subkey_theta, subkey_tdot, subkey_act = jax.random.split(rng, 4)
    rand_theta = jax.random.uniform(
        subkey_theta, (num_rand,), minval=-jnp.pi, maxval=jnp.pi
    )
    rand_tdot = jax.random.uniform(subkey_tdot, (num_rand,), minval=-8.0, maxval=8.0)
    rand_act = jax.random.uniform(subkey_act, (num_rand, 1), minval=-2.0, maxval=2.0)

    rand_obs = jnp.stack([jnp.cos(rand_theta), jnp.sin(rand_theta), rand_tdot], axis=-1)
    x_rand = jnp.concatenate([rand_obs, rand_act], axis=-1)

    _, mean_y_rand = jax.vmap(model.__call__, in_axes=(0, None))(x_rand, dummy_z)
    pred_delta_obs_rand = mean_y_rand[..., :-2]
    pred_reward_rand = mean_y_rand[..., -2]

    true_delta_obs_rand = ground_truth.true_delta_obs(
        env, env_params, rand_theta, rand_tdot, rand_act[:, 0]
    )
    true_reward_rand = ground_truth.true_reward(
        env, env_params, rand_theta, rand_tdot, rand_act[:, 0]
    )

    # Plot true vs predicted
    plotting.plot_true_vs_predicted(
        true_delta_obs,
        pred_delta_obs,
        true_delta_obs_rand,
        pred_delta_obs_rand,
        true_reward,
        pred_reward,
        true_reward_rand,
        pred_reward_rand,
        j,
    )

    # Evaluate MAE on validation set (mean model)
    _, mean_y_val = jax.vmap(model.__call__, in_axes=(0, None))(val_x, dummy_z)
    pred_delta_obs_val = mean_y_val[..., :-2]
    pred_reward_val = mean_y_val[..., -2]

    dyn_mae = jnp.mean(jnp.abs(val_true_delta_obs - pred_delta_obs_val))
    rew_mae = jnp.mean(jnp.abs(val_true_reward - pred_reward_val))
    print(f"Iteration {j}: Dynamics MAE = {dyn_mae:.4f}, Reward MAE = {rew_mae:.4f}")

    return dyn_mae, rew_mae, rng
