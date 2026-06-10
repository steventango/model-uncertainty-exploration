import jax
import jax.numpy as jnp
import plotting
import ground_truth
from env_config import get_env_config, sample_validation_batch


def generate_validation_data(env, env_params, env_name, num_val=1000):
    val_rng = jax.random.PRNGKey(42)
    val_rng, val_x, _, val_s1, val_s2, val_act = sample_validation_batch(
        val_rng, env_name, env_params, num_val
    )

    val_true_delta_obs = ground_truth.true_delta_obs(
        env, env_params, env_name, val_s1, val_s2, val_act[:, 0]
    )
    val_true_reward = ground_truth.true_reward(
        env, env_params, env_name, val_s1, val_s2, val_act[:, 0]
    )

    return val_x, val_true_delta_obs, val_true_reward


def evaluate_validation(
    model,
    env,
    env_params,
    env_name,
    val_x,
    val_true_delta_obs,
    val_true_reward,
    dataset,
    pointer,
    rng,
    j,
):
    batch = jax.tree_util.tree_map(lambda x: x[:pointer], dataset)
    env_config = get_env_config(env_name)

    # Training data predictions (mean model, i.e. zero epistemic index)
    x_data = jnp.concatenate([batch.obs, batch.action], axis=-1)
    x_data = model.normalize_input(x_data)
    dummy_z = jnp.zeros(model.index_dim)
    _, mean_y = jax.vmap(model.__call__, in_axes=(0, None))(x_data, dummy_z)
    pred_delta_obs = model.denormalize_delta_obs(mean_y[..., :-2])
    pred_reward = model.denormalize_reward(mean_y[..., -2])

    true_delta_obs = batch.info["next_obs"] - batch.obs
    true_reward = batch.reward

    # Uniformly sampled state-space validation points
    num_rand = 1000
    rng, x_rand, _, rand_s1, rand_s2, rand_act = sample_validation_batch(
        rng, env_name, env_params, num_rand
    )
    x_rand = model.normalize_input(x_rand)

    _, mean_y_rand = jax.vmap(model.__call__, in_axes=(0, None))(x_rand, dummy_z)
    pred_delta_obs_rand = model.denormalize_delta_obs(mean_y_rand[..., :-2])
    pred_reward_rand = model.denormalize_reward(mean_y_rand[..., -2])

    true_delta_obs_rand = ground_truth.true_delta_obs(
        env, env_params, env_name, rand_s1, rand_s2, rand_act[:, 0]
    )
    true_reward_rand = ground_truth.true_reward(
        env, env_params, env_name, rand_s1, rand_s2, rand_act[:, 0]
    )

    # Epistemic uncertainty per output dimension
    unc_data, rng = plotting.compute_epistemic_uncertainty(model, x_data, rng)
    unc_rand, rng = plotting.compute_epistemic_uncertainty(model, x_rand, rng)
    unc_delta_obs = unc_data[..., :-2]
    unc_delta_obs_rand = unc_rand[..., :-2]
    unc_reward = unc_data[..., -2]
    unc_reward_rand = unc_rand[..., -2]

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
        unc_delta_obs,
        unc_delta_obs_rand,
        unc_reward,
        unc_reward_rand,
        env_config.delta_obs_labels,
        j,
    )

    # Evaluate MAE on validation set (mean model)
    val_x_norm = model.normalize_input(val_x)
    _, mean_y_val = jax.vmap(model.__call__, in_axes=(0, None))(val_x_norm, dummy_z)
    pred_delta_obs_val = model.denormalize_delta_obs(mean_y_val[..., :-2])
    pred_reward_val = model.denormalize_reward(mean_y_val[..., -2])

    dyn_mae = jnp.mean(jnp.abs(val_true_delta_obs - pred_delta_obs_val))
    rew_mae = jnp.mean(jnp.abs(val_true_reward - pred_reward_val))
    print(f"Iteration {j}: Dynamics MAE = {dyn_mae:.4f}, Reward MAE = {rew_mae:.4f}")

    return dyn_mae, rew_mae, rng
