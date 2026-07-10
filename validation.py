import jax
import jax.numpy as jnp
import optax
from flax import nnx
import plotting
import ground_truth
from env_config import get_env_config, sample_validation_batch


def generate_brax_validation_data(env, env_params, num_val=1000):
    """Generate validation data for brax envs by running rollout transitions.

    Returns the same 5-tuple as generate_validation_data().
    """
    rng = jax.random.PRNGKey(42)
    all_keys = jax.random.split(rng, num_val * 3).reshape(num_val, 3, -1)
    reset_keys, act_keys, step_keys = (
        all_keys[:, 0, :],
        all_keys[:, 1, :],
        all_keys[:, 2, :],
    )

    def sample_transition(k_reset, k_act, k_step):
        obs, state = env.reset(k_reset, env_params)
        action = env.action_space(env_params).sample(k_act)
        _, _, reward, terminated, _, info = env.step(k_step, state, action, env_params)
        delta_obs = info["next_obs"] - obs
        return obs, action, delta_obs, reward, terminated.astype(jnp.float32)

    val_obs, val_act, val_delta_obs, val_reward, val_terminated = jax.vmap(
        sample_transition
    )(reset_keys, act_keys, step_keys)
    return val_obs, val_act, val_delta_obs, val_reward, val_terminated


def generate_validation_data(env, env_params, env_name, num_val=1000):
    val_rng = jax.random.PRNGKey(42)
    val_rng, val_obs, val_s1, val_s2, val_act = sample_validation_batch(
        val_rng, env, env_params, env_name, num_val
    )

    val_true_delta_obs = ground_truth.true_delta_obs(
        env, env_params, env_name, val_s1, val_s2, val_act[:, 0]
    )
    val_true_reward = ground_truth.true_reward(
        env, env_params, env_name, val_s1, val_s2, val_act[:, 0]
    )
    val_true_terminated = ground_truth.true_terminated(
        env, env_params, env_name, val_s1, val_s2, val_act[:, 0]
    )

    return val_obs, val_act, val_true_delta_obs, val_true_reward, val_true_terminated


def _model_inputs(model, obs, action):
    return model.normalize_input(model.build_input(obs, action))


_NUM_UNC_SAMPLES = 10


def validation_metrics(
    model,
    val_obs,
    val_act,
    val_true_delta_obs,
    val_true_reward,
    val_true_terminated,
    key,
):
    """Pure-JAX validation metrics (mean model + uncertainty). Vmappable over a batch of models."""
    val_x_norm = _model_inputs(model, val_obs, val_act)
    mean_y_val = jax.vmap(model.predict_mean)(val_x_norm)
    pred_delta_obs_val = model.denormalize_delta_obs(mean_y_val[..., : model.obs_dim])

    dyn_mae = jnp.mean(jnp.abs(val_true_delta_obs - pred_delta_obs_val))

    if model.predict_reward_terminated:
        pred_reward_val = model.denormalize_reward(mean_y_val[..., -2])
        pred_terminated_val = jax.nn.sigmoid(mean_y_val[..., -1])
        rew_mae = jnp.mean(jnp.abs(val_true_reward - pred_reward_val))
        term_bce = jnp.mean(
            optax.sigmoid_binary_cross_entropy(mean_y_val[..., -1], val_true_terminated)
        )
        pred_pos = pred_terminated_val > 0.5
        true_pos = val_true_terminated.astype(bool)
        tp = jnp.sum(pred_pos & true_pos)
        fp = jnp.sum(pred_pos & ~true_pos)
        fn = jnp.sum(~pred_pos & true_pos)
        precision = tp / jnp.maximum(tp + fp, 1)
        recall = tp / jnp.maximum(tp + fn, 1)
        term_f1 = 2 * precision * recall / jnp.maximum(precision + recall, 1e-8)
    else:
        rew_mae = jnp.zeros(())
        term_bce = jnp.zeros(())
        term_f1 = jnp.zeros(())

    idx = model.sample_index(key, _NUM_UNC_SAMPLES)
    mean_uncertainty = model.batch_uncertainty(
        val_x_norm, idx, reduce_output=True
    ).mean()

    return dyn_mae, rew_mae, term_bce, term_f1, mean_uncertainty


def make_batched_validation_metrics():
    """Vmap validation_metrics over a leading seed axis on the model."""
    return nnx.jit(
        nnx.vmap(
            validation_metrics,
            in_axes=(0, None, None, None, None, None, 0),
            out_axes=0,
        )
    )


def evaluate_validation(
    model,
    env,
    env_params,
    env_name,
    val_obs,
    val_act,
    val_true_delta_obs,
    val_true_reward,
    val_true_terminated,
    dataset,
    pointer,
    rng,
    j,
    run_dir,
    *,
    plot=False,
):
    if plot:
        batch = jax.tree_util.tree_map(lambda x: x[:pointer], dataset)
        env_config = get_env_config(env_name)

        # Training data predictions (mean model, i.e. zero epistemic index)
        x_data = _model_inputs(model, batch.obs, batch.action)
        mean_y = jax.vmap(model.predict_mean)(x_data)
        pred_delta_obs = model.denormalize_delta_obs(mean_y[..., : model.obs_dim])
        pred_reward = (
            model.denormalize_reward(mean_y[..., -2])
            if model.predict_reward_terminated
            else None
        )
        pred_terminated = (
            jax.nn.sigmoid(mean_y[..., -1]) if model.predict_reward_terminated else None
        )

        true_delta_obs = batch.info["next_obs"] - batch.obs
        true_reward = batch.reward
        true_terminated = batch.terminated.astype(jnp.float32)

        # Uniformly sampled state-space validation points
        num_rand = 1000
        rng, rand_obs, rand_s1, rand_s2, rand_act = sample_validation_batch(
            rng, env, env_params, env_name, num_rand
        )
        x_rand = _model_inputs(model, rand_obs, rand_act)

        mean_y_rand = jax.vmap(model.predict_mean)(x_rand)
        pred_delta_obs_rand = model.denormalize_delta_obs(
            mean_y_rand[..., : model.obs_dim]
        )
        pred_reward_rand = (
            model.denormalize_reward(mean_y_rand[..., -2])
            if model.predict_reward_terminated
            else None
        )
        pred_terminated_rand = (
            jax.nn.sigmoid(mean_y_rand[..., -1])
            if model.predict_reward_terminated
            else None
        )

        true_delta_obs_rand = ground_truth.true_delta_obs(
            env, env_params, env_name, rand_s1, rand_s2, rand_act[:, 0]
        )
        true_reward_rand = ground_truth.true_reward(
            env, env_params, env_name, rand_s1, rand_s2, rand_act[:, 0]
        )
        true_terminated_rand = ground_truth.true_terminated(
            env, env_params, env_name, rand_s1, rand_s2, rand_act[:, 0]
        )

        # Epistemic uncertainty per output dimension
        rng, key_data, key_rand = jax.random.split(rng, 3)
        idx_data = model.sample_index(key_data, 10)
        unc_data = model.batch_uncertainty(x_data, idx_data, reduce_output=False)
        idx_rand = model.sample_index(key_rand, 10)
        unc_rand = model.batch_uncertainty(x_rand, idx_rand, reduce_output=False)
        unc_delta_obs = unc_data[..., : model.obs_dim]
        unc_delta_obs_rand = unc_rand[..., : model.obs_dim]
        unc_reward = unc_data[..., -2] if model.predict_reward_terminated else None
        unc_reward_rand = unc_rand[..., -2] if model.predict_reward_terminated else None
        unc_terminated = unc_data[..., -1] if model.predict_reward_terminated else None
        unc_terminated_rand = (
            unc_rand[..., -1] if model.predict_reward_terminated else None
        )

        plotting.plot_true_vs_predicted(
            true_delta_obs,
            pred_delta_obs,
            true_delta_obs_rand,
            pred_delta_obs_rand,
            true_reward,
            pred_reward,
            true_reward_rand,
            pred_reward_rand,
            true_terminated,
            pred_terminated,
            true_terminated_rand,
            pred_terminated_rand,
            unc_delta_obs,
            unc_delta_obs_rand,
            unc_reward,
            unc_reward_rand,
            unc_terminated,
            unc_terminated_rand,
            env_config.delta_obs_labels,
            j,
            run_dir,
            predict_reward_terminated=model.predict_reward_terminated,
        )

    # Evaluate MAE on validation set (mean model)
    rng, val_key = jax.random.split(rng)
    dyn_mae, rew_mae, term_bce, term_f1, mean_uncertainty = validation_metrics(
        model,
        val_obs,
        val_act,
        val_true_delta_obs,
        val_true_reward,
        val_true_terminated,
        val_key,
    )
    print(
        f"Iteration {j}: Dynamics MAE = {dyn_mae:.4f}, Reward MAE = {rew_mae:.4f}, "
        f"Termination BCE = {term_bce:.4f}, Termination F1 = {term_f1:.4f}, "
        f"Mean Uncertainty = {mean_uncertainty:.4f}"
    )

    return dyn_mae, rew_mae, term_bce, term_f1, mean_uncertainty, rng
