import matplotlib.pyplot as plt
import seaborn as sns
import jax
import jax.numpy as jnp
import pandas as pd
import ground_truth
from env_config import (
    action_title,
    action_visit_mask,
    get_env_config,
    grid_coords,
    obs_from_coords,
    visited_coords_from_obs,
)


def evaluate_and_plot_uncertainty(
    model, env, env_params, env_name, rng, dataset, pointer, j
):
    env_config = get_env_config(env_name)
    num_grid = 100
    s1_axis, s2_axis = grid_coords(env_name, env_params, num_grid)
    s1_grid, s2_grid = jnp.meshgrid(s1_axis, s2_axis)

    s1_flat = s1_grid.flatten()
    s2_flat = s2_grid.flatten()
    obs_grid = obs_from_coords(env_name, s1_flat, s2_flat)

    actions = list(env_config.representative_actions)

    S_samples = 10
    rng, subkey = jax.random.split(rng)
    z_samples = jax.random.normal(subkey, (S_samples, model.index_dim))

    # Pre-calculate grids for all actions to evaluate global color scales
    unc_grids = []
    true_rew_grids = []
    pred_rew_grids = []
    true_dyn_grids = []
    pred_dyn_grids = []

    dyn_dim = env_config.dynamics_dim

    for idx, act in enumerate(actions):
        action_flat = jnp.full((obs_grid.shape[0], 1), act)
        x_grid = jnp.concatenate([obs_grid, action_flat], axis=-1)
        x_grid_norm = model.normalize_input(x_grid)

        # 1. Epistemic Uncertainty (std of normalized model outputs)
        y_samples = jax.vmap(
            lambda z_j: jax.vmap(lambda xi: model(xi, z_j)[1])(x_grid_norm),
        )(z_samples)
        std_y = y_samples.std(axis=0).mean(axis=-1)
        unc_grids.append(std_y.reshape(num_grid, num_grid))

        # 2. Mean Predictions (using the base network output)
        dummy_z = jnp.zeros(model.index_dim)
        _, mean_y = jax.vmap(model.__call__, in_axes=(0, None))(x_grid_norm, dummy_z)
        pred_delta = model.denormalize_delta_obs(mean_y[..., :-2])
        pred_rew_grids.append(
            model.denormalize_reward(mean_y[..., -2]).reshape(num_grid, num_grid)
        )
        pred_dyn_grids.append(
            pred_delta[..., dyn_dim].reshape(num_grid, num_grid)
        )

        # 3. True Physics and Rewards (from the env, the single source of truth)
        act_flat = jnp.full_like(s1_flat, act)
        true_delta = ground_truth.true_delta_obs(
            env, env_params, env_name, s1_flat, s2_flat, act_flat
        )
        true_dyn_grids.append(
            true_delta[:, dyn_dim].reshape(num_grid, num_grid)
        )

        true_reward = ground_truth.true_reward(
            env, env_params, env_name, s1_flat, s2_flat, act_flat
        )
        true_rew_grids.append(true_reward.reshape(num_grid, num_grid))

    plot_uncertainty(
        env_config,
        s1_axis,
        s2_axis,
        unc_grids,
        pred_rew_grids,
        pred_dyn_grids,
        true_dyn_grids,
        true_rew_grids,
        actions,
        env_name,
        pointer,
        dataset,
        j,
    )
    return rng


def plot_dataset(dataset, pointer, env, env_params, rollout_config, j):
    n = (
        env.observation_space(env_params).shape[0]
        + env.action_space(env_params).shape[0]
        + 1
    )
    fig_width = min(3 * (j + 1), 50)
    fig, axs = plt.subplots(n, 1, figsize=(fig_width, 3 * n))

    # Observational dimensions
    for i in range(env.observation_space(env_params).shape[0]):
        axs[i].plot(dataset.obs[:pointer, i])
        axs[i].set_xlabel("Timestep")
        axs[i].set_ylabel(f"Observation {i}")
        axs[i].set_title(f"Observation {i} over timesteps")

    # Action dimensions
    for i in range(env.action_space(env_params).shape[0]):
        axs[env.observation_space(env_params).shape[0] + i].plot(
            dataset.action[:pointer, i]
        )
        axs[env.observation_space(env_params).shape[0] + i].set_xlabel("Timestep")
        axs[env.observation_space(env_params).shape[0] + i].set_ylabel(f"Action {i}")
        axs[env.observation_space(env_params).shape[0] + i].set_title(
            f"Action {i} over timesteps"
        )

    axs[-1].plot(dataset.reward[:pointer])
    axs[-1].set_xlabel("Timestep")
    axs[-1].set_ylabel("Reward")
    axs[-1].set_title("Reward over timesteps")

    # add vertical line when terminated | truncated
    done = dataset.terminated[:pointer] | dataset.truncated[:pointer]
    done_indices = jnp.where(done)[0]
    for idx in done_indices:
        for ax in axs:
            ax.axvline(x=idx, color="red", linestyle="--", alpha=0.5)

    for idx in range(
        0, pointer, rollout_config["NUM_STEPS"] * rollout_config["NUM_ENVS"]
    ):
        for ax in axs:
            ax.axvline(x=idx, color="blue", linestyle="--", alpha=0.5)

    fig_path = "/tmp/dataset.png"
    plt.savefig(fig_path)
    print(f"Saved dataset {j} curves to {fig_path}")
    plt.close()


def plot_losses(history):
    for loss, loss_history in history.items():
        plt.figure()
        plt.plot(loss_history)
        plt.xlabel("Epoch")
        plt.ylabel(loss)
        plt.title(f"{loss} over epochs")
        fig_path = f"/tmp/ppo_continuous_action_{loss}.png"
        plt.savefig(fig_path)
        plt.close()


def plot_true_vs_predicted(
    true_delta_obs,
    pred_delta_obs,
    true_delta_obs_rand,
    pred_delta_obs_rand,
    true_reward,
    pred_reward,
    true_reward_rand,
    pred_reward_rand,
    delta_obs_labels,
    j,
):
    obs_dim = true_delta_obs.shape[1]
    n_plots = obs_dim + 1
    n_cols = 2
    n_rows = (n_plots + 1) // 2
    fig_tp, axs_tp = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axs_tp = axs_tp.flatten()

    for i in range(obs_dim):
        axs_tp[i].scatter(
            true_delta_obs[:, i],
            pred_delta_obs[:, i],
            alpha=0.4,
            color="blue",
            s=5,
            label="Training Data",
        )
        axs_tp[i].scatter(
            true_delta_obs_rand[:, i],
            pred_delta_obs_rand[:, i],
            alpha=0.4,
            color="orange",
            s=5,
            label="Uniform Space",
        )
        min_val = min(
            true_delta_obs[:, i].min(),
            pred_delta_obs[:, i].min(),
            true_delta_obs_rand[:, i].min(),
            pred_delta_obs_rand[:, i].min(),
        )
        max_val = max(
            true_delta_obs[:, i].max(),
            pred_delta_obs[:, i].max(),
            true_delta_obs_rand[:, i].max(),
            pred_delta_obs_rand[:, i].max(),
        )
        axs_tp[i].plot(
            [min_val, max_val], [min_val, max_val], "r--", label="Perfect Match"
        )
        axs_tp[i].set_xlabel("True")
        axs_tp[i].set_ylabel("Predicted")
        axs_tp[i].set_title(f"Dynamics: {delta_obs_labels[i]}")
        axs_tp[i].legend()
        axs_tp[i].grid(True, linestyle=":", alpha=0.6)

    reward_ax = axs_tp[obs_dim]
    reward_ax.scatter(
        true_reward, pred_reward, alpha=0.4, color="blue", s=5, label="Training Data"
    )
    reward_ax.scatter(
        true_reward_rand,
        pred_reward_rand,
        alpha=0.4,
        color="orange",
        s=5,
        label="Uniform Space",
    )
    min_val = min(
        true_reward.min(),
        pred_reward.min(),
        true_reward_rand.min(),
        pred_reward_rand.min(),
    )
    max_val = max(
        true_reward.max(),
        pred_reward.max(),
        true_reward_rand.max(),
        pred_reward_rand.max(),
    )
    reward_ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Match")
    reward_ax.set_xlabel("True")
    reward_ax.set_ylabel("Predicted")
    reward_ax.set_title("Reward")
    reward_ax.legend()
    reward_ax.grid(True, linestyle=":", alpha=0.6)

    for ax in axs_tp[n_plots:]:
        ax.set_visible(False)

    fig_tp.suptitle(
        f"True vs Predicted Dynamics & Rewards (Iteration {j})", fontsize=14
    )
    fig_tp_path = "/tmp/ppo_continuous_action_true_vs_pred.png"
    plt.savefig(fig_tp_path, bbox_inches="tight", dpi=150)
    print(f"Saved true vs predicted validation plot to {fig_tp_path}")
    plt.close()


def plot_uncertainty(
    env_config,
    s1_axis,
    s2_axis,
    unc_grids,
    pred_rew_grids,
    pred_dyn_grids,
    true_dyn_grids,
    true_rew_grids,
    actions,
    env_name,
    pointer,
    dataset,
    j,
):
    # Calculate global color scales across all subplots
    unc_min, unc_max = min(g.min() for g in unc_grids), max(g.max() for g in unc_grids)
    unc_levels = jnp.linspace(unc_min, unc_max, 50) if unc_max > unc_min else 50

    rew_min = min(
        min(g.min() for g in true_rew_grids), min(g.min() for g in pred_rew_grids)
    )
    rew_max = max(
        max(g.max() for g in true_rew_grids), max(g.max() for g in pred_rew_grids)
    )
    rew_levels = jnp.linspace(rew_min, rew_max, 50) if rew_max > rew_min else 50

    rew_err_grids = [jnp.abs(t - p) for t, p in zip(true_rew_grids, pred_rew_grids)]
    rew_err_max = max(g.max() for g in rew_err_grids)
    rew_err_levels = jnp.linspace(0.0, rew_err_max, 50) if rew_err_max > 0.0 else 50

    dyn_min = min(
        min(g.min() for g in true_dyn_grids), min(g.min() for g in pred_dyn_grids)
    )
    dyn_max = max(
        max(g.max() for g in true_dyn_grids), max(g.max() for g in pred_dyn_grids)
    )
    dyn_levels = jnp.linspace(dyn_min, dyn_max, 50) if dyn_max > dyn_min else 50

    dyn_err_grids = [jnp.abs(t - p) for t, p in zip(true_dyn_grids, pred_dyn_grids)]
    dyn_err_max = max(g.max() for g in dyn_err_grids)
    dyn_err_levels = jnp.linspace(0.0, dyn_err_max, 50) if dyn_err_max > 0.0 else 50

    # Plot 7x3 grid
    fig, axs = plt.subplots(7, 3, figsize=(20, 35), sharex=True, sharey=True)

    y_label = env_config.y_label
    x_label = env_config.x_label
    dynamics_label = env_config.dynamics_label

    for idx, act in enumerate(actions):
        # Row 1: Epistemic Uncertainty
        ax_unc = axs[0, idx]
        cf_unc = ax_unc.contourf(
            s1_axis, s2_axis, unc_grids[idx], levels=unc_levels, cmap="viridis"
        )
        if idx == 0:
            ax_unc.set_ylabel(f"{y_label}\n[Epistemic Uncertainty]")
        ax_unc.set_title(action_title(env_name, act))

        # Overlay training data points on the uncertainty plot
        if pointer > 0:
            visited_obs = dataset.obs[:pointer]
            visited_actions = dataset.action[:pointer]
            mask = action_visit_mask(env_name, visited_actions, act)
            visited_s1, visited_s2 = visited_coords_from_obs(env_name, visited_obs[mask])
            ax_unc.scatter(
                visited_s1,
                visited_s2,
                color="red",
                alpha=0.3,
                s=2,
                label="Visited States",
            )
            if idx == 0:
                ax_unc.legend()

        # Row 2: True Reward
        ax_true_rew = axs[1, idx]
        cf_true_rew = ax_true_rew.contourf(
            s1_axis, s2_axis, true_rew_grids[idx], levels=rew_levels, cmap="inferno"
        )
        if idx == 0:
            ax_true_rew.set_ylabel(f"{y_label}\n[True Reward]")

        # Row 3: Predicted Reward
        ax_pred_rew = axs[2, idx]
        cf_pred_rew = ax_pred_rew.contourf(
            s1_axis, s2_axis, pred_rew_grids[idx], levels=rew_levels, cmap="inferno"
        )
        if idx == 0:
            ax_pred_rew.set_ylabel(f"{y_label}\n[Predicted Reward]")

        # Row 4: Reward Prediction Error
        ax_rew_err = axs[3, idx]
        cf_rew_err = ax_rew_err.contourf(
            s1_axis, s2_axis, rew_err_grids[idx], levels=rew_err_levels, cmap="Reds"
        )
        if idx == 0:
            ax_rew_err.set_ylabel(f"{y_label}\n[Reward Error]")

        # Row 5: True Dynamics
        ax_true_dyn = axs[4, idx]
        cf_true_dyn = ax_true_dyn.contourf(
            s1_axis, s2_axis, true_dyn_grids[idx], levels=dyn_levels, cmap="coolwarm"
        )
        if idx == 0:
            ax_true_dyn.set_ylabel(f"{y_label}\n[True {dynamics_label}]")

        # Row 6: Predicted Dynamics
        ax_pred_dyn = axs[5, idx]
        cf_pred_dyn = ax_pred_dyn.contourf(
            s1_axis, s2_axis, pred_dyn_grids[idx], levels=dyn_levels, cmap="coolwarm"
        )
        if idx == 0:
            ax_pred_dyn.set_ylabel(f"{y_label}\n[Predicted {dynamics_label}]")

        # Row 7: Dynamics Prediction Error
        ax_dyn_err = axs[6, idx]
        cf_dyn_err = ax_dyn_err.contourf(
            s1_axis, s2_axis, dyn_err_grids[idx], levels=dyn_err_levels, cmap="Reds"
        )
        if idx == 0:
            ax_dyn_err.set_ylabel(f"{y_label}\n[Dynamics Error]")
        ax_dyn_err.set_xlabel(x_label)

        # Add shared colorbars to the right of the rows
        if idx == 2:
            fig.colorbar(
                cf_unc, ax=axs[0, :], shrink=0.8, pad=0.02, location="right"
            ).set_label("Uncertainty (Std Dev)")
            fig.colorbar(
                cf_true_rew, ax=axs[1:3, :], shrink=0.8, pad=0.02, location="right"
            ).set_label("Reward Scale")
            fig.colorbar(
                cf_rew_err, ax=axs[3, :], shrink=0.8, pad=0.02, location="right"
            ).set_label("Reward Abs Error")
            fig.colorbar(
                cf_true_dyn, ax=axs[4:6, :], shrink=0.8, pad=0.02, location="right"
            ).set_label(f"{dynamics_label} Scale")
            fig.colorbar(
                cf_dyn_err, ax=axs[6, :], shrink=0.8, pad=0.02, location="right"
            ).set_label("Dynamics Abs Error")

    fig.suptitle(
        f"Model Uncertainty, True vs Predicted & Absolute Errors (Iteration {j})",
        fontsize=16,
        y=0.98,
    )
    fig_path = "/tmp/ppo_continuous_action_uncertainty.png"
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    print(
        f"Saved action-dependent uncertainty, comparison and error plots to {fig_path}"
    )
    plt.close()


def plot_training_curve(timesteps, returns, title, fig_path):
    df = pd.DataFrame(
        {
            "Steps": timesteps.flatten(),
            "Returns": returns.flatten(),
        }
    )
    plt.figure()
    sns.lineplot(x="Steps", y="Returns", data=df)
    plt.xlabel("Steps")
    plt.ylabel("Returns")
    plt.title(title)
    plt.savefig(fig_path)
    plt.close()


def plot_eval_returns(real_steps, eval_returns, env_name, fig_path):
    plt.figure()
    plt.plot(real_steps, eval_returns)
    plt.xlabel("Steps")
    plt.ylabel("Mean Evaluation Return")
    plt.title(f"PPO on Model({env_name}) Evaluation Returns")
    plt.savefig(fig_path)
    plt.close()
