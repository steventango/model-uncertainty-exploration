import os

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import ground_truth
from env_config import (
    action_title,
    action_visit_mask,
    get_env_config,
    grid_coords,
    is_discrete,
    obs_from_coords,
    visited_coords_from_obs,
)


def _contour_levels(vmin, vmax, n=50):
    """Strictly increasing levels for matplotlib.contourf."""
    vmin, vmax = float(vmin), float(vmax)
    if not jnp.isfinite(vmin) or not jnp.isfinite(vmax):
        return jnp.linspace(0.0, 1.0, n)
    span = vmax - vmin
    tol = 1e-10 * max(abs(vmin), abs(vmax), 1.0)
    if span <= tol:
        mid = vmin
        pad = max(0.5, abs(mid) * 0.1) if mid != 0.0 else 0.5
        return jnp.linspace(mid - pad, mid + pad, n)
    return jnp.linspace(vmin, vmax, n)


def _contour_levels_from_zero(vmax, n=50):
    """Strictly increasing levels from 0 for error contours."""
    vmax = float(vmax)
    if not jnp.isfinite(vmax) or vmax <= 0.0:
        return jnp.linspace(0.0, 1.0, n)
    return jnp.linspace(0.0, vmax, n)


def compute_epistemic_uncertainty(model, x_norm, rng, num_samples=10):
    """Std of normalized model outputs across epistemic index samples."""
    rng, subkey = jax.random.split(rng)
    index = model.sample_index(subkey, num_samples)
    return model.uncertainty(
        model.batch_predict_samples(x_norm, index), reduce_output=False
    ), rng


def evaluate_and_plot_uncertainty(
    model, env, env_params, env_name, rng, dataset, pointer, j, run_dir
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
    z_samples = model.sample_index(subkey, S_samples)

    # Pre-calculate grids for all actions to evaluate global color scales
    unc_grids = []
    true_rew_grids = []
    pred_rew_grids = []
    true_dyn_grids = []
    pred_dyn_grids = []
    true_term_grids = []
    pred_term_grids = []

    dyn_dim = env_config.dynamics_dim

    for idx, act in enumerate(actions):
        action_flat = jnp.full((obs_grid.shape[0], 1), act)
        x_grid = model.build_input(obs_grid, action_flat)
        x_grid_norm = model.normalize_input(x_grid)

        # 1. Epistemic Uncertainty (std of normalized model outputs)
        std_y = model.uncertainty(model.batch_predict_samples(x_grid_norm, z_samples))
        unc_grids.append(std_y.reshape(num_grid, num_grid))

        # 2. Mean Predictions (using the base network output)
        mean_y = jax.vmap(model.predict_mean)(x_grid_norm)
        pred_delta = model.denormalize_delta_obs(mean_y[..., : model.obs_dim])
        if model.predict_reward_terminated:
            pred_rew_grids.append(
                model.denormalize_reward(mean_y[..., -2]).reshape(num_grid, num_grid)
            )
            pred_term_grids.append(
                jax.nn.sigmoid(mean_y[..., -1]).reshape(num_grid, num_grid)
            )
        else:
            pred_rew_grids.append(jnp.zeros((num_grid, num_grid)))
            pred_term_grids.append(jnp.zeros((num_grid, num_grid)))
        pred_dyn_grids.append(pred_delta[..., dyn_dim].reshape(num_grid, num_grid))

        # 3. True Physics and Rewards (from the env, the single source of truth)
        act_flat = jnp.full_like(s1_flat, act)
        true_delta = ground_truth.true_delta_obs(
            env, env_params, env_name, s1_flat, s2_flat, act_flat
        )
        true_dyn_grids.append(true_delta[:, dyn_dim].reshape(num_grid, num_grid))

        true_reward = ground_truth.true_reward(
            env, env_params, env_name, s1_flat, s2_flat, act_flat
        )
        true_rew_grids.append(true_reward.reshape(num_grid, num_grid))

        true_term = ground_truth.true_terminated(
            env, env_params, env_name, s1_flat, s2_flat, act_flat
        )
        true_term_grids.append(true_term.reshape(num_grid, num_grid))

    plot_uncertainty(
        env_config,
        s1_axis,
        s2_axis,
        unc_grids,
        pred_rew_grids,
        pred_dyn_grids,
        true_dyn_grids,
        true_rew_grids,
        true_term_grids,
        pred_term_grids,
        actions,
        env_name,
        pointer,
        dataset,
        j,
        run_dir,
        discrete=is_discrete(env, env_params),
    )
    return rng


def _scatter_with_uncertainty(
    ax,
    true_vals,
    pred_vals,
    unc_vals,
    label,
    marker,
    vmin,
    vmax,
):
    return ax.scatter(
        true_vals,
        pred_vals,
        c=unc_vals,
        alpha=0.6,
        s=8,
        marker=marker,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        label=label,
    )


def plot_true_vs_predicted(
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
    delta_obs_labels,
    j,
    run_dir,
):
    obs_dim = true_delta_obs.shape[1]
    n_plots = obs_dim + 2
    n_cols = 2
    n_rows = (n_plots + 1) // 2
    fig_tp, axs_tp = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axs_tp = axs_tp.flatten()

    for i in range(obs_dim):
        unc_min = min(unc_delta_obs[:, i].min(), unc_delta_obs_rand[:, i].min())
        unc_max = max(unc_delta_obs[:, i].max(), unc_delta_obs_rand[:, i].max())
        sc = _scatter_with_uncertainty(
            axs_tp[i],
            true_delta_obs[:, i],
            pred_delta_obs[:, i],
            unc_delta_obs[:, i],
            "Training Data",
            "o",
            unc_min,
            unc_max,
        )
        _scatter_with_uncertainty(
            axs_tp[i],
            true_delta_obs_rand[:, i],
            pred_delta_obs_rand[:, i],
            unc_delta_obs_rand[:, i],
            "Uniform Space",
            "^",
            unc_min,
            unc_max,
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
        axs_tp[i].legend(loc="upper left", markerscale=2)
        axs_tp[i].grid(True, linestyle=":", alpha=0.6)
        cbar = fig_tp.colorbar(sc, ax=axs_tp[i], shrink=0.8, pad=0.02)
        cbar.set_label("Uncertainty (Std Dev)")

    reward_ax = axs_tp[obs_dim]
    unc_rew_min = min(unc_reward.min(), unc_reward_rand.min())
    unc_rew_max = max(unc_reward.max(), unc_reward_rand.max())
    sc_rew = _scatter_with_uncertainty(
        reward_ax,
        true_reward,
        pred_reward,
        unc_reward,
        "Training Data",
        "o",
        unc_rew_min,
        unc_rew_max,
    )
    _scatter_with_uncertainty(
        reward_ax,
        true_reward_rand,
        pred_reward_rand,
        unc_reward_rand,
        "Uniform Space",
        "^",
        unc_rew_min,
        unc_rew_max,
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
    reward_ax.legend(loc="upper left", markerscale=2)
    reward_ax.grid(True, linestyle=":", alpha=0.6)
    cbar_rew = fig_tp.colorbar(sc_rew, ax=reward_ax, shrink=0.8, pad=0.02)
    cbar_rew.set_label("Uncertainty (Std Dev)")

    term_ax = axs_tp[obs_dim + 1]
    unc_term_min = min(unc_terminated.min(), unc_terminated_rand.min())
    unc_term_max = max(unc_terminated.max(), unc_terminated_rand.max())
    sc_term = _scatter_with_uncertainty(
        term_ax,
        true_terminated,
        pred_terminated,
        unc_terminated,
        "Training Data",
        "o",
        unc_term_min,
        unc_term_max,
    )
    _scatter_with_uncertainty(
        term_ax,
        true_terminated_rand,
        pred_terminated_rand,
        unc_terminated_rand,
        "Uniform Space",
        "^",
        unc_term_min,
        unc_term_max,
    )
    term_ax.plot([0, 1], [0, 1], "r--", label="Perfect Match")
    term_ax.set_xlim(-0.05, 1.05)
    term_ax.set_ylim(-0.05, 1.05)
    term_ax.set_xlabel("True")
    term_ax.set_ylabel("Predicted")
    term_ax.set_title("Termination (probability)")
    term_ax.legend(loc="upper left", markerscale=2)
    term_ax.grid(True, linestyle=":", alpha=0.6)
    cbar_term = fig_tp.colorbar(sc_term, ax=term_ax, shrink=0.8, pad=0.02)
    cbar_term.set_label("Uncertainty (Std Dev)")

    for ax in axs_tp[n_plots:]:
        ax.set_visible(False)

    fig_tp.suptitle(
        f"True vs Predicted Dynamics, Rewards & Termination with Uncertainty (Iteration {j})",
        fontsize=14,
    )
    os.makedirs(run_dir, exist_ok=True)
    fig_tp_path = os.path.join(run_dir, f"true_vs_pred_{j:04d}.png")
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
    true_term_grids,
    pred_term_grids,
    actions,
    env_name,
    pointer,
    dataset,
    j,
    run_dir,
    *,
    discrete: bool,
):
    # Calculate global color scales across all subplots
    unc_min, unc_max = min(g.min() for g in unc_grids), max(g.max() for g in unc_grids)
    unc_levels = _contour_levels(unc_min, unc_max)

    rew_min = min(
        min(g.min() for g in true_rew_grids), min(g.min() for g in pred_rew_grids)
    )
    rew_max = max(
        max(g.max() for g in true_rew_grids), max(g.max() for g in pred_rew_grids)
    )
    rew_levels = _contour_levels(rew_min, rew_max)

    rew_err_grids = [jnp.abs(t - p) for t, p in zip(true_rew_grids, pred_rew_grids)]
    rew_err_max = max(g.max() for g in rew_err_grids)
    rew_err_levels = _contour_levels_from_zero(rew_err_max)

    term_levels = jnp.linspace(0.0, 1.0, 50)
    term_err_grids = [jnp.abs(t - p) for t, p in zip(true_term_grids, pred_term_grids)]
    term_err_max = max(g.max() for g in term_err_grids)
    term_err_levels = _contour_levels_from_zero(term_err_max)

    dyn_min = min(
        min(g.min() for g in true_dyn_grids), min(g.min() for g in pred_dyn_grids)
    )
    dyn_max = max(
        max(g.max() for g in true_dyn_grids), max(g.max() for g in pred_dyn_grids)
    )
    dyn_levels = _contour_levels(dyn_min, dyn_max)

    dyn_err_grids = [jnp.abs(t - p) for t, p in zip(true_dyn_grids, pred_dyn_grids)]
    dyn_err_max = max(g.max() for g in dyn_err_grids)
    dyn_err_levels = _contour_levels_from_zero(dyn_err_max)

    n_cols = len(actions)
    fig, axs = plt.subplots(
        10, n_cols, figsize=(6.5 * n_cols, 50), sharex=True, sharey=True
    )
    if n_cols == 1:
        axs = axs.reshape(10, 1)

    y_label = env_config.y_label
    x_label = env_config.x_label
    dynamics_label = env_config.dynamics_label

    for idx, act in enumerate(actions):
        # Row 0: Epistemic Uncertainty
        ax_unc = axs[0, idx]
        cf_unc = ax_unc.contourf(
            s1_axis, s2_axis, unc_grids[idx], levels=unc_levels, cmap="viridis"
        )
        if idx == 0:
            ax_unc.set_ylabel(f"{y_label}\n[Epistemic Uncertainty]")
        ax_unc.set_title(action_title(env_name, act, discrete=discrete))

        if pointer > 0:
            visited_obs = dataset.obs[:pointer]
            visited_actions = dataset.action[:pointer]
            mask = action_visit_mask(env_name, visited_actions, act, discrete=discrete)
            visited_s1, visited_s2 = visited_coords_from_obs(
                env_name, visited_obs[mask]
            )
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

        # Rows 1-3: Reward
        ax_true_rew = axs[1, idx]
        cf_true_rew = ax_true_rew.contourf(
            s1_axis, s2_axis, true_rew_grids[idx], levels=rew_levels, cmap="inferno"
        )
        if idx == 0:
            ax_true_rew.set_ylabel(f"{y_label}\n[True Reward]")

        ax_pred_rew = axs[2, idx]
        ax_pred_rew.contourf(
            s1_axis, s2_axis, pred_rew_grids[idx], levels=rew_levels, cmap="inferno"
        )
        if idx == 0:
            ax_pred_rew.set_ylabel(f"{y_label}\n[Predicted Reward]")

        ax_rew_err = axs[3, idx]
        cf_rew_err = ax_rew_err.contourf(
            s1_axis, s2_axis, rew_err_grids[idx], levels=rew_err_levels, cmap="Reds"
        )
        if idx == 0:
            ax_rew_err.set_ylabel(f"{y_label}\n[Reward Error]")

        # Rows 4-6: Termination
        ax_true_term = axs[4, idx]
        cf_true_term = ax_true_term.contourf(
            s1_axis, s2_axis, true_term_grids[idx], levels=term_levels, cmap="inferno"
        )
        if idx == 0:
            ax_true_term.set_ylabel(f"{y_label}\n[True Termination]")

        ax_pred_term = axs[5, idx]
        ax_pred_term.contourf(
            s1_axis, s2_axis, pred_term_grids[idx], levels=term_levels, cmap="inferno"
        )
        if idx == 0:
            ax_pred_term.set_ylabel(f"{y_label}\n[Predicted Termination]")

        ax_term_err = axs[6, idx]
        cf_term_err = ax_term_err.contourf(
            s1_axis, s2_axis, term_err_grids[idx], levels=term_err_levels, cmap="Reds"
        )
        if idx == 0:
            ax_term_err.set_ylabel(f"{y_label}\n[Termination Error]")

        # Rows 7-9: Dynamics
        ax_true_dyn = axs[7, idx]
        cf_true_dyn = ax_true_dyn.contourf(
            s1_axis, s2_axis, true_dyn_grids[idx], levels=dyn_levels, cmap="coolwarm"
        )
        if idx == 0:
            ax_true_dyn.set_ylabel(f"{y_label}\n[True {dynamics_label}]")

        ax_pred_dyn = axs[8, idx]
        ax_pred_dyn.contourf(
            s1_axis, s2_axis, pred_dyn_grids[idx], levels=dyn_levels, cmap="coolwarm"
        )
        if idx == 0:
            ax_pred_dyn.set_ylabel(f"{y_label}\n[Predicted {dynamics_label}]")

        ax_dyn_err = axs[9, idx]
        cf_dyn_err = ax_dyn_err.contourf(
            s1_axis, s2_axis, dyn_err_grids[idx], levels=dyn_err_levels, cmap="Reds"
        )
        if idx == 0:
            ax_dyn_err.set_ylabel(f"{y_label}\n[Dynamics Error]")
        ax_dyn_err.set_xlabel(x_label)

        if idx == n_cols - 1:
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
                cf_true_term, ax=axs[4:6, :], shrink=0.8, pad=0.02, location="right"
            ).set_label("Termination Prob.")
            fig.colorbar(
                cf_term_err, ax=axs[6, :], shrink=0.8, pad=0.02, location="right"
            ).set_label("Termination Abs Error")
            fig.colorbar(
                cf_true_dyn, ax=axs[7:9, :], shrink=0.8, pad=0.02, location="right"
            ).set_label(f"{dynamics_label} Scale")
            fig.colorbar(
                cf_dyn_err, ax=axs[9, :], shrink=0.8, pad=0.02, location="right"
            ).set_label("Dynamics Abs Error")

    fig.suptitle(
        f"Model Uncertainty, True vs Predicted & Absolute Errors (Iteration {j})",
        fontsize=16,
        y=0.98,
    )
    os.makedirs(run_dir, exist_ok=True)
    fig_path = os.path.join(run_dir, f"uncertainty_{j:04d}.png")
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    print(
        f"Saved action-dependent uncertainty, comparison and error plots to {fig_path}"
    )
    plt.close()
