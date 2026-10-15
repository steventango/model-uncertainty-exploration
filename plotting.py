import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import jax
import jax.numpy as jnp
import ground_truth
from model_env import reset_weights
from ppo import make_rollout, unstack_train_state
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
    return model.batch_uncertainty(x_norm, index, reduce_output=False), rng


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

    obs_dim = model.obs_dim

    for idx, act in enumerate(actions):
        action_flat = jnp.full((obs_grid.shape[0], 1), act)
        x_grid = model.build_input(obs_grid, action_flat)
        x_grid_norm = model.normalize_input(x_grid)

        # 1. Epistemic Uncertainty (std of normalized model outputs)
        std_y = model.batch_uncertainty(x_grid_norm, z_samples)
        unc_grids.append(std_y.reshape(num_grid, num_grid))

        # 2. Mean Predictions (using the base network output)
        mean_y = jax.vmap(model.predict_mean)(x_grid_norm)
        pred_delta = model.denormalize_delta_obs(mean_y[..., :obs_dim])
        if model.predict_reward_terminated:
            pred_rew_grids.append(
                model.denormalize_reward(mean_y[..., model.reward_index]).reshape(num_grid, num_grid)
            )
            pred_term_grids.append(
                jax.nn.sigmoid(mean_y[..., model.terminated_index]).reshape(num_grid, num_grid)
            )
        # Store all obs dims: list[n_actions] of list[obs_dim] of (G, G) grids
        pred_dyn_grids.append(
            [pred_delta[..., d].reshape(num_grid, num_grid) for d in range(obs_dim)]
        )

        # 3. True Physics and Rewards (from the env, the single source of truth)
        act_flat = jnp.full_like(s1_flat, act)
        true_delta = ground_truth.true_delta_obs(
            env, env_params, env_name, s1_flat, s2_flat, act_flat
        )
        true_dyn_grids.append(
            [true_delta[:, d].reshape(num_grid, num_grid) for d in range(obs_dim)]
        )

        if model.predict_reward_terminated:
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
        predict_reward_terminated=model.predict_reward_terminated,
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
    *,
    predict_reward_terminated: bool = True,
):
    obs_dim = true_delta_obs.shape[1]
    n_plots = obs_dim + (2 if predict_reward_terminated else 0)
    n_cols = 2
    n_rows = (n_plots + 1) // 2
    fig_tp, axs_tp = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
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

    if predict_reward_terminated:
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
        reward_ax.plot(
            [min_val, max_val], [min_val, max_val], "r--", label="Perfect Match"
        )
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

    title_suffix = (
        "Dynamics, Rewards & Termination" if predict_reward_terminated else "Dynamics"
    )
    fig_tp.suptitle(
        f"True vs Predicted {title_suffix} with Uncertainty (Iteration {j})",
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
    predict_reward_terminated: bool = True,
):
    # Calculate global color scales across all subplots
    unc_min, unc_max = min(g.min() for g in unc_grids), max(g.max() for g in unc_grids)
    unc_levels = _contour_levels(unc_min, unc_max)

    # Per-obs-dim color scales: n_obs_dims x n_actions grids
    n_obs_dims = len(true_dyn_grids[0])
    dyn_levels_per_dim = []
    dyn_err_grids_per_dim = []
    dyn_err_levels_per_dim = []
    for d in range(n_obs_dims):
        grids_true = [true_dyn_grids[i][d] for i in range(len(true_dyn_grids))]
        grids_pred = [pred_dyn_grids[i][d] for i in range(len(pred_dyn_grids))]
        dmin = min(min(g.min() for g in grids_true), min(g.min() for g in grids_pred))
        dmax = max(max(g.max() for g in grids_true), max(g.max() for g in grids_pred))
        dyn_levels_per_dim.append(_contour_levels(dmin, dmax))
        err_grids = [jnp.abs(t - p) for t, p in zip(grids_true, grids_pred)]
        dyn_err_grids_per_dim.append(err_grids)
        dyn_err_levels_per_dim.append(
            _contour_levels_from_zero(max(g.max() for g in err_grids))
        )

    if predict_reward_terminated:
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
        term_err_grids = [
            jnp.abs(t - p) for t, p in zip(true_term_grids, pred_term_grids)
        ]
        term_err_max = max(g.max() for g in term_err_grids)
        term_err_levels = _contour_levels_from_zero(term_err_max)

    n_dyn_rows = 3 * n_obs_dims
    n_rows = (7 if predict_reward_terminated else 1) + n_dyn_rows
    n_cols = len(actions)

    FONT = "Arial"
    FS_TITLE = 11
    FS_LABEL = 10
    FS_TICK = 9
    FS_ROW = 10

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT, "Helvetica", "DejaVu Sans"],
            "font.size": FS_LABEL,
            "axes.titlesize": FS_TITLE,
            "axes.labelsize": FS_LABEL,
            "xtick.labelsize": FS_TICK,
            "ytick.labelsize": FS_TICK,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        fig, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=(2.4 * n_cols, 1.8 * n_rows),
            sharex=True,
            sharey=True,
            layout="constrained",
        )
        if n_cols == 1:
            axs = axs.reshape(n_rows, 1)

        y_label = env_config.y_label
        x_label = env_config.x_label
        delta_obs_labels = env_config.delta_obs_labels

        def _hat(label):
            inner = label.strip("$")
            return rf"$\widehat{{{inner}}}$"

        def _err(label):
            inner = label.strip("$")
            return rf"$|{inner} - \widehat{{{inner}}}|$"

        dyn_row = 7 if predict_reward_terminated else 1

        def _ylabel(ax, label):
            ax.set_ylabel(label, rotation=0, ha="right", labelpad=4)

        mid_col = n_cols // 2

        CBAR_WIDTH = 0.012  # fixed fraction of figure width → consistent bar widths

        def _cbar(mappable, axes, row_label):
            cb = fig.colorbar(
                mappable, ax=axes, fraction=CBAR_WIDTH, pad=0.02, location="right"
            )
            cb.ax.tick_params(labelsize=FS_TICK, width=0.6, length=2)
            cb.ax.yaxis.set_major_locator(mticker.LinearLocator(3))
            cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            for sp in cb.ax.spines.values():
                sp.set_linewidth(0.4)
            # Place row label on the middle subplot; prepend existing title with \n if present
            axes_2d = np.atleast_2d(axes)
            mid_ax = axes_2d[0, mid_col]
            existing = mid_ax.get_title()
            mid_ax.set_title(
                row_label + "\n" + existing if existing else row_label,
                fontsize=FS_ROW,
                fontweight="bold",
            )
            return cb

        for idx, act in enumerate(actions):
            # Row 0: Epistemic Uncertainty
            ax_unc = axs[0, idx]
            cf_unc = ax_unc.contourf(
                s1_axis, s2_axis, unc_grids[idx], levels=unc_levels, cmap="viridis"
            )
            if idx == 0:
                _ylabel(ax_unc, y_label)
            ax_unc.set_title(action_title(env_name, act, discrete=discrete))

            if pointer > 0:
                visited_obs = dataset.obs[:pointer]
                visited_actions = dataset.action[:pointer]
                mask = action_visit_mask(
                    env_name, visited_actions, act, discrete=discrete
                )
                visited_s1, visited_s2 = visited_coords_from_obs(
                    env_name, visited_obs[mask]
                )
                ax_unc.scatter(
                    visited_s1,
                    visited_s2,
                    color="white",
                    alpha=0.4,
                    s=1.5,
                    linewidths=0,
                )

            if predict_reward_terminated:
                # Rows 1-3: Reward
                ax_true_rew = axs[1, idx]
                cf_true_rew = ax_true_rew.contourf(
                    s1_axis,
                    s2_axis,
                    true_rew_grids[idx],
                    levels=rew_levels,
                    cmap="inferno",
                )
                if idx == 0:
                    _ylabel(ax_true_rew, y_label)

                ax_pred_rew = axs[2, idx]
                ax_pred_rew.contourf(
                    s1_axis,
                    s2_axis,
                    pred_rew_grids[idx],
                    levels=rew_levels,
                    cmap="inferno",
                )
                if idx == 0:
                    _ylabel(ax_pred_rew, y_label)

                ax_rew_err = axs[3, idx]
                cf_rew_err = ax_rew_err.contourf(
                    s1_axis,
                    s2_axis,
                    rew_err_grids[idx],
                    levels=rew_err_levels,
                    cmap="Reds",
                )
                if idx == 0:
                    _ylabel(ax_rew_err, y_label)

                # Rows 4-6: Termination
                ax_true_term = axs[4, idx]
                cf_true_term = ax_true_term.contourf(
                    s1_axis,
                    s2_axis,
                    true_term_grids[idx],
                    levels=term_levels,
                    cmap="inferno",
                )
                if idx == 0:
                    _ylabel(ax_true_term, y_label)

                ax_pred_term = axs[5, idx]
                ax_pred_term.contourf(
                    s1_axis,
                    s2_axis,
                    pred_term_grids[idx],
                    levels=term_levels,
                    cmap="inferno",
                )
                if idx == 0:
                    _ylabel(ax_pred_term, y_label)

                ax_term_err = axs[6, idx]
                cf_term_err = ax_term_err.contourf(
                    s1_axis,
                    s2_axis,
                    term_err_grids[idx],
                    levels=term_err_levels,
                    cmap="Reds",
                )
                if idx == 0:
                    _ylabel(ax_term_err, y_label)

            # Dynamics rows — one group of 3 rows per obs dim
            cf_true_dyn_per_dim = []
            cf_dyn_err_per_dim = []
            for d in range(n_obs_dims):
                row_base = dyn_row + 3 * d
                ax_true = axs[row_base, idx]
                cf_true = ax_true.contourf(
                    s1_axis,
                    s2_axis,
                    true_dyn_grids[idx][d],
                    levels=dyn_levels_per_dim[d],
                    cmap="coolwarm",
                )
                cf_true_dyn_per_dim.append(cf_true)
                if idx == 0:
                    _ylabel(ax_true, y_label)

                ax_pred = axs[row_base + 1, idx]
                ax_pred.contourf(
                    s1_axis,
                    s2_axis,
                    pred_dyn_grids[idx][d],
                    levels=dyn_levels_per_dim[d],
                    cmap="coolwarm",
                )
                if idx == 0:
                    _ylabel(ax_pred, y_label)

                ax_err = axs[row_base + 2, idx]
                cf_err = ax_err.contourf(
                    s1_axis,
                    s2_axis,
                    dyn_err_grids_per_dim[d][idx],
                    levels=dyn_err_levels_per_dim[d],
                    cmap="Reds",
                )
                cf_dyn_err_per_dim.append(cf_err)
                if idx == 0:
                    _ylabel(ax_err, y_label)
                ax_err.set_xlabel(x_label)

            # Suppress interior tick labels (sharex/sharey keeps scales aligned)
            def _fmt(x, _):
                return f"{int(x)}" if x == int(x) else f"{x:.2f}"

            for ax in axs[:, idx]:
                ax.tick_params(which="both", direction="in")
                ax.xaxis.set_major_locator(mticker.LinearLocator(3))
                ax.yaxis.set_major_locator(mticker.LinearLocator(3))
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt))
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt))
                if idx > 0:
                    ax.yaxis.set_tick_params(labelleft=False)
                if ax not in axs[-1, :]:
                    ax.xaxis.set_tick_params(labelbottom=False)

        # Colorbars — one pass after loop so closures are resolved
        _cbar(cf_unc, axs[0, :], "Epistemic Uncertainty")
        if predict_reward_terminated:
            _cbar(cf_true_rew, axs[1:3, :], "Reward")
            _cbar(cf_rew_err, axs[3, :], "|Reward error|")
            _cbar(cf_true_term, axs[4:6, :], "Termination prob.")
            _cbar(cf_term_err, axs[6, :], "|Term. error|")
        for d in range(n_obs_dims):
            row_base = dyn_row + 3 * d
            lbl = delta_obs_labels[d]
            _cbar(cf_true_dyn_per_dim[d], axs[row_base : row_base + 2, :], lbl)
            axs[row_base + 1, mid_col].set_title(
                _hat(lbl), fontsize=FS_ROW, fontweight="bold"
            )
            _cbar(cf_dyn_err_per_dim[d], axs[row_base + 2, :], _err(lbl))

        os.makedirs(run_dir, exist_ok=True)
        fig_path = os.path.join(run_dir, f"uncertainty_{j:04d}.png")
        plt.savefig(fig_path, bbox_inches="tight", dpi=300)
        print(f"Saved uncertainty plot to {fig_path}")
        plt.close()


def _obs_from_area(area, model_obs_dim: int, time: float = 0.0):
    """Pad log-area with a constant day when the model/policy obs includes time."""
    area = jnp.asarray(area)
    if area.ndim == 1:
        area = area[:, None]
    if model_obs_dim <= 1:
        return area
    t = jnp.full((area.shape[0], 1), float(time), dtype=area.dtype)
    return jnp.concatenate([area, t], axis=-1)


def _policy_mean_action(
    train_state,
    area_min: float,
    area_max: float,
    act_low: float,
    act_high: float,
    num_grid: int = 200,
    time: float = 0.0,
    obs_dim: int = 1,
):
    """Return ``(areas, mean_actions)`` from ``pi.mean()`` over a log-area grid."""
    area_axis = jnp.linspace(area_min, area_max, num_grid)
    obs_grid = _obs_from_area(area_axis, obs_dim, time=time)
    network, _, normalize_vec_obs, _ = train_state
    normalize_vec_obs.eval()
    pi, _ = network(normalize_vec_obs(obs_grid))
    actions = np.clip(np.asarray(pi.mean())[:, 0], act_low, act_high)
    return np.asarray(area_axis), actions


def _quiver_traj(
    ax,
    traj,
    color,
    label,
    mean_action=None,
    day: int | None = None,
    mean_only: bool = False,
    lw: float = 1.6,
):
    """Overlay rollout trajectories on a (log-area, intensity) axes.

    Individual episodes are drawn as thin translucent arrows.  When
    ``mean_action=(areas, actions)`` is given, that policy mean-action curve is
    drawn bold on top; otherwise the empirical mean trajectory is used.

    If ``day`` is set and ``traj.obs`` includes a time dim, only steps with
    ``round(time) == day`` are shown (as a scatter — one step per day per
    episode), matching a day-faceted reward panel. Set ``mean_only`` to skip
    episode overlays and draw only the mean-action curve.
    """
    areas = np.asarray(traj.obs[..., 0])  # (T, E)
    acts = np.asarray(traj.action[..., 0])  # (T, E)
    E = areas.shape[1]
    has_time = traj.obs.shape[-1] > 1

    if not mean_only:
        if day is not None and has_time:
            times = np.asarray(traj.obs[..., 1])
            mask = np.isclose(times, float(day), atol=0.5)
            ax.scatter(
                areas[mask],
                acts[mask],
                color=color,
                alpha=0.22,
                s=4,
                zorder=4,
                linewidths=0,
                rasterized=True,
            )
        else:
            for e in range(E):
                x, y = areas[:, e], acts[:, e]
                ax.quiver(
                    x[:-1],
                    y[:-1],
                    np.diff(x),
                    np.diff(y),
                    color=color,
                    alpha=0.2,
                    scale_units="xy",
                    angles="xy",
                    scale=1,
                    width=0.0025,
                    headwidth=3,
                    headlength=4,
                    zorder=4,
                )

    if mean_action is not None:
        x, y = mean_action
    elif day is not None and has_time:
        ax.plot([], [], color=color, alpha=1.0, lw=lw, label=label, zorder=6)
        return
    else:
        x = areas.mean(axis=1)
        y = acts.mean(axis=1)
    ax.plot(
        x,
        y,
        color=color,
        alpha=1.0,
        lw=lw,
        label=label,
        zorder=6,
        solid_capstyle="round",
    )


def plot_area_action_uncertainty(
    model,
    area_min: float,
    area_max: float,
    act_low: float,
    act_high: float,
    dataset,
    pointer: int,
    rng,
    j: int,
    run_dir: str,
    explore_visits=None,
    exploit_visits=None,
    explore_traj=None,
    exploit_traj=None,
    explore_mean_action=None,
    exploit_mean_action=None,
    num_grid: int = 50,
    num_samples: int = 10,
    bonus: str = "std",
) -> None:
    """Plot epistemic uncertainty and predicted growth over (log-area, intensity) space.

    Parameters
    ----------
    model:
        A single (non-batched) :class:`models.WorldModel`.
    area_min / area_max:
        Observed log-area bounds from the offline dataset.
    act_low / act_high:
        Scalar action bounds (for the 1-D intensity action).
    dataset:
        Offline :class:`ppo.Transition` pytree (shape ``(N, ...)``).
    pointer:
        Number of valid dataset entries; used for the data scatter overlay.
    rng:
        JAX random key for posterior index sampling.
    j:
        Rollout iteration index (included in the filename).
    run_dir:
        Directory where the PNG is saved.
    explore_visits / exploit_visits:
        Optional ``(areas, actions)`` tuples (both 1-D arrays) of ``(log-area,
        intensity)`` pairs visited by the explore / exploit agent respectively.
        When provided, they are overlaid as scatter plots for comparison.
    num_grid:
        Resolution of the 2-D grid (``num_grid × num_grid`` points).
    num_samples:
        Number of posterior samples for uncertainty estimation.
    bonus:
        ``"std"`` or ``"eig"`` — passed to :meth:`~models.WorldModel.uncertainty`.
    """
    area_axis = jnp.linspace(area_min, area_max, num_grid)
    act_axis = jnp.linspace(act_low, act_high, num_grid)
    area_grid, act_grid = jnp.meshgrid(area_axis, act_axis)  # (G, G) each

    area_flat = area_grid.flatten()[:, None]  # (G*G, 1)
    act_flat = act_grid.flatten()[:, None]  # (G*G, 1)
    obs_flat = _obs_from_area(area_flat, model.obs_dim, time=0.0)

    x_grid = model.build_input(obs_flat, act_flat)  # (G*G, in_features)
    x_grid_norm = model.normalize_input(x_grid)  # (G*G, in_features)

    rng, subkey = jax.random.split(rng)
    z = model.sample_index(subkey, num_samples)  # (S, index_dim)

    samples = model.batch_predict_samples(x_grid_norm, z)  # (S, G*G, out_dim)

    unc_flat = model.batch_uncertainty(x_grid_norm, z, bonus)  # (G*G,)
    unc_grid = unc_flat.reshape(num_grid, num_grid)

    mean_y_flat = jax.vmap(model.predict_mean)(x_grid_norm)  # (G*G, out_dim)
    growth_flat = model.denormalize_delta_obs(mean_y_flat[..., : model.dyn_dim])[
        ..., 0
    ]  # (G*G,)
    growth_grid = growth_flat.reshape(num_grid, num_grid)

    unc_levels = _contour_levels(unc_flat.min(), unc_flat.max())
    growth_levels = _contour_levels(growth_flat.min(), growth_flat.max())

    n_cols = 2
    fig, axs = plt.subplots(1, n_cols, figsize=(12, 5))

    # — Left panel: uncertainty —
    ax_unc = axs[0]
    cf_unc = ax_unc.contourf(
        area_axis, act_axis, unc_grid, levels=unc_levels, cmap="viridis"
    )
    fig.colorbar(cf_unc, ax=ax_unc).set_label(f"Uncertainty ({bonus})")
    ax_unc.set_xlabel("log(Area)")
    ax_unc.set_ylabel("Intensity")
    ax_unc.set_title("Epistemic Uncertainty")

    # Dataset scatter overlay
    if pointer > 0:
        d_area = jnp.asarray(dataset.obs[:pointer, 0])
        d_act = jnp.asarray(dataset.action[:pointer, 0])
        ax_unc.scatter(d_area, d_act, color="white", alpha=0.3, s=4, label="Dataset")

    # — Right panel: predicted growth —
    ax_growth = axs[1]
    cf_growth = ax_growth.contourf(
        area_axis, act_axis, growth_grid, levels=growth_levels, cmap="RdYlGn"
    )
    fig.colorbar(cf_growth, ax=ax_growth).set_label("Predicted Δlog(area)")
    ax_growth.set_xlabel("log(Area)")
    ax_growth.set_ylabel("Intensity")
    ax_growth.set_title("Predicted Growth")

    # Agent trajectory overlays
    if explore_visits is not None:
        exp_areas, exp_acts = explore_visits
        ax_unc.scatter(
            exp_areas, exp_acts, color="cyan", alpha=0.5, s=8, label="Explore"
        )
        ax_growth.scatter(
            exp_areas, exp_acts, color="cyan", alpha=0.5, s=8, label="Explore"
        )

    if exploit_visits is not None:
        expt_areas, expt_acts = exploit_visits
        ax_unc.scatter(
            expt_areas, expt_acts, color="magenta", alpha=0.5, s=8, label="Exploit"
        )
        ax_growth.scatter(
            expt_areas, expt_acts, color="magenta", alpha=0.5, s=8, label="Exploit"
        )

    for ax in (ax_unc, ax_growth):
        if explore_traj is not None:
            _quiver_traj(
                ax,
                explore_traj,
                color="cyan",
                label="Explore",
                mean_action=explore_mean_action,
            )
        if exploit_traj is not None:
            _quiver_traj(
                ax,
                exploit_traj,
                color="magenta",
                label="Exploit",
                mean_action=exploit_mean_action,
            )

    for ax in axs:
        ax.legend(loc="upper right", markerscale=3, fontsize=7)

    fig.suptitle(f"Plant Model: Area × Intensity (Iteration {j})", fontsize=13)
    fig.tight_layout()
    os.makedirs(run_dir, exist_ok=True)
    fig_path = os.path.join(run_dir, f"area_action_uncertainty_{j:04d}.png")
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    print(f"Saved area×action uncertainty plot to {fig_path}")
    plt.close()


def plot_policy_rollouts(
    explore_traj,
    exploit_traj,
    j: int,
    run_dir: str,
    area_min: float | None = None,
    area_max: float | None = None,
    act_low: float | None = None,
    act_high: float | None = None,
) -> None:
    """Plot area / intensity / reward trajectories for explore vs exploit policies.

    Each subplot shows individual episode traces (thin, translucent) plus a
    bold mean line and ±1σ shaded band.

    Parameters
    ----------
    explore_traj / exploit_traj:
        :class:`ppo.Transition` pytrees with leaves shaped ``(T, num_episodes, ...)``.
    j:
        Outer-loop iteration index (used in the filename).
    run_dir:
        Directory where the PNG is saved.
    area_min / area_max:
        Optional y-axis bounds for the log-area panels.
    act_low / act_high:
        Optional y-axis bounds for the intensity panels.
    """
    areas_exp = np.asarray(explore_traj.obs[..., 0])  # (T, E)
    acts_exp = np.asarray(explore_traj.action[..., 0])  # (T, E)

    areas_expt = np.asarray(exploit_traj.obs[..., 0])
    acts_expt = np.asarray(exploit_traj.action[..., 0])

    T = areas_exp.shape[0]
    timesteps = np.arange(T)

    COLOR_EXP = "cyan"
    COLOR_EXPT = "magenta"

    def _iqm(values, axis):
        """Interquartile mean: trim the bottom/top 25% (by value) and average the rest."""
        n = values.shape[axis]
        k = int(np.floor(0.25 * n))
        sorted_vals = np.sort(values, axis=axis)
        if k == 0:
            return sorted_vals.mean(axis=axis)
        sl = [slice(None)] * values.ndim
        sl[axis] = slice(k, n - k)
        return sorted_vals[tuple(sl)].mean(axis=axis)

    def _plot_traces(ax, values, color, ylabel, ylim=None, estimator="mean"):
        E = values.shape[1]
        for e in range(E):
            ax.plot(timesteps, values[:, e], color=color, alpha=0.12, lw=0.8)
        if estimator == "iqm":
            center = _iqm(values, axis=1)
            lo, hi = np.percentile(values, [25, 75], axis=1)
            center_label, band_label = "IQM", "IQR"
        else:
            center = values.mean(axis=1)
            std = values.std(axis=1)
            lo, hi = center - std, center + std
            center_label, band_label = "mean", "±1σ"
        ax.plot(timesteps, center, color=color, lw=2.0, label=center_label)
        ax.fill_between(timesteps, lo, hi, color=color, alpha=0.22, label=band_label)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)

    fig, axs = plt.subplots(3, 2, figsize=(11, 9), sharex=True)

    area_ylim = (
        (area_min, area_max) if area_min is not None and area_max is not None else None
    )
    act_ylim = (
        (act_low, act_high) if act_low is not None and act_high is not None else None
    )

    # Row 0: log(Area)
    axs[0, 0].set_title("Explore", color=COLOR_EXP, fontsize=12, fontweight="bold")
    axs[0, 1].set_title("Exploit", color=COLOR_EXPT, fontsize=12, fontweight="bold")
    _plot_traces(
        axs[0, 0], areas_exp, COLOR_EXP, "log(Area)", ylim=area_ylim, estimator="iqm"
    )
    _plot_traces(
        axs[0, 1], areas_expt, COLOR_EXPT, "log(Area)", ylim=area_ylim, estimator="iqm"
    )

    # Row 1: Intensity
    _plot_traces(axs[1, 0], acts_exp, COLOR_EXP, "Intensity", ylim=act_ylim)
    _plot_traces(axs[1, 1], acts_expt, COLOR_EXPT, "Intensity", ylim=act_ylim)

    # Row 2: Δlog(area) per step — same metric for both agents so they're comparable
    growth_exp = np.diff(areas_exp, axis=0, prepend=areas_exp[:1])  # (T, E)
    growth_expt = np.diff(areas_expt, axis=0, prepend=areas_expt[:1])
    growth_all = np.concatenate(
        [growth_exp[1:], growth_expt[1:]]
    )  # skip step-0 artifact
    growth_lim = (float(growth_all.min()), float(growth_all.max()))
    _plot_traces(axs[2, 0], growth_exp, COLOR_EXP, "Δlog(area) / step", ylim=growth_lim)
    _plot_traces(
        axs[2, 1], growth_expt, COLOR_EXPT, "Δlog(area) / step", ylim=growth_lim
    )
    for ax in axs[2, :]:
        ax.set_xlabel("Step")

    fig.suptitle(f"Policy Rollouts in World Model (Iteration {j})", fontsize=13)
    fig.tight_layout()
    os.makedirs(run_dir, exist_ok=True)
    fig_path = os.path.join(run_dir, f"policy_rollouts_{j:04d}.png")
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    print(f"Saved policy rollout trajectories to {fig_path}")
    plt.close()


def plot_reward_landscape(
    model,
    reward_fn,
    area_min: float,
    area_max: float,
    act_low: float,
    act_high: float,
    j: int,
    run_dir: str,
    num_grid: int = 50,
    explore_traj=None,
    exploit_traj=None,
    explore_mean_action=None,
    exploit_mean_action=None,
    explore_train_state=None,
    exploit_train_state=None,
    obs_dim: int = 1,
    reward_days: tuple[int, ...] | None = None,
    gate_log_area_fn=None,
    compose_next_obs_fn=None,
    rng=None,
    bonus: str = "std",
    num_samples: int = 10,
    dataset=None,
) -> None:
    """2-D contour plot of model-predicted uncertainty, growth, and total reward.

    X-axis: log(Area), Y-axis: intensity. Same orientation as the uncertainty plot.

    The figure always shows uncertainty (explore policy only), growth, and total
    reward each as a single row of panels. When ``reward_days`` is set
    (day-dependent masked rewards), each row facets across days 0–13 and policy
    overlays on a day's panel use that day's mean action and only trajectory
    steps from that day; otherwise the figure has a single column. ``gate_log_area_fn(day)``
    may return a vertical gate line in log-area (called with day 0 when
    ``reward_days`` is unset). Uncertainty panels also scatter offline dataset
    ``(log-area, intensity)`` points for that day when ``dataset`` is set.
    """
    area_axis = jnp.linspace(area_min, area_max, num_grid)
    act_axis = jnp.linspace(act_low, act_high, num_grid)
    area_mg, act_mg = jnp.meshgrid(area_axis, act_axis, indexing="ij")  # (G, G)

    area_flat = area_mg.flatten()[:, None]  # (G*G, 1)
    act_flat = act_mg.flatten()[:, None]  # (G*G, 1)

    def _predict_growth(time: float | None):
        """Model-predicted Δlog-area, next area and reward at a fixed calendar day.

        ``model_reward`` is the model's own reward head (``None`` unless the model
        was built with one) — the reward source for reward modes with no closed
        form, e.g. the per-(experiment, day) differential reward.
        """
        t = 0.0 if time is None else float(time)
        obs_flat = _obs_from_area(area_flat, model.obs_dim, time=t)
        x_flat = model.build_input(obs_flat, act_flat)
        x_flat_norm = model.normalize_input(x_flat)
        y_flat = jax.vmap(model.predict_mean)(x_flat_norm)
        dyn_flat = model.denormalize_delta_obs(y_flat[..., : model.dyn_dim])
        # The model output is not always the observation delta: the plant
        # differential setup predicts a growth advantage that the env turns into
        # an area by adding the control's own growth back. Use the env's mapping
        # so the plotted growth and the reward agree with the rollout.
        if compose_next_obs_fn is not None:
            next_obs_full = compose_next_obs_fn(obs_flat, dyn_flat)
            next_obs = next_obs_full[..., :1]
        else:
            next_obs = area_flat + dyn_flat[..., :1]
        growth = (next_obs - area_flat)[..., 0]
        model_reward = (
            model.denormalize_reward(y_flat[..., model.reward_index])
            if model.predict_reward
            else None
        )
        return growth, next_obs, model_reward

    days = reward_days if reward_days else (None,)
    growth_by_day: list[tuple[int | None, np.ndarray]] = []
    total_by_day: list[tuple[int | None, np.ndarray]] = []
    unc_by_day: list[tuple[int | None, np.ndarray]] = []

    # Shared posterior indices across days for comparable uncertainty maps.
    if rng is None:
        rng = jax.random.PRNGKey(0)
    rng, z_key = jax.random.split(rng)
    z = model.sample_index(z_key, num_samples)

    for day in days:
        t = 0.0 if day is None else float(day)
        obs_flat = _obs_from_area(area_flat, model.obs_dim, time=t)
        x_flat = model.build_input(obs_flat, act_flat)
        x_flat_norm = model.normalize_input(x_flat)
        unc_flat = model.batch_uncertainty(x_flat_norm, z, bonus)
        unc_by_day.append((day, np.asarray(unc_flat.reshape(num_grid, num_grid))))

        growth_flat, next_obs_flat, model_reward_flat = _predict_growth(day)
        growth_by_day.append(
            (day, np.asarray(growth_flat.reshape(num_grid, num_grid)))
        )
        if reward_fn is None and model_reward_flat is not None:
            total_flat = model_reward_flat
        elif reward_fn is None:
            total_flat = growth_flat
        elif day is None:
            total_flat = reward_fn(area_flat, act_flat, next_obs_flat)
        else:
            total_flat = reward_fn(area_flat, act_flat, next_obs_flat, time=day)
        total_by_day.append(
            (day, np.asarray(total_flat.reshape(num_grid, num_grid)))
        )

    area_axis_np = np.asarray(area_axis)
    act_axis_np = np.asarray(act_axis)

    # Explore orange / Exploit blue (swapped for heatmap contrast).
    EXPLORE_COLOR = "#FF5A00"
    EXPLOIT_COLOR = "#00B0F0"

    def _mean_action_for_day(train_state, day: int | None):
        if train_state is None:
            return None
        return _policy_mean_action(
            train_state,
            area_min,
            area_max,
            act_low,
            act_high,
            time=float(day if day is not None else 0),
            obs_dim=obs_dim,
        )

    def _style_ax(ax, *, xlabel=False, ylabel=False, xticklabels=False, yticklabels=False):
        ax.tick_params(
            axis="both",
            which="major",
            length=2.5,
            pad=1.5,
            labelbottom=xticklabels,
            labelleft=yticklabels,
        )
        if xlabel:
            ax.set_xlabel("log(Area)")
        else:
            ax.set_xlabel("")
        if ylabel:
            ax.set_ylabel("Intensity")
        else:
            ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    def _draw_panel(
        ax,
        grid,
        cmap,
        gate_x=None,
        levels=None,
        norm=None,
        day: int | None = None,
        overlay_full_traj: bool = False,
        show_explore: bool = True,
        show_exploit: bool = True,
        mean_only: bool = True,
        rasterize: bool = True,
    ):
        if levels is None:
            levels = _contour_levels(float(grid.min()), float(grid.max()))
        cf = ax.contourf(
            area_axis_np,
            act_axis_np,
            grid.T,
            levels=levels,
            cmap=cmap,
            norm=norm,
            vmin=norm.vmin if norm is not None else None,
            vmax=norm.vmax if norm is not None else None,
        )
        if rasterize:
            try:
                for coll in cf.collections:
                    coll.set_rasterized(True)
            except AttributeError:
                pass
        if gate_x is not None and area_min <= gate_x <= area_max:
            ax.axvline(gate_x, color="0.15", ls="--", lw=0.7, alpha=0.85, zorder=5)

        if day is not None and not overlay_full_traj:
            exp_ma = _mean_action_for_day(explore_train_state, day)
            expt_ma = _mean_action_for_day(exploit_train_state, day)
            if exp_ma is None:
                exp_ma = explore_mean_action
            if expt_ma is None:
                expt_ma = exploit_mean_action
            traj_day = day
        else:
            exp_ma = explore_mean_action
            expt_ma = exploit_mean_action
            traj_day = None

        if show_explore and explore_traj is not None:
            _quiver_traj(
                ax,
                explore_traj,
                color=EXPLORE_COLOR,
                label="Explore",
                mean_action=exp_ma,
                day=traj_day,
                mean_only=mean_only or (traj_day is not None),
                lw=1.8,
            )
        if show_exploit and exploit_traj is not None:
            _quiver_traj(
                ax,
                exploit_traj,
                color=EXPLOIT_COLOR,
                label="Exploit",
                mean_action=expt_ma,
                day=traj_day,
                mean_only=mean_only or (traj_day is not None),
                lw=1.8,
            )
        ax.set_xlim(area_min, area_max)
        ax.set_ylim(act_low, act_high)
        return cf

    n_day_cols = len(days)  # one panel per day (or a single panel when day-invariant)
    nature_rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "axes.labelpad": 2,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 6,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "figure.dpi": 150,
    }
    # One column per day (14 at most) × 3 metric rows; a single column when day-invariant.
    fig_w, fig_h = max(3.6, 1.05 * n_day_cols + 2.2), 5.2
    # Left/right margins hold fixed-size text (row labels, colorbars) regardless of
    # fig_w, so reserve them in absolute inches rather than a fixed figure fraction.
    left_margin_in, right_margin_in = 0.62, 0.95
    left_frac = left_margin_in / fig_w
    right_frac = 1.0 - right_margin_in / fig_w
    with plt.rc_context(nature_rc):
        fig = plt.figure(figsize=(fig_w, fig_h))
        n_rows = 3
        gs = fig.add_gridspec(
            n_rows,
            n_day_cols + 1,
            width_ratios=[1.0] * n_day_cols + [0.06],
            height_ratios=[1.0] * n_rows,
            left=left_frac,
            right=right_frac,
            top=0.90,
            bottom=0.10,
            wspace=0.08,
            hspace=0.35,
        )
        axes = np.empty((n_rows, n_day_cols), dtype=object)
        for r in range(n_rows):
            for c in range(n_day_cols):
                axes[r, c] = fig.add_subplot(gs[r, c])

        def _block_scales(by_day):
            all_vals = np.concatenate([g.ravel() for _, g in by_day])
            vmin, vmax = float(all_vals.min()), float(all_vals.max())
            return _contour_levels(vmin, vmax), plt.Normalize(vmin=vmin, vmax=vmax)

        def _fill_day_block(
            row,
            by_day,
            cmap,
            *,
            show_explore,
            show_exploit,
            block_label,
            show_data: bool = False,
        ):
            levels, norm = _block_scales(by_day)
            cf = None
            data_area = data_act = data_time = None
            if show_data and dataset is not None:
                data_area = np.asarray(dataset.obs[..., 0]).reshape(-1)
                data_act = np.asarray(dataset.action[..., 0]).reshape(-1)
                if dataset.obs.shape[-1] > 1:
                    data_time = np.asarray(dataset.obs[..., 1]).reshape(-1)
            for i, (day, grid) in enumerate(by_day):
                ax = axes[row, i]
                gate = (
                    gate_log_area_fn(day if day is not None else 0)
                    if gate_log_area_fn is not None
                    else None
                )
                cf = _draw_panel(
                    ax,
                    grid,
                    cmap,
                    gate_x=gate,
                    levels=levels,
                    norm=norm,
                    day=day,
                    show_explore=show_explore,
                    show_exploit=show_exploit,
                    mean_only=True,
                )
                if data_area is not None:
                    if data_time is not None and day is not None:
                        mask = np.isclose(data_time, float(day), atol=0.5)
                        xa, ya = data_area[mask], data_act[mask]
                    else:
                        xa, ya = data_area, data_act
                    ax.scatter(
                        xa,
                        ya,
                        c="white",
                        s=2.5,
                        alpha=0.35,
                        linewidths=0,
                        zorder=3,
                        rasterized=True,
                        label="Data" if i == 0 else None,
                    )
                if day is not None:
                    ax.set_title(str(day), fontsize=7, pad=1.5)
                _style_ax(
                    ax,
                    xlabel=(i == n_day_cols // 2),
                    ylabel=(i == 0),
                    xticklabels=True,
                    yticklabels=(i == 0),
                )
                if i == n_day_cols // 2:
                    ax.set_xlabel("log(Area)", labelpad=2)
                if i == 0:
                    ax.set_ylabel("Intensity", labelpad=2)
                ax.tick_params(axis="x", which="major", pad=1.5, labelsize=5)
                if i == 0:
                    ax.tick_params(axis="y", which="major", pad=2)
            pos = axes[row, 0].get_position()
            fig.text(
                0.08 / fig_w,
                0.5 * (pos.y0 + pos.y1),
                block_label,
                rotation=90,
                va="center",
                ha="center",
                fontsize=7.5,
            )
            return cf, norm

        cf_unc, _ = _fill_day_block(
            0,
            unc_by_day,
            "cividis",
            show_explore=True,
            show_exploit=False,
            block_label="Uncertainty",
            show_data=True,
        )
        cf_growth, _ = _fill_day_block(
            1,
            growth_by_day,
            "RdYlGn",
            show_explore=False,
            show_exploit=False,
            block_label="Growth (Δlog area)",
        )
        cf_reward, _ = _fill_day_block(
            2,
            total_by_day,
            "RdYlGn",
            show_explore=False,
            show_exploit=True,
            block_label="Total reward",
        )

        def _cax(r):
            return fig.add_subplot(gs[r, -1])

        from matplotlib.ticker import FormatStrFormatter, MaxNLocator
        from matplotlib.lines import Line2D

        def _format_cbar(cbar):
            cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2g"))
            cbar.ax.tick_params(labelsize=6, length=2, width=0.5)
            cbar.outline.set_linewidth(0.5)

        for cax, cf, label in (
            (_cax(0), cf_unc, f"Uncertainty ({bonus})"),
            (_cax(1), cf_growth, "Δlog area"),
            (_cax(2), cf_reward, "Total reward"),
        ):
            cbar = fig.colorbar(cf, cax=cax)
            cbar.set_label(label, fontsize=7)
            _format_cbar(cbar)

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="white",
                markeredgecolor="0.4",
                markersize=4,
                linestyle="None",
                label="Data",
            ),
            Line2D([0], [0], color=EXPLORE_COLOR, lw=1.8, label="Explore"),
            Line2D([0], [0], color=EXPLOIT_COLOR, lw=1.8, label="Exploit"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(0.93, 0.995),
            frameon=False,
            fontsize=7,
            ncol=3,
            handlelength=1.8,
        )

        if n_day_cols > 1:
            fig.text(
                0.49,
                axes[0, 0].get_position().y1 + 0.02,
                "Day →",
                ha="center",
                va="bottom",
                fontsize=7,
                color="0.35",
            )

        os.makedirs(run_dir, exist_ok=True)
        fig_path = os.path.join(run_dir, f"reward_landscape_{j:04d}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
        pdf_path = os.path.join(run_dir, f"reward_landscape_{j:04d}.pdf")
        plt.savefig(pdf_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
        print(f"Saved reward landscape to {fig_path}")
        print(f"Saved reward landscape PDF to {pdf_path}")
        plt.close()


def _collect_agent_visits(
    train_state, model_env, model_env_params, rollout_config, rng
):
    """Roll out a single-seed policy in the model env; return (areas, actions) arrays."""
    rng, key_reset, key_roll = jax.random.split(rng, 3)
    reset_keys = jax.random.split(key_reset, rollout_config["NUM_ENVS"])
    obsv, env_state = model_env.reset(reset_keys, model_env_params)
    rollout_fn = make_rollout(
        rollout_config, model_env, model_env_params, training=False
    )
    _, traj = rollout_fn((train_state, env_state, obsv, key_roll))
    areas = jnp.asarray(traj.obs[..., 0]).flatten()
    acts = jnp.asarray(traj.action[..., 0]).flatten()
    return areas, acts


def _collect_policy_trajectories(
    train_state,
    model_env,
    model_env_params,
    ppo_config,
    rng,
    num_episodes: int = 20,
    num_steps: int = 14,
):
    """Roll out ``num_episodes`` independent episodes; return trajectory shaped ``(num_steps, num_episodes, ...)``."""
    traj_cfg = {**ppo_config, "NUM_ENVS": num_episodes, "NUM_STEPS": num_steps}
    rng, key_reset, key_roll = jax.random.split(rng, 3)
    reset_keys = jax.random.split(key_reset, num_episodes)
    obsv, env_state = model_env.reset(reset_keys, model_env_params)
    rollout_fn = make_rollout(traj_cfg, model_env, model_env_params, training=False)
    _, traj = rollout_fn((train_state, env_state, obsv, key_roll))
    return traj


def plot_policy_action(
    explore_train_state,
    exploit_train_state,
    area_min: float,
    area_max: float,
    act_low: float,
    act_high: float,
    j: int,
    run_dir: str,
    num_grid: int = 200,
    obs_dim: int = 1,
    policy_days: tuple[int, ...] | None = None,
) -> None:
    """Plot mean action chosen by explore vs exploit policy across log(area).

    When ``obs_dim == 2`` (masked rewards with time), draw one curve per day in
    ``policy_days`` so time-dependence is visible.
    """
    days = policy_days if (obs_dim > 1 and policy_days) else (0,)
    fig, ax = plt.subplots(figsize=(8, 4))
    cmap_exp = plt.cm.cool(np.linspace(0.3, 0.9, len(days)))
    cmap_expt = plt.cm.autumn(np.linspace(0.2, 0.8, len(days)))
    for i, day in enumerate(days):
        areas_np, explore_actions = _policy_mean_action(
            explore_train_state,
            area_min,
            area_max,
            act_low,
            act_high,
            num_grid,
            time=float(day),
            obs_dim=obs_dim,
        )
        _, exploit_actions = _policy_mean_action(
            exploit_train_state,
            area_min,
            area_max,
            act_low,
            act_high,
            num_grid,
            time=float(day),
            obs_dim=obs_dim,
        )
        day_lab = f" day {day}" if len(days) > 1 else ""
        ax.plot(
            areas_np,
            explore_actions,
            color=cmap_exp[i],
            lw=2,
            label=f"Explore{day_lab}",
        )
        ax.plot(
            areas_np,
            exploit_actions,
            color=cmap_expt[i],
            lw=2,
            ls="--",
            label=f"Exploit{day_lab}",
        )
    ax.axhline(act_low, color="gray", lw=0.8, linestyle=":")
    ax.axhline(act_high, color="gray", lw=0.8, linestyle=":")
    ax.set_xlabel("log(Area)")
    ax.set_ylabel("Mean action (intensity)")
    ax.set_ylim(
        act_low - 0.05 * (act_high - act_low), act_high + 0.05 * (act_high - act_low)
    )
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, linestyle=":", alpha=0.5)
    title = f"Policy Action vs log(Area) (Iteration {j})"
    if len(days) > 1:
        title += " — by day"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    os.makedirs(run_dir, exist_ok=True)
    fig_path = os.path.join(run_dir, f"policy_action_{j:04d}.png")
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    print(f"Saved policy action plot to {fig_path}")
    plt.close()


def plot_offline_visualization(
    models,
    explore_train_state,
    eval_train_state,
    model_env,
    config,
    num_steps,
    rng,
    dataset,
    env_params,
    j,
    run_dir,
    explore_bonus,
    reward_fn=None,
    reward_days: tuple[int, ...] | None = None,
    gate_log_area_fn=None,
    compose_next_obs_fn=None,
):
    """Collect agent visits / trajectories and emit all three offline plots (seed 0)."""
    area_min = float(env_params.area_min)
    area_max = float(env_params.area_max)
    act_low = float(env_params.act_low[0])
    act_high = float(env_params.act_high[0])
    N = dataset.obs.shape[0]

    model0 = unstack_train_state(models, 0)
    explore_ts0 = unstack_train_state(explore_train_state, 0)
    exploit_ts0 = unstack_train_state(eval_train_state, 0)

    model0_env_params = model_env.default_params.replace(
        model=model0,
        alpha=jnp.float32(1.0),
        beta=jnp.float32(0.0),
    )
    if model_env.reset_source != "env":
        weights = reset_weights(
            dataset.terminated, dataset.truncated, N, model_env.reset_source
        )
        model0_env_params = model0_env_params.replace(
            init_obs=dataset.obs, init_weights=weights
        )
    rollout_cfg = {
        **config,
        "NUM_ENVS": 64,
        "TOTAL_TIMESTEPS": num_steps * 64,
        "NUM_UPDATES": 1,
    }

    rng, rng_exp, rng_expt = jax.random.split(rng, 3)
    explore_visits = _collect_agent_visits(
        explore_ts0, model_env, model0_env_params, rollout_cfg, rng_exp
    )
    exploit_visits = _collect_agent_visits(
        exploit_ts0, model_env, model0_env_params, rollout_cfg, rng_expt
    )

    rng, rng_traj_exp, rng_traj_expt = jax.random.split(rng, 3)
    explore_traj = _collect_policy_trajectories(
        explore_ts0,
        model_env,
        model0_env_params,
        config,
        rng_traj_exp,
        num_episodes=50,
        num_steps=num_steps,
    )
    exploit_traj = _collect_policy_trajectories(
        exploit_ts0,
        model_env,
        model0_env_params,
        config,
        rng_traj_expt,
        num_episodes=50,
        num_steps=num_steps,
    )

    obs_dim = int(dataset.obs.shape[-1])
    policy_days = reward_days if obs_dim > 1 else None

    explore_mean_action = _policy_mean_action(
        explore_ts0,
        area_min,
        area_max,
        act_low,
        act_high,
        obs_dim=obs_dim,
        time=0.0,
    )
    exploit_mean_action = _policy_mean_action(
        exploit_ts0,
        area_min,
        area_max,
        act_low,
        act_high,
        obs_dim=obs_dim,
        time=0.0,
    )

    plot_area_action_uncertainty(
        model0,
        area_min,
        area_max,
        act_low,
        act_high,
        dataset,
        N,
        rng,
        j,
        run_dir,
        explore_visits=explore_visits,
        exploit_visits=exploit_visits,
        explore_traj=explore_traj,
        exploit_traj=exploit_traj,
        explore_mean_action=explore_mean_action,
        exploit_mean_action=exploit_mean_action,
        bonus=explore_bonus,
    )
    plot_policy_rollouts(
        explore_traj,
        exploit_traj,
        j,
        run_dir,
        area_min=area_min,
        area_max=area_max,
        act_low=act_low,
        act_high=act_high,
    )
    rng, rng_landscape = jax.random.split(rng)
    plot_reward_landscape(
        model0,
        reward_fn,
        area_min=area_min,
        area_max=area_max,
        act_low=act_low,
        act_high=act_high,
        j=j,
        run_dir=run_dir,
        explore_traj=explore_traj,
        exploit_traj=exploit_traj,
        explore_mean_action=explore_mean_action,
        exploit_mean_action=exploit_mean_action,
        explore_train_state=explore_ts0,
        exploit_train_state=exploit_ts0,
        obs_dim=obs_dim,
        reward_days=reward_days,
        gate_log_area_fn=gate_log_area_fn,
        compose_next_obs_fn=compose_next_obs_fn,
        rng=rng_landscape,
        bonus=explore_bonus,
        dataset=dataset,
    )
    plot_policy_action(
        explore_ts0,
        exploit_ts0,
        area_min=area_min,
        area_max=area_max,
        act_low=act_low,
        act_high=act_high,
        j=j,
        run_dir=run_dir,
        obs_dim=obs_dim,
        policy_days=policy_days,
    )
