import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import jax
import jax.numpy as jnp
import ground_truth
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
                model.denormalize_reward(mean_y[..., -2]).reshape(num_grid, num_grid)
            )
            pred_term_grids.append(
                jax.nn.sigmoid(mean_y[..., -1]).reshape(num_grid, num_grid)
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


def _quiver_traj(ax, traj, color, label):
    """Overlay rollout trajectories as quiver arrows on a (log-area, intensity) axes.

    Individual episodes are drawn as thin translucent arrows; the mean trajectory
    is drawn bold on top.
    """
    areas = np.asarray(traj.obs[..., 0])  # (T, E)
    acts = np.asarray(traj.action[..., 0])  # (T, E)
    E = areas.shape[1]

    # Individual episodes (thin, translucent)
    for e in range(E):
        x, y = areas[:, e], acts[:, e]
        ax.quiver(
            x[:-1],
            y[:-1],
            np.diff(x),
            np.diff(y),
            color=color,
            alpha=0.12,
            scale_units="xy",
            angles="xy",
            scale=1,
            width=0.003,
            headwidth=3,
            headlength=4,
            zorder=4,
        )

    # Mean trajectory (bold)
    x = areas.mean(axis=1)
    y = acts.mean(axis=1)
    ax.quiver(
        x[:-1],
        y[:-1],
        np.diff(x),
        np.diff(y),
        color=color,
        alpha=0.95,
        scale_units="xy",
        angles="xy",
        scale=1,
        width=0.006,
        headwidth=4,
        headlength=5,
        label=label,
        zorder=6,
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

    x_grid = model.build_input(area_flat, act_flat)  # (G*G, in_features)
    x_grid_norm = model.normalize_input(x_grid)  # (G*G, in_features)

    rng, subkey = jax.random.split(rng)
    z = model.sample_index(subkey, num_samples)  # (S, index_dim)

    samples = model.batch_predict_samples(x_grid_norm, z)  # (S, G*G, out_dim)

    unc_flat = model.batch_uncertainty(x_grid_norm, z, bonus)  # (G*G,)
    unc_grid = unc_flat.reshape(num_grid, num_grid)

    mean_y_flat = jax.vmap(model.predict_mean)(x_grid_norm)  # (G*G, out_dim)
    growth_flat = model.denormalize_delta_obs(mean_y_flat[..., : model.obs_dim])[
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
            exp_areas, exp_acts, color="red", alpha=0.5, s=8, label="Explore"
        )
        ax_growth.scatter(
            exp_areas, exp_acts, color="red", alpha=0.5, s=8, label="Explore"
        )

    if exploit_visits is not None:
        expt_areas, expt_acts = exploit_visits
        ax_unc.scatter(
            expt_areas, expt_acts, color="orange", alpha=0.5, s=8, label="Exploit"
        )
        ax_growth.scatter(
            expt_areas, expt_acts, color="orange", alpha=0.5, s=8, label="Exploit"
        )

    for ax in (ax_unc, ax_growth):
        if explore_traj is not None:
            _quiver_traj(ax, explore_traj, color="#4c72b0", label="Explore traj")
        if exploit_traj is not None:
            _quiver_traj(ax, exploit_traj, color="#55a868", label="Exploit traj")

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

    COLOR_EXP = "#4c72b0"  # blue
    COLOR_EXPT = "#55a868"  # green

    def _plot_traces(ax, values, color, ylabel, ylim=None):
        E = values.shape[1]
        for e in range(E):
            ax.plot(timesteps, values[:, e], color=color, alpha=0.12, lw=0.8)
        mean = values.mean(axis=1)
        std = values.std(axis=1)
        ax.plot(timesteps, mean, color=color, lw=2.0, label="mean")
        ax.fill_between(
            timesteps, mean - std, mean + std, color=color, alpha=0.22, label="±1σ"
        )
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
    _plot_traces(axs[0, 0], areas_exp, COLOR_EXP, "log(Area)", ylim=area_ylim)
    _plot_traces(axs[0, 1], areas_expt, COLOR_EXPT, "log(Area)", ylim=area_ylim)

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
) -> None:
    """2-D contour plot of model-predicted growth, energy penalty, and total reward.

    X-axis: intensity, Y-axis: log(Area).  Same orientation as the uncertainty plot.
    """
    area_axis = jnp.linspace(area_min, area_max, num_grid)
    act_axis = jnp.linspace(act_low, act_high, num_grid)
    area_mg, act_mg = jnp.meshgrid(area_axis, act_axis, indexing="ij")  # (G, G)

    area_flat = area_mg.flatten()[:, None]  # (G*G, 1)
    act_flat = act_mg.flatten()[:, None]  # (G*G, 1)

    x_flat = model.build_input(area_flat, act_flat)  # (G*G, in_features)
    x_flat_norm = model.normalize_input(x_flat)  # (G*G, in_features)
    y_flat = jax.vmap(model.predict_mean)(x_flat_norm)  # (G*G, out_dim)
    delta_flat = model.denormalize_delta_obs(y_flat[..., : model.obs_dim])  # (G*G, 1)
    next_obs_flat = area_flat + delta_flat  # (G*G, 1)
    growth_flat = delta_flat[..., 0]  # (G*G,)

    if reward_fn is not None:
        total_flat = reward_fn(area_flat, act_flat, next_obs_flat)  # (G*G,)
        penalty_flat = growth_flat - total_flat
    else:
        total_flat = growth_flat
        penalty_flat = jnp.zeros_like(growth_flat)

    # indexing="ij" → result_grid[i, k] = value at (area_axis[i], act_axis[k])
    # contourf(x, y, z) expects z[k, i], so pass grid.T
    growth_grid = np.asarray(growth_flat.reshape(num_grid, num_grid))
    penalty_grid = np.asarray(penalty_flat.reshape(num_grid, num_grid))
    total_grid = np.asarray(total_flat.reshape(num_grid, num_grid))
    area_axis_np = np.asarray(area_axis)
    act_axis_np = np.asarray(act_axis)

    titles = ["Growth (Δlog area)", "Energy penalty", "Total reward"]
    grids = [growth_grid, penalty_grid, total_grid]
    cmaps = ["RdYlGn", "RdYlGn_r", "RdYlGn"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, title, grid, cmap in zip(axes, titles, grids, cmaps):
        levels = _contour_levels(grid.min(), grid.max())
        cf = ax.contourf(area_axis_np, act_axis_np, grid.T, levels=levels, cmap=cmap)
        fig.colorbar(cf, ax=ax)
        ax.set_xlabel("log(Area)")
        ax.set_ylabel("Intensity")
        ax.set_title(title)
        if explore_traj is not None:
            _quiver_traj(ax, explore_traj, color="#4c72b0", label="Explore")
        if exploit_traj is not None:
            _quiver_traj(ax, exploit_traj, color="#55a868", label="Exploit")
        if explore_traj is not None or exploit_traj is not None:
            ax.legend(loc="upper right", fontsize=7)

    fig.suptitle(f"Reward Landscape (Iteration {j})", fontsize=13)
    fig.tight_layout()
    os.makedirs(run_dir, exist_ok=True)
    fig_path = os.path.join(run_dir, f"reward_landscape_{j:04d}.png")
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    print(f"Saved reward landscape to {fig_path}")
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
) -> None:
    """Plot mean action chosen by explore vs exploit policy across log(area) values."""
    area_axis = jnp.linspace(area_min, area_max, num_grid)
    obs_grid = area_axis[:, None]  # (G, 1)

    def query_policy(train_state):
        network, _, normalize_vec_obs, _ = train_state
        normalize_vec_obs.eval()
        obs_norm = normalize_vec_obs(obs_grid)
        pi, _ = network(obs_norm)
        return np.asarray(pi.mean())  # (G, act_dim)

    explore_actions = np.clip(query_policy(explore_train_state), act_low, act_high)
    exploit_actions = np.clip(query_policy(exploit_train_state), act_low, act_high)
    areas_np = np.asarray(area_axis)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(areas_np, explore_actions[:, 0], color="#4c72b0", lw=2, label="Explore")
    ax.plot(areas_np, exploit_actions[:, 0], color="#55a868", lw=2, label="Exploit")
    ax.axhline(act_low, color="gray", lw=0.8, linestyle=":")
    ax.axhline(act_high, color="gray", lw=0.8, linestyle=":")
    ax.set_xlabel("log(Area)")
    ax.set_ylabel("Mean action (intensity)")
    ax.set_ylim(
        act_low - 0.05 * (act_high - act_low), act_high + 0.05 * (act_high - act_low)
    )
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.suptitle(f"Policy Action vs log(Area) (Iteration {j})", fontsize=13)
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
    )
