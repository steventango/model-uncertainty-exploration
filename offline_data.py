from collections import defaultdict

import jax.numpy as jnp
import minari
import numpy as np

from ppo import Transition

AREA_INDEX = 28
# Number of day slots in the control tables (plant-data exports day 0..14).
NUM_DAYS = 15


def _control_tables(
    ctl_rows: dict[tuple[int, int], tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-day control growth and energy, averaged UNWEIGHTED over experiments.

    ``ctl_rows`` maps ``(experiment, day)`` to that experiment's control
    (constant-white Z11) growth and energy on that day.

    Unweighted rather than weighted by row count: these tables are only used to
    turn a predicted growth *advantage* back into an absolute area (see
    :meth:`plant_env.PlantEnv.compose_next_obs`), so they are a choice of *which
    batch to simulate*, not an estimate of the training mix. An equal average
    over the available controls is the most representative "typical batch".
    (When the dynamics target still contained the batch effect there was an
    argument for row-count weighting, to match the model's implicit mix; a
    de-confounded target removes that reason.)
    """
    growth = np.full(NUM_DAYS, np.nan, dtype=np.float64)
    energy = np.full(NUM_DAYS, np.nan, dtype=np.float64)
    per_day: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for (_exp, day), (g, e) in ctl_rows.items():
        if 0 <= day < NUM_DAYS and np.isfinite(g) and np.isfinite(e):
            per_day[day].append((g, e))
    for day, vals in per_day.items():
        growth[day] = np.mean([v[0] for v in vals])
        energy[day] = np.mean([v[1] for v in vals])
    # Day 0 is a reset row with no incoming transition, so it has no control
    # growth; fill from day 1 so a day-0 lookup is defined.
    for arr in (growth, energy):
        if not np.isfinite(arr[0]) and np.isfinite(arr[1]):
            arr[0] = arr[1]
    return growth, energy


def load_offline_transitions(
    dataset_id: str,
    area_index: int = AREA_INDEX,
    include_time: bool = False,
    dyn_target: str = "delta_obs",
) -> tuple:
    """Load a minari dataset.

    Args:
        dataset_id: Minari dataset id, e.g. ``"plant-data/e18e21-daily-diff-v29"``.
        area_index: Column index for ``log_clean_area`` in the observation.
        include_time: If True, append the day index as a second obs dim
            ``[log_area, day]``. Needed by day-dependent rewards (the masked
            gates, and the per-day control of the differential reward). The day
            is read from ``infos["day"]``, not from the position in the episode.
        dyn_target: What the dynamics model should be trained to predict.
            ``"delta_obs"`` (default) is the log-area delta.
            ``"advantage"`` is ``infos["area_reward_diff"]`` — the growth
            advantage over the same-batch, same-day control, i.e.
            ``Δlog A − Δlog A_ctl[experiment, day]``. plant-data builds that
            column per row with the true experiment id, so the shared
            unobserved batch factor is already removed from the target and the
            model does not need to observe the experiment to be unconfounded.

    Returns:
        transitions: :class:`ppo.Transition`, leaves of shape ``(N, ...)``.
            ``info["next_obs"]`` is the resulting observation;
            ``info["dyn_target"]`` is the dynamics-model target (equal to
            ``next_obs - obs`` for ``dyn_target="delta_obs"``).
        action_space, observation_space: gymnax spaces from the dataset.
        max_ep_len: Longest episode in the dataset.
        ctl_growth, ctl_energy: Per-day control tables, shape ``(NUM_DAYS,)``.

    Indexing note: ``rewards[t]`` pairs with ``infos[key][t + 1]`` — every
    per-step info describes the *resulting* row, so infos are sliced ``[1:]``
    exactly like ``next_obs``. Index 0 is the reset observation and has no
    incoming transition, which is why its info entries are NaN; no transition
    loses anything as long as the ``[1:]`` slice is used.
    """
    dataset = minari.load_dataset(dataset_id)
    action_space = dataset.action_space
    observation_space = dataset.observation_space

    obss: list[np.ndarray] = []
    next_obss: list[np.ndarray] = []
    dyn_targets: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    terminateds: list[np.ndarray] = []
    truncateds: list[np.ndarray] = []
    ctl_rows: dict[tuple[int, int], tuple[float, float]] = {}
    max_ep_len = 0
    skipped_eps = skipped_steps = 0

    for ep in dataset.iterate_episodes():
        ep_obs = np.asarray(ep.observations)  # (T+1, obs_full_dim)
        ep_actions = np.asarray(ep.actions)  # (T, act_dim)
        ep_rewards = np.asarray(ep.rewards)  # (T,)
        ep_terms = np.asarray(ep.terminations)  # (T,)
        ep_truncs = np.asarray(ep.truncations)  # (T,)
        infos = ep.infos
        T = len(ep_rewards)

        if T == 0:
            continue

        day = np.asarray(infos["day"]).astype(np.float32)  # (T+1,)
        exps = np.asarray(infos["experiment"]).astype(int)
        ctl_g = np.asarray(infos["control_area_reward"], dtype=np.float64)
        ctl_e = np.asarray(infos["control_energy"], dtype=np.float64)
        for i in range(1, T + 1):
            ctl_rows.setdefault(
                (int(exps[i]), int(round(float(day[i])))), (ctl_g[i], ctl_e[i])
            )

        area = ep_obs[:, area_index : area_index + 1].astype(np.float32)  # (T+1, 1)
        if include_time:
            packed = np.concatenate([area, day[:, None]], axis=-1)  # (T+1, 2)
        else:
            packed = area

        if dyn_target == "advantage":
            # The advantage target is only defined where an in-batch control
            # exists; plant-data writes NaN otherwise (every E19/E20 row, which
            # have no constant-white zone). Drop those episodes whole rather
            # than mid-episode, so the episode-start flags that drive
            # ``model_env.reset_weights("init")`` stay meaningful.
            target = np.asarray(infos["area_reward_diff"], dtype=np.float32)[1:, None]
            if not np.all(np.isfinite(target)):
                skipped_eps += 1
                skipped_steps += T
                continue
        elif dyn_target == "delta_obs":
            # Only the stochastic dim. The day, when present, is a deterministic
            # clock the env sets in ``compose_next_obs``, so it is not a
            # regression target — see ``PlantEnv.dyn_dim``.
            target = area[1:] - area[:-1]
        else:
            raise ValueError(f"unknown dyn_target: {dyn_target!r}")

        obss.append(packed[:-1])
        next_obss.append(packed[1:])
        dyn_targets.append(target)
        actions.append(ep_actions.astype(np.float32))
        rewards.append(ep_rewards.astype(np.float32))
        terminateds.append(ep_terms.astype(bool))
        truncateds.append(ep_truncs.astype(bool))
        max_ep_len = max(max_ep_len, T)

    if not obss:
        raise ValueError(
            f"{dataset_id} has no usable episodes for dyn_target={dyn_target!r}"
        )
    if skipped_eps:
        print(
            f"  dropped {skipped_eps} episodes ({skipped_steps} steps) with no "
            f"in-batch control — no {dyn_target} target is defined for them"
        )

    obs = jnp.asarray(np.concatenate(obss, axis=0))
    next_obs = jnp.asarray(np.concatenate(next_obss, axis=0))
    dyn = jnp.asarray(np.concatenate(dyn_targets, axis=0))
    action = jnp.asarray(np.concatenate(actions, axis=0))
    reward = jnp.asarray(np.concatenate(rewards, axis=0))
    terminated = jnp.asarray(np.concatenate(terminateds, axis=0))
    truncated = jnp.asarray(np.concatenate(truncateds, axis=0))

    zeros = jnp.zeros(obs.shape[0])
    transitions = Transition(
        terminated=terminated,
        truncated=truncated,
        action=action,
        value=zeros,
        next_value=zeros,
        reward=reward,
        log_prob=zeros,
        obs=obs,
        info={"next_obs": next_obs, "dyn_target": dyn},
    )

    ctl_growth, ctl_energy = _control_tables(ctl_rows)
    return (
        transitions,
        action_space,
        observation_space,
        max_ep_len,
        ctl_growth,
        ctl_energy,
    )
