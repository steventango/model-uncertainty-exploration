import jax.numpy as jnp
import minari
import numpy as np

from ppo import Transition

AREA_INDEX = 28


def load_offline_transitions(
    dataset_id: str,
    area_index: int = AREA_INDEX,
    include_time: bool = False,
) -> tuple:
    """Load a minari dataset.

    Args:
        dataset_id: Minari dataset id, e.g. ``"plant-data/visu-v28"``.
        area_index: Column index for ``log_clean_area`` in the observation.
        include_time: If True, append the day index as a second obs dim
            ``[log_area, day]``. Needed by day-dependent rewards (the masked
            gates). The day is read from ``infos["day"]``, not from the
            position in the episode.

    Returns:
        transitions: :class:`ppo.Transition`, leaves of shape ``(N, ...)``.
            ``info["next_obs"]`` is the resulting observation.
        action_space, observation_space: gymnax spaces from the dataset.
        max_ep_len: Longest episode in the dataset.

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
    actions: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    terminateds: list[np.ndarray] = []
    truncateds: list[np.ndarray] = []
    max_ep_len = 0

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

        area = ep_obs[:, area_index : area_index + 1].astype(np.float32)  # (T+1, 1)
        if include_time:
            packed = np.concatenate([area, day[:, None]], axis=-1)  # (T+1, 2)
        else:
            packed = area

        obss.append(packed[:-1])
        next_obss.append(packed[1:])
        actions.append(ep_actions.astype(np.float32))
        rewards.append(ep_rewards.astype(np.float32))
        terminateds.append(ep_terms.astype(bool))
        truncateds.append(ep_truncs.astype(bool))
        max_ep_len = max(max_ep_len, T)

    obs = jnp.asarray(np.concatenate(obss, axis=0))
    next_obs = jnp.asarray(np.concatenate(next_obss, axis=0))
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
        info={"next_obs": next_obs},
    )

    return transitions, action_space, observation_space, max_ep_len
