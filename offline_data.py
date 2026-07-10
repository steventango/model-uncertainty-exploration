import jax.numpy as jnp
import minari
import numpy as np

from ppo import Transition

AREA_INDEX = 28


def load_offline_transitions(
    dataset_id: str,
    area_index: int = AREA_INDEX,
) -> tuple:
    """Load a minari dataset

    Args:
        dataset_id: Minari dataset id, e.g. ``"plant-data/mixed-e18-daily"``.
        area_index: Column index for ``log_clean_area`` in the observation (default 28).

    Returns:
        transitions: :class:`ppo.Transition` with leaves of shape ``(N, ...)``.
            Only ``obs``, ``action``, ``reward``, ``terminated``, ``truncated``,
            and ``info["next_obs"]`` are meaningful; ``value``/``next_value``/
            ``log_prob`` are zeros.
        action_space: :class:`gymnax.spaces.Box` — action space.
        observation_space: :class:`gymnax.spaces.Box` — observation space.
        max_ep_len: Maximum episode length in the dataset.

    The dataset's initial-state distribution (episode starts) is recovered
    generically from the per-transition ``terminated``/``truncated`` flags by
    :func:`model_env.reset_weights` (``reset_source="init"``).
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
        T = len(ep_rewards)

        if T == 0:
            continue

        # Extract scalar log-area from the full observation vector.
        area = ep_obs[:, area_index : area_index + 1].astype(np.float32)  # (T+1, 1)

        obss.append(area[:-1])  # (T, 1)
        next_obss.append(area[1:])  # (T, 1)
        actions.append(ep_actions.astype(np.float32))  # (T, act_dim)
        rewards.append(ep_rewards.astype(np.float32))  # (T,)
        terminateds.append(ep_terms.astype(bool))
        truncateds.append(ep_truncs.astype(bool))
        max_ep_len = max(max_ep_len, T)

    obs = jnp.asarray(np.concatenate(obss, axis=0))  # (N, 1)
    next_obs = jnp.asarray(np.concatenate(next_obss, axis=0))  # (N, 1)
    action = jnp.asarray(np.concatenate(actions, axis=0))  # (N, act_dim)
    reward = jnp.asarray(np.concatenate(rewards, axis=0))  # (N,)
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
