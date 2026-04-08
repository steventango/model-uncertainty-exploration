from typing import NamedTuple

import jax.numpy as jnp


class Transition(NamedTuple):
    obs: jnp.ndarray  # (obs_dim,)
    action: jnp.ndarray  # (act_dim,)
    next_obs: jnp.ndarray  # (obs_dim,)
    reward: jnp.ndarray  # scalar
    done: jnp.ndarray  # scalar


class ModelPrediction(NamedTuple):
    mean: jnp.ndarray  # (batch, obs_dim + 1) — last dim is reward
    covariance: jnp.ndarray  # (batch, obs_dim + 1, obs_dim + 1) — total predictive
    epistemic_covariance: jnp.ndarray | None = None  # reducible uncertainty
    aleatoric_covariance: jnp.ndarray | None = None  # irreducible noise


class EnsemblePrediction(NamedTuple):
    mean: jnp.ndarray  # (batch, obs_dim + 1) — ensemble mean
    covariance: jnp.ndarray  # (batch, obs_dim + 1, obs_dim + 1) — from disagreement
    member_means: jnp.ndarray  # (n_members, batch, obs_dim + 1) — per-member


class BatchTransition(NamedTuple):
    obs: jnp.ndarray  # (batch, obs_dim)
    action: jnp.ndarray  # (batch, act_dim)
    next_obs: jnp.ndarray  # (batch, obs_dim)
    reward: jnp.ndarray  # (batch,)
    done: jnp.ndarray  # (batch,)

    @staticmethod
    def from_jax_dict(d: dict[str, jnp.ndarray]) -> "BatchTransition":
        return BatchTransition(
            obs=d["obs"],
            action=d["action"],
            next_obs=d["next_obs"],
            reward=d["reward"],
            done=d["done"],
        )
