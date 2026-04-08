"""Tests for mue/types.py."""

import jax.numpy as jnp

from mue.types import BatchTransition, ModelPrediction, Transition


def test_transition_fields():
    t = Transition(
        obs=jnp.array([1.0]),
        action=jnp.array([2.0]),
        next_obs=jnp.array([3.0]),
        reward=jnp.array(4.0),
        done=jnp.array(0.0),
    )
    assert t._fields == ("obs", "action", "next_obs", "reward", "done")


def test_batch_transition():
    bt = BatchTransition(
        obs=jnp.ones((4, 3)),
        action=jnp.ones((4, 1)),
        next_obs=jnp.ones((4, 3)),
        reward=jnp.ones(4),
        done=jnp.zeros(4),
    )
    assert bt.obs.shape == (4, 3)
    assert bt.action.shape == (4, 1)
    assert bt.reward.shape == (4,)


def test_model_prediction_defaults():
    pred = ModelPrediction(
        mean=jnp.zeros((2, 4)),
        covariance=jnp.zeros((2, 4, 4)),
    )
    assert pred.epistemic_covariance is None
    assert pred.aleatoric_covariance is None


def test_model_prediction_with_uncertainty():
    pred = ModelPrediction(
        mean=jnp.zeros((2, 4)),
        covariance=jnp.eye(4)[None].repeat(2, axis=0),
        epistemic_covariance=jnp.eye(4)[None].repeat(2, axis=0) * 0.5,
        aleatoric_covariance=jnp.eye(4)[None].repeat(2, axis=0) * 0.5,
    )
    assert pred.epistemic_covariance is not None
    assert pred.aleatoric_covariance is not None
