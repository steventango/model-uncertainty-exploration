"""Tests for mue/types.py."""

import jax.numpy as jnp

from mue.types import ModelPrediction


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
