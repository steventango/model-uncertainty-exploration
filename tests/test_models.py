"""Tests for world model classes."""

import jax
import jax.numpy as jnp
import pytest


OBS_DIM = 3
ACT_DIM = 1
N = 20


@pytest.fixture
def training_data():
    key = jax.random.key(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    obs = jax.random.normal(k1, (N, OBS_DIM))
    actions = jax.random.uniform(k2, (N, ACT_DIM), minval=-2.0, maxval=2.0)
    next_obs = obs + 0.01 * jax.random.normal(k3, (N, OBS_DIM))
    rewards = jax.random.normal(k4, (N,))
    return obs, actions, next_obs, rewards


class TestEnsembleModel:
    @pytest.fixture
    def model(self):
        from mue.models.ensemble import EnsembleModel

        return EnsembleModel(
            obs_dim=OBS_DIM,
            act_dim=ACT_DIM,
            num_members=2,
            hidden_sizes=(8, 8),
            learning_rate=1e-3,
            train_epochs=2,
            batch_size=8,
            seed=0,
        )

    def test_fit_returns_metrics(self, model, training_data):
        obs, actions, next_obs, rewards = training_data
        metrics = model.fit(obs, actions, next_obs, rewards)
        assert "model/loss" in metrics
        assert isinstance(metrics["model/loss"], float)

    def test_predict_shapes(self, model, training_data):
        obs, actions, next_obs, rewards = training_data
        model.fit(obs, actions, next_obs, rewards)

        test_obs = obs[:5]
        test_act = actions[:5]
        pred = model.predict(test_obs, test_act)

        out_dim = OBS_DIM + 1
        assert pred.mean.shape == (5, out_dim)
        assert pred.covariance.shape == (5, out_dim, out_dim)
        assert pred.epistemic_covariance is not None
        assert pred.epistemic_covariance.shape == (5, out_dim, out_dim)

    def test_predict_members_length(self, model, training_data):
        obs, actions, next_obs, rewards = training_data
        model.fit(obs, actions, next_obs, rewards)
        members = model.predict_members(obs[:3], actions[:3])
        assert len(members) == 2

    def test_covariance_nonnegative_diagonal(self, model, training_data):
        obs, actions, next_obs, rewards = training_data
        model.fit(obs, actions, next_obs, rewards)
        pred = model.predict(obs[:5], actions[:5])
        for i in range(OBS_DIM + 1):
            assert jnp.all(pred.covariance[:, i, i] >= 0)


class TestGPModel:
    @pytest.fixture
    def model(self):
        from mue.models.gp import GPModel

        return GPModel(
            obs_dim=OBS_DIM,
            act_dim=ACT_DIM,
            learning_rate=0.01,
            num_iters=50,
            seed=0,
        )

    def test_fit_returns_metrics(self, model, training_data):
        obs, actions, next_obs, rewards = training_data
        metrics = model.fit(obs, actions, next_obs, rewards)
        assert "model/loss" in metrics
        assert isinstance(metrics["model/loss"], float)

    def test_predict_shapes(self, model, training_data):
        obs, actions, next_obs, rewards = training_data
        model.fit(obs, actions, next_obs, rewards)
        pred = model.predict(obs[:5], actions[:5])

        out_dim = OBS_DIM + 1
        assert pred.mean.shape == (5, out_dim)
        assert pred.covariance.shape == (5, out_dim, out_dim)
        assert pred.epistemic_covariance is not None
        assert pred.aleatoric_covariance is not None

    def test_covariance_diagonal_nonnegative(self, model, training_data):
        obs, actions, next_obs, rewards = training_data
        model.fit(obs, actions, next_obs, rewards)
        pred = model.predict(obs[:5], actions[:5])
        for i in range(OBS_DIM + 1):
            assert jnp.all(pred.covariance[:, i, i] >= 0)

    def test_condition_on_fantasy(self, model, training_data):
        obs, actions, next_obs, rewards = training_data
        model.fit(obs, actions, next_obs, rewards)

        # Fantasy observations and targets
        fantasy_obs = obs[:2]
        fantasy_actions = actions[:2]
        fantasy_targets = (next_obs - obs)[:2]
        fantasy_targets_with_r = jnp.concatenate(
            [fantasy_targets, rewards[:2].reshape(-1, 1)], axis=-1
        )

        # Condition on fantasy data
        fantasy_model = model.condition_on(
            fantasy_obs, fantasy_actions, fantasy_targets_with_r
        )

        # Should still be able to predict
        pred = fantasy_model.predict(obs[:3], actions[:3])
        out_dim = OBS_DIM + 1
        assert pred.mean.shape == (3, out_dim)


class TestBLRModel:
    @pytest.fixture
    def model(self):
        from mue.models.blr import BLRModel

        return BLRModel(
            obs_dim=OBS_DIM,
            act_dim=ACT_DIM,
            num_features=16,
            seed=0,
        )

    def test_fit_returns_metrics(self, model, training_data):
        obs, actions, next_obs, rewards = training_data
        metrics = model.fit(obs, actions, next_obs, rewards)
        assert "model/loss" in metrics

    def test_predict_shapes(self, model, training_data):
        obs, actions, next_obs, rewards = training_data
        model.fit(obs, actions, next_obs, rewards)
        pred = model.predict(obs[:5], actions[:5])

        out_dim = OBS_DIM + 1
        assert pred.mean.shape == (5, out_dim)
        assert pred.covariance.shape == (5, out_dim, out_dim)
        assert pred.epistemic_covariance is not None
        assert pred.aleatoric_covariance is not None

    def test_covariance_positive_diagonal(self, model, training_data):
        obs, actions, next_obs, rewards = training_data
        model.fit(obs, actions, next_obs, rewards)
        pred = model.predict(obs[:5], actions[:5])
        for i in range(OBS_DIM + 1):
            assert jnp.all(pred.covariance[:, i, i] >= 0)

    def test_has_aleatoric(self, model, training_data):
        obs, actions, next_obs, rewards = training_data
        model.fit(obs, actions, next_obs, rewards)
        pred = model.predict(obs[:5], actions[:5])
        assert pred.aleatoric_covariance is not None
        # Aleatoric should be constant (noise variance)
        assert jnp.all(pred.aleatoric_covariance[:, 0, 0] > 0)


class TestBNNModel:
    @pytest.fixture
    def model(self):
        from mue.models.bnn import BNNModel

        return BNNModel(
            obs_dim=OBS_DIM,
            act_dim=ACT_DIM,
            hidden_dims=(8, 8),
            learning_rate=1e-3,
            train_epochs=2,
            batch_size=8,
            num_mc_samples=3,
            seed=0,
        )

    def test_fit_returns_metrics(self, model, training_data):
        obs, actions, next_obs, rewards = training_data
        metrics = model.fit(obs, actions, next_obs, rewards)
        assert "model/loss" in metrics

    def test_predict_shapes(self, model, training_data):
        obs, actions, next_obs, rewards = training_data
        model.fit(obs, actions, next_obs, rewards)
        pred = model.predict(obs[:5], actions[:5])

        out_dim = OBS_DIM + 1
        assert pred.mean.shape == (5, out_dim)
        assert pred.covariance.shape == (5, out_dim, out_dim)

    def test_kl_divergence_shape(self, model, training_data):
        obs, actions, next_obs, rewards = training_data
        model.fit(obs, actions, next_obs, rewards)
        kl = model.kl_divergence(obs[:5], actions[:5], next_obs[:5])
        assert kl.shape == (5,)
