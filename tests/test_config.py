"""Tests for mue/config.py."""

import pytest

from mue.config import ExperimentConfig, apply_overrides


def test_to_dict_returns_all_fields():
    config = ExperimentConfig()
    d = config.to_dict()
    assert "env_id" in d
    assert "seed" in d
    assert "model_type" in d
    assert "ensemble_hidden_dims" in d


def test_from_dict_with_valid_keys():
    d = {"env_id": "CartPole-v1", "seed": 123, "max_steps": 500}
    config = ExperimentConfig.from_dict(d)
    assert config.env_id == "CartPole-v1"
    assert config.seed == 123
    assert config.max_steps == 500


def test_from_dict_ignores_unknown_keys():
    d = {"env_id": "Pendulum-v1", "unknown_key": 999}
    config = ExperimentConfig.from_dict(d)
    assert config.env_id == "Pendulum-v1"


def test_from_dict_to_dict_roundtrip():
    original = ExperimentConfig(env_id="CartPole-v1", seed=7)
    d = original.to_dict()
    restored = ExperimentConfig.from_dict(d)
    assert restored.env_id == original.env_id
    assert restored.seed == original.seed


def test_to_dict_tuple_field():
    """Tuple fields should survive to_dict -> from_dict roundtrip."""
    config = ExperimentConfig(ensemble_hidden_dims=(32, 32))
    d = config.to_dict()
    restored = ExperimentConfig.from_dict(d)
    assert len(restored.ensemble_hidden_dims) == 2
    assert restored.ensemble_hidden_dims[0] == 32


def test_apply_overrides_string():
    config = ExperimentConfig()
    config = apply_overrides(config, ["--env_id=CartPole-v1"])
    assert config.env_id == "CartPole-v1"


def test_apply_overrides_int():
    config = ExperimentConfig()
    config = apply_overrides(config, ["--max_steps=100"])
    assert config.max_steps == 100
    assert isinstance(config.max_steps, int)


def test_apply_overrides_float():
    config = ExperimentConfig()
    config = apply_overrides(config, ["--alpha=0.5"])
    assert config.alpha == 0.5
    assert isinstance(config.alpha, float)


def test_apply_overrides_bool_true():
    config = ExperimentConfig()
    config = apply_overrides(config, ["--ensemble_bootstrap=true"])
    assert config.ensemble_bootstrap is True


def test_apply_overrides_bool_false():
    config = ExperimentConfig()
    config = apply_overrides(config, ["--ensemble_bootstrap=false"])
    assert config.ensemble_bootstrap is False


def test_apply_overrides_unknown_key_raises():
    config = ExperimentConfig()
    with pytest.raises(ValueError, match="Unknown config key"):
        apply_overrides(config, ["--nonexistent_key=value"])
