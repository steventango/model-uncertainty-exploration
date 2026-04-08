"""Tests for mue/config.py."""

from dataclasses import asdict

import tyro

from mue.config import ExperimentConfig


def test_defaults():
    config = ExperimentConfig()
    assert config.env_id == "Pendulum-v1"
    assert config.seed == 42
    assert config.log_dir == "results"


def test_asdict_returns_all_fields():
    config = ExperimentConfig()
    d = asdict(config)
    assert "env_id" in d
    assert "seed" in d
    assert "max_steps" in d
    assert "log_dir" in d


def test_tyro_cli_override():
    config = tyro.cli(ExperimentConfig, args=["--env-id", "CartPole-v1", "--seed", "7"])
    assert config.env_id == "CartPole-v1"
    assert config.seed == 7


def test_tyro_cli_defaults():
    config = tyro.cli(ExperimentConfig, args=[])
    assert config == ExperimentConfig()
