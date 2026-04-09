from dataclasses import dataclass, field

from mue.data import BufferConfig


@dataclass
class ModelConfig:
    """Configuration for world model."""

    model_type: str = "gp"


@dataclass
class ExperimentConfig:
    env_id: str = "Pendulum-v1"
    seed: int = 42
    max_steps: int = 10_000
    eval_episodes: int = 5
    log_dir: str = "results"
    buffer: BufferConfig = field(default_factory=BufferConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
