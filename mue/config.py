from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    env_id: str = "Pendulum-v1"
    seed: int = 42
    max_steps: int = 10_000
    eval_episodes: int = 5
    log_dir: str = "results"
