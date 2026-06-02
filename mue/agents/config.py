from mue.agents.ppo.config import PPOConfig
from mue.agents.random.config import RandomConfig

AgentConfig = PPOConfig | RandomConfig
