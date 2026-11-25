
import jax
from flax import struct


@struct.dataclass
class ModelProperties:
    alpha: jax.Array | float = 1.0
    bias_obs: jax.Array | float = 0.0
    bias_act: jax.Array | float = 0.0
    bias_out: jax.Array | float = 0.0
    scale_obs: jax.Array | float = 1.0
    scale_act: jax.Array | float = 1.0
    scale_out: jax.Array | float = 1.0

@struct.dataclass
class PolicyProperties:
    policy_bias_obs: jax.Array | float = 0.0
    policy_scale_obs: jax.Array | float = 1.0


# Shared defaults for `model_props` and "policy_props" arguments.
# Safe to reuse: struct.dataclass is frozen,
# and a call in a default argument was already evaluated once per function anyway.
default_model_properties = ModelProperties()
default_policy_properties = PolicyProperties()


