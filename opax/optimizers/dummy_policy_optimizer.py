import flax
import jax


@flax.struct.dataclass
class BestSequences:
    evaluation_sequences: jax.Array
    exploration_sequence: jax.Array


class DummyPolicyOptimizer:

    def __init__(self, *args, **kwargs):
        return

    def train(self, *args, **kwargs):
        return {}

    def reset(self):
        pass
