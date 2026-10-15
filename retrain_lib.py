"""Load the nightly retrain artifacts archived by plant-rl's AdaptivePPOPolicy.

The archives hold, per night: the updated ENN world model, the freshly trained
PPO explore policy + its observation normalizer (all orbax StandardCheckpointer
trees, same layout as this repo's main.py writes), the online transitions
collected so far, and a metadata JSON.
"""
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx

from models.enn import _make_batched_model
from networks import ActorCritic
from normalization import NormalizeVecObs

CPU = jax.devices("cpu")[0]

# plant-rl src/algorithms/jax/mbrl/hparams.py ENN_DEFAULTS
ENN_DEFAULTS = {
    "LR": 1e-3,
    "HIDDEN_DIM": 64,
    "LEARNABLE_HIDDEN_DIM": 15,
    "PRIOR_HIDDEN_DIM": 5,
    "INDEX_DIM": 8,
    "UPDATE_STEPS": 10000,
    "USE_LAYER_NORM": True,
    "MINIBATCH_SIZE": 256,
}


def _unstack(batched, i):
    graphdef, state = nnx.split(batched)
    return nnx.merge(graphdef, jax.tree.map(lambda x: x[i], state))


def build_model(obs_dim, action_dim, dyn_dim, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 1)
    models, _ = _make_batched_model(
        ENN_DEFAULTS,
        obs_dim + action_dim,
        obs_dim,
        dyn_dim,
        None,
        keys,
        predict_reward_terminated=False,
        dyn_dim=dyn_dim,
    )
    return _unstack(models, 0)


def restore(module, path):
    graphdef, state = nnx.split(module)
    state = ocp.StandardCheckpointer().restore(os.path.abspath(path), target=state)
    return nnx.merge(graphdef, state)


def load_night(archive_dir, obs_dim, action_dim, dyn_dim, hidden_dim=64,
               activation="tanh", seed=0):
    """Return (model, network, obs_norm, online_transitions, metadata)."""
    with jax.default_device(CPU):
        model = restore(build_model(obs_dim, action_dim, dyn_dim, seed),
                        os.path.join(archive_dir, "model"))
        network = restore(
            ActorCritic(obs_dim, action_dim, hidden_dim, activation=activation,
                        use_layer_norm=False, rngs=nnx.Rngs(seed)),
            os.path.join(archive_dir, "network"),
        )
        obs_norm = restore(NormalizeVecObs(jnp.zeros(obs_dim)),
                           os.path.join(archive_dir, "obs_norm"))
    online = dict(np.load(os.path.join(archive_dir, "online_transitions.npz")))
    with open(os.path.join(archive_dir, "metadata.json")) as f:
        meta = json.load(f)
    return model, network, obs_norm, online, meta


def nights(archive_root):
    """Archive dirs that carry a full artifact set, in chronological order."""
    out = []
    for d in sorted(os.listdir(archive_root)):
        p = os.path.join(archive_root, d)
        if os.path.isfile(os.path.join(p, "metadata.json")):
            out.append(p)
    return out
