import dataclasses

import jax
import jax.numpy as jnp
from flax import nnx

from config import candidate_configs, model_config_dict
from models.base import make_batched_model, make_batched_train_model

SWEEPABLE: dict[str, set[str]] = {
    "blr": {"lam", "a0", "b0", "length_scale"},
    "enn": {"lr", "weight_decay"},
}


def validate_and_expand(model_cfg):
    """Validate sweep fields and expand tuple fields via Cartesian product.

    Returns a list of scalar candidate configs.
    Raises ValueError if any tuple field is not sweepable for this model type.
    """
    name = model_cfg.name
    tupled = {
        f.name
        for f in dataclasses.fields(model_cfg)
        if isinstance(getattr(model_cfg, f.name), tuple)
    }
    if tupled:
        allowed = SWEEPABLE.get(name, set())
        bad = tupled - allowed
        if bad:
            raise ValueError(
                f"Model '{name}' cannot sweep field(s) {sorted(bad)}. "
                f"Sweepable fields for '{name}': {sorted(allowed)}. "
                "Only non-shape Variable-backed fields are supported."
            )
    return list(candidate_configs(model_cfg))


def make_candidates(
    configs,
    model_name,
    *,
    in_features,
    obs_dim,
    out_features,
    act_dim,
    keys,
    max_data,
    minibatch_size,
    predict_reward_terminated,
):
    """Build C independent seed-batched candidate (models, train_state) pairs.

    Each candidate has its own fresh model built from its config's hyperparameters.
    Candidates are throwaway — they exist only to score configs on the val split.

    Returns:
        candidates: list of (models_B, train_state_B) of length C
        sweepable_vals: dict {field: jnp.array of shape (C,)} for all swept fields
    """
    candidates = []
    for c_idx, cfg in enumerate(configs):
        cfg_dict = model_config_dict(cfg, max_data=max_data, minibatch_size=minibatch_size)
        # Derive per-candidate keys so candidates are independent.
        seed_keys_c = jax.vmap(lambda k, i=c_idx: jax.random.fold_in(k, i))(keys)
        m_c, ts_c = make_batched_model(
            model_name,
            cfg_dict,
            in_features,
            obs_dim,
            out_features,
            act_dim,
            seed_keys_c,
            predict_reward_terminated=predict_reward_terminated,
        )
        candidates.append((m_c, ts_c))

    # Gather per-config values for each swept field (all fields, not just tupled ones,
    # so single-config runs produce a (1,) array that's still usable).
    swept_fields = SWEEPABLE.get(model_name, set())
    sweepable_vals = {
        f: jnp.asarray([getattr(c, f) for c in configs], dtype=jnp.float32)
        for f in swept_fields
    }

    return candidates, sweepable_vals


def make_candidate_train_fn(model_name, update_steps, minibatch_size):
    """Build a train function that maps over a list of C candidates sequentially.

    Returns:
        train_candidates(candidates, dataset, pointer, keys) -> list of C histories

    Each candidate is seed-vmapped internally via make_batched_train_model.
    Keys is a list of C seed-batched rngs (one per candidate).
    """
    seed_train = make_batched_train_model(model_name, update_steps, minibatch_size)

    def train_all(candidates, dataset, pointer, rngs_list):
        histories = []
        for (m_c, ts_c), rngs_c in zip(candidates, rngs_list):
            h = seed_train(m_c, ts_c, dataset, pointer, rngs_c)
            histories.append(h)
        return histories

    return train_all


def select_per_seed(candidates, best_c):
    """Gather the per-seed best candidate weights into a single seed-batched model.

    candidates: list of C (models_B, train_state_B)
    best_c: jnp.array shape (B,) — index of winning config per seed

    Returns (models_B, train_state_B) with per-seed winning weights gathered.
    """
    B = best_c.shape[0]
    models_list = [m for m, _ in candidates]
    ts_list = [ts for _, ts in candidates]

    graphdef, _ = nnx.split(models_list[0])
    stacked_state = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs, axis=0),
        *[nnx.split(m)[1] for m in models_list],
    )  # (C, B, ...)
    selected_state = jax.tree_util.tree_map(
        lambda x: x[best_c, jnp.arange(B)], stacked_state
    )  # (B, ...)
    selected_models = nnx.merge(graphdef, selected_state)

    if ts_list[0] is None:
        selected_ts = None
    else:
        # ENN: (optimizer, metrics)
        opts = [ts[0] for ts in ts_list]
        mets = [ts[1] for ts in ts_list]
        opt_gdef, _ = nnx.split(opts[0])
        met_gdef, _ = nnx.split(mets[0])
        stacked_opt = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs, axis=0), *[nnx.split(o)[1] for o in opts]
        )
        stacked_met = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs, axis=0), *[nnx.split(m)[1] for m in mets]
        )
        sel_opt = jax.tree_util.tree_map(lambda x: x[best_c, jnp.arange(B)], stacked_opt)
        sel_met = jax.tree_util.tree_map(lambda x: x[best_c, jnp.arange(B)], stacked_met)
        selected_ts = (nnx.merge(opt_gdef, sel_opt), nnx.merge(met_gdef, sel_met))

    return selected_models, selected_ts


def _set_state_leaves(state, target_key: str, values):
    """Return a new state with every DictKey leaf named `target_key` replaced by `values`."""

    def _update(path, leaf):
        # Path ends with GetAttrKey('value'); second-to-last is DictKey(key=name).
        if len(path) >= 2:
            parent = path[-2]
            if isinstance(parent, jax.tree_util.DictKey) and parent.key == target_key:
                return values
        return leaf

    return jax.tree_util.tree_map_with_path(_update, state)


def apply_winning_hypers(deployed_models, _configs, best_c, _model_name, sweepable_vals):
    """Write the per-seed winning hyperparameter Variables into the deployed model in-place.

    best_c: (B,) array of per-seed winning config indices
    sweepable_vals: dict {field: (C,) array} from make_candidates
    """
    _, state = nnx.split(deployed_models)

    for field, vals_C in sweepable_vals.items():
        sel = vals_C[best_c]  # (B,)
        state = _set_state_leaves(state, field, sel)

    # Write updated state back into deployed_models in-place.
    nnx.update(deployed_models, state)
