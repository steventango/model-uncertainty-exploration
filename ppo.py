from typing import NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp
import optax

from gymnax.environments import spaces

from networks import ActorCritic
from normalization import NormalizeVecObs, NormalizeVecReward


class Transition(NamedTuple):
    terminated: jnp.ndarray
    truncated: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    next_value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_rollout(config, env, env_params, training=True):
    discrete = isinstance(env.action_space(env_params), spaces.Discrete)

    def _rollout(runner_state):
        # COLLECT TRAJECTORIES
        def _env_step(runner_state, unused):
            train_state, env_state, last_obs, rng = runner_state
            network, _, normalize_vec_obs, normalize_vec_reward = train_state
            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            if config["NORMALIZE_ENV"] and not training:
                normalize_vec_obs.eval()
                network_last_obs = normalize_vec_obs(last_obs)
            else:
                network_last_obs = last_obs
            pi, value = network(network_last_obs)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state, reward, terminated, truncated, info = env.step(
                rng_step, env_state, action, env_params
            )
            next_obs = info["next_obs"]

            if config["NORMALIZE_ENV"]:
                if training:
                    normalize_vec_obs.train()
                    obsv = normalize_vec_obs(obsv)
                normalize_vec_obs.eval()
                next_obs = normalize_vec_obs(next_obs)
                if training:
                    normalize_vec_reward.train()
                    reward = normalize_vec_reward(reward, terminated, truncated)

            _, next_value = network(next_obs)

            if not discrete and not training:
                action = jnp.clip(
                    action,
                    env.action_space(env_params).low,
                    env.action_space(env_params).high,
                )

            transition = Transition(
                terminated,
                truncated,
                action,
                value,
                next_value,
                reward,
                log_prob,
                last_obs,
                info,
            )
            runner_state = (train_state, env_state, obsv, rng)
            return runner_state, transition

        runner_state, traj_batch = nnx.scan(_env_step, length=config["NUM_STEPS"])(
            runner_state, None
        )

        return runner_state, traj_batch

    return _rollout


def make_train(env, config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    def train(train_state, env_params, rng):
        _, _, normalize_vec_obs, _ = train_state

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)

        if config["NORMALIZE_ENV"]:
            normalize_vec_obs.train()
            obsv = normalize_vec_obs(obsv)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            _rollout = make_rollout(config, env, env_params, training=True)
            runner_state, traj_batch = _rollout(runner_state)

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state

            def _calculate_gae(traj_batch):
                def _get_advantages(gae, transition):
                    terminated, truncated, value, next_value, reward = (
                        transition.terminated,
                        transition.truncated,
                        transition.value,
                        transition.next_value,
                        transition.reward,
                    )
                    delta = (
                        reward + config["GAMMA"] * next_value * (1 - terminated) - value
                    )
                    done = terminated | truncated
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return gae, gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    jnp.zeros_like(traj_batch.value[0]),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    network, optimizer, _, _ = train_state
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(network, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network(traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = nnx.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        network, traj_batch, advantages, targets
                    )
                    optimizer.update(network, grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert batch_size == config["NUM_STEPS"] * config["NUM_ENVS"], (
                    "batch size must be equal to number of steps * number of envs"
                )
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = nnx.scan(_update_minbatch)(
                    train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = nnx.scan(
                _update_epoch, length=config["UPDATE_EPOCHS"]
            )(update_state, None)
            train_state = update_state[0]
            metric = {
                "returned_episode_returns": traj_batch.info["returned_episode_returns"],
                "returned_episode": traj_batch.info["returned_episode"],
            }
            rng = update_state[-1]
            if config.get("DEBUG"):

                def callback(info):
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    )
                    for t in range(len(timesteps)):
                        print(
                            f"global step={timesteps[t]}, episodic return={return_values[t]}"
                        )

                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = nnx.scan(_update_step, length=config["NUM_UPDATES"])(
            runner_state, None
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


def make_train_state(config, env, env_params, rngs):
    action_space = env.action_space(env_params)
    discrete = isinstance(action_space, spaces.Discrete)
    action_dim = action_space.n if discrete else action_space.shape[0]
    network = ActorCritic(
        env.observation_space(env_params).shape[0],
        action_dim,
        config["HIDDEN_DIM"],
        activation=config["ACTIVATION"],
        discrete=discrete,
        use_layer_norm=config["USE_LAYER_NORM"],
        rngs=rngs,
    )

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    if config["ANNEAL_LR"]:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )
    optimizer = nnx.Optimizer(network, tx, wrt=nnx.Param)
    normalize_vec_obs = NormalizeVecObs(
        jnp.zeros(env.observation_space(env_params).shape)
    )
    normalize_vec_reward = NormalizeVecReward(
        jnp.zeros(config["NUM_ENVS"]), config["GAMMA"]
    )
    train_state = (network, optimizer, normalize_vec_obs, normalize_vec_reward)
    return train_state


def make_batched_train(env, env_params, config):
    """Vmap PPO ``train`` over B seeds (outer) and C configs (inner).

    ``env_params`` carries everything that varies: its ``model`` subtree has a
    leading B (seed) axis and its ``alpha``/``beta`` weights have a leading C
    (config) axis. Argument shapes expected by the returned function:
      train_state: (B, C) leading;   env_params: one struct with ``model`` on a
      leading B axis and ``alpha``/``beta`` on a leading C axis;   rng: (B, C).
    Outputs come back with (B, C, ...) leading axes.
    """
    train = make_train(env, config)
    # Inner vmap over C configs: map alpha/beta (prefix in_axes), broadcast model.
    inner = nnx.vmap(train, in_axes=(0, env_params.config_vmap_axes(), 0), out_axes=0)
    # Outer vmap over B seeds: map the model subtree (prefix in_axes), broadcast weights.
    return nnx.vmap(inner, in_axes=(0, env_params.seed_vmap_axes(), 0), out_axes=0)


def make_batched_train_state(config, env, env_params, keys):
    """Build a (B, C) grid of independent PPO train states. ``keys`` is (B, C)."""

    @nnx.vmap
    def build_seed(seed_keys):  # seed_keys: (C,) keys
        @nnx.vmap
        def build_config(key):
            return make_train_state(config, env, env_params, nnx.Rngs(params=key))

        return build_config(seed_keys)

    return build_seed(keys)


def _index_train_state(batched_train_state, index_fn):
    """Rebuild a train state with ``index_fn`` applied to every leaf of its state."""
    graphdef, state = nnx.split(batched_train_state)
    return nnx.merge(graphdef, jax.tree.map(index_fn, state))


def unstack_train_state(batched_train_state, i):
    return _index_train_state(batched_train_state, lambda x: x[i])


def select_config_train_state(batched_train_state, c):
    """Select config index ``c`` (axis 1), keeping the seed batch axis 0."""
    return _index_train_state(batched_train_state, lambda x: x[:, c])
