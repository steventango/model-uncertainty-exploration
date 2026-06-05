import os
from typing import NamedTuple

import gymnax
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from model import train_model
from networks import MLP, ActorCritic
from normalization import NormalizeVecObs, NormalizeVecReward
from wrappers import (
    ClipAction,
    LogWrapper,
    VecEnv,
)


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

    def train(train_state, rng):
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
            metric = traj_batch.info
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


if __name__ == "__main__":
    config = {
        "LR": 3e-4,
        "NUM_ENVS": 2048,
        "NUM_STEPS": 10,
        "TOTAL_TIMESTEPS": 5e7,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 32,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.0,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "HIDDEN_DIM": 64,
        "ACTIVATION": "tanh",
        "ENV_NAME": "Pendulum-v1",
        "ANNEAL_LR": False,
        "NORMALIZE_ENV": True,
        "DEBUG": True,
    }
    rollout_config = config.copy()
    rollout_config["NUM_ENVS"] = 1
    rollout_config["NUM_STEPS"] = 200
    rollout_config["DATASET_SIZE"] = 10000
    model_config = {
        "LR": 1e-3,
        "HIDDEN_DIM": 64,
        "ACTIVATION": "tanh",
        "EPOCHS": 1000,
        "MINIBATCH_SIZE": rollout_config["NUM_STEPS"],
    }

    rng = jax.random.PRNGKey(30)

    env, env_params = gymnax.make(rollout_config["ENV_NAME"])
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)

    # INIT NETWORK
    rng, _rng = jax.random.split(rng)
    rngs = nnx.Rngs(_rng)
    network = ActorCritic(
        env.observation_space(env_params).shape[0],
        env.action_space(env_params).shape[0],
        config["HIDDEN_DIM"],
        activation=config["ACTIVATION"],
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

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, rollout_config["NUM_ENVS"])
    obsv, env_state = env.reset(reset_rng, env_params)

    # ROLLOUT
    rng, _rng = jax.random.split(rng)
    runner_state = (train_state, env_state, obsv, _rng)
    _rollout = make_rollout(rollout_config, env, env_params, training=False)
    runner_state, traj_batch = _rollout(runner_state)

    # INIT DATASET
    dataset = jax.tree_util.tree_map(
        lambda x: jnp.zeros(
            (rollout_config["DATASET_SIZE"],) + x.shape[2:], dtype=x.dtype
        ),
        traj_batch,
    )
    pointer = 0

    # UPDATE DATASET
    traj_batch = jax.tree_util.tree_map(
        lambda x: x.reshape((-1,) + x.shape[2:]), traj_batch
    )
    dataset = jax.tree_util.tree_map(
        lambda old, new: jax.lax.dynamic_update_slice_in_dim(old, new, pointer, axis=0),
        dataset,
        traj_batch,
    )
    pointer += traj_batch.obs.shape[0]

    # INIT MODEL
    model = MLP(
        env.observation_space(env_params).shape[0]
        + env.action_space(env_params).shape[0],
        env.observation_space(env_params).shape[0] + 2,
        hidden_dim=model_config["HIDDEN_DIM"],
        activation=model_config["ACTIVATION"],
        rngs=rngs,
    )
    optimizer = nnx.Optimizer(model, optax.adamw(model_config["LR"]), wrt=nnx.Param)
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        delta_next_state_loss=nnx.metrics.Average("delta_next_state_loss"),
        reward_loss=nnx.metrics.Average("reward_loss"),
        terminated_loss=nnx.metrics.Average("terminated_loss"),
    )

    # TRAIN MODEL
    # TODO: use mask to handle variable dataset size
    batch = jax.tree_util.tree_map(lambda x: x[:pointer], dataset)
    history = train_model(
        model,
        optimizer,
        metrics,
        batch,
        model_config["EPOCHS"],
        pointer,
        model_config["MINIBATCH_SIZE"],
        rngs=rngs,
    )

    # PLOT LOSSES
    for loss, loss_history in history.items():
        plt.figure()
        plt.plot(loss_history)
        plt.xlabel("Epoch")
        plt.ylabel(loss)
        plt.title(f"{loss} over epochs")
        fig_path = f"/tmp/ppo_continuous_action_{loss}.png"
        plt.savefig(fig_path)
        print(f"Saved {loss} curve to {fig_path}")
        plt.close()
    out = train_jit(train_state, rng)
