from typing import NamedTuple

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from gymnax.environments import spaces

from networks import MLP
from normalization import NormalizeVecObs, NormalizeVecReward

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class Batch(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    next_obs: jnp.ndarray


class SquashedNormal(distrax.Transformed):
    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class Actor(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        activation: str,
        discrete: bool,
        low: tuple[float, ...],
        high: tuple[float, ...],
        *,
        rngs: nnx.Rngs,
    ):
        self.discrete = discrete
        self.low = low
        self.high = high
        out_features = action_dim if discrete else 2 * action_dim
        self.mlp = MLP(
            obs_dim,
            hidden_dim,
            out_features,
            rngs=rngs,
            num_layers=2,
            activation=activation,
        )

    def __call__(self, obs):
        out, _ = self.mlp(obs)
        if self.discrete:
            return distrax.Categorical(logits=out)
        mean, log_std = jnp.split(out, 2, axis=-1)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        low, high = jnp.asarray(self.low), jnp.asarray(self.high)
        bijector = distrax.Block(
            distrax.Chain(
                [
                    distrax.ScalarAffine((high + low) / 2, (high - low) / 2),
                    distrax.Tanh(),
                ]
            ),
            ndims=1,
        )
        return SquashedNormal(
            distrax.MultivariateNormalDiag(mean, jnp.exp(log_std)), bijector
        )


class TwinQ(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int,
        activation: str,
        *,
        rngs: nnx.Rngs,
    ):
        self.q1 = MLP(
            in_features,
            hidden_dim,
            out_features,
            rngs=rngs,
            num_layers=2,
            activation=activation,
        )
        self.q2 = MLP(
            in_features,
            hidden_dim,
            out_features,
            rngs=rngs,
            num_layers=2,
            activation=activation,
        )

    def __call__(self, x):
        q1, _ = self.q1(x)
        q2, _ = self.q2(x)
        return q1, q2


class Temperature(nnx.Module):
    def __init__(self, init_temperature: float):
        self.log_temperature = nnx.Param(jnp.log(jnp.asarray(init_temperature)))

    def __call__(self):
        return jnp.exp(self.log_temperature[...])


class SAC(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        activation: str,
        discrete: bool,
        low: tuple[float, ...],
        high: tuple[float, ...],
        init_temperature: float,
        *,
        rngs: nnx.Rngs,
    ):
        self.discrete = discrete
        self.actor = Actor(
            obs_dim,
            action_dim,
            hidden_dim,
            activation,
            discrete,
            low,
            high,
            rngs=rngs,
        )
        q_in, q_out = (obs_dim, action_dim) if discrete else (obs_dim + action_dim, 1)
        self.q = TwinQ(q_in, q_out, hidden_dim, activation, rngs=rngs)
        self.target_q = nnx.clone(self.q)
        self.temperature = Temperature(init_temperature)

    def __call__(self, obs):
        pi = self.actor(obs)
        q1, q2 = q_values(self.q, obs, pi.mode(), self.discrete)
        return pi, jnp.minimum(q1, q2)


class SACOptimizer(nnx.Module):
    def __init__(self, sac: SAC, lr: float):
        self.actor = nnx.Optimizer(sac.actor, optax.adam(lr), wrt=nnx.Param)
        self.q = nnx.Optimizer(sac.q, optax.adam(lr), wrt=nnx.Param)
        self.temperature = nnx.Optimizer(sac.temperature, optax.adam(lr), wrt=nnx.Param)


def q_values(q, obs, action, discrete):
    if discrete:
        q1, q2 = q(obs)
        index = action[..., None].astype(jnp.int32)
        q1 = jnp.take_along_axis(q1, index, axis=-1)
        q2 = jnp.take_along_axis(q2, index, axis=-1)
    else:
        q1, q2 = q(jnp.concatenate([obs, action], axis=-1))
    return q1[..., 0], q2[..., 0]


def target_entropy(config, action_space):
    if isinstance(action_space, spaces.Discrete):
        return config["TARGET_ENTROPY_SCALE"] * np.log(action_space.n)
    return -float(np.prod(action_space.shape))


def make_update(config, entropy_target):
    def update(sac, optimizer, batch, key):
        key_next, key_pi = jax.random.split(key)
        temperature = sac.temperature()

        next_pi = sac.actor(batch.next_obs)
        if sac.discrete:
            next_log_probs = jax.nn.log_softmax(next_pi.logits)
            target_q1, target_q2 = sac.target_q(batch.next_obs)
            next_value = (
                jnp.exp(next_log_probs)
                * (jnp.minimum(target_q1, target_q2) - temperature * next_log_probs)
            ).sum(-1)
        else:
            next_action, next_log_prob = next_pi.sample_and_log_prob(seed=key_next)
            target_q1, target_q2 = q_values(
                sac.target_q, batch.next_obs, next_action, False
            )
            next_value = jnp.minimum(target_q1, target_q2) - temperature * next_log_prob
        target = jax.lax.stop_gradient(
            batch.reward + config["GAMMA"] * (1 - batch.terminated) * next_value
        )

        def q_loss_fn(q):
            q1, q2 = q_values(q, batch.obs, batch.action, sac.discrete)
            loss1 = 0.5 * jnp.square(q1 - target).mean()
            loss2 = 0.5 * jnp.square(q2 - target).mean()
            return loss1 + loss2, jnp.minimum(q1, q2).mean()

        (q_loss, q_mean), grads = nnx.value_and_grad(q_loss_fn, has_aux=True)(sac.q)
        optimizer.q.update(sac.q, grads)

        def actor_loss_fn(actor, q):
            pi = actor(batch.obs)
            if sac.discrete:
                log_probs = jax.nn.log_softmax(pi.logits)
                probs = jnp.exp(log_probs)
                q1, q2 = q(batch.obs)
                loss = (probs * (temperature * log_probs - jnp.minimum(q1, q2))).sum(-1)
                return loss.mean(), -(probs * log_probs).sum(-1)
            action, log_prob = pi.sample_and_log_prob(seed=key_pi)
            q1, q2 = q_values(q, batch.obs, action, False)
            loss = temperature * log_prob - jnp.minimum(q1, q2)
            return loss.mean(), -log_prob

        (actor_loss, entropy), grads = nnx.value_and_grad(actor_loss_fn, has_aux=True)(
            sac.actor, sac.q
        )
        optimizer.actor.update(sac.actor, grads)

        def temperature_loss_fn(temperature):
            entropy_gap = jax.lax.stop_gradient(entropy) - entropy_target
            return (temperature() * entropy_gap).mean()

        temperature_loss, grads = nnx.value_and_grad(temperature_loss_fn)(
            sac.temperature
        )
        optimizer.temperature.update(sac.temperature, grads)

        target_params = optax.incremental_update(
            nnx.state(sac.q, nnx.Param),
            nnx.state(sac.target_q, nnx.Param),
            config["TAU"],
        )
        nnx.update(sac.target_q, target_params)

        return {
            "q_loss": q_loss,
            "actor_loss": actor_loss,
            "temperature_loss": temperature_loss,
            "q": q_mean,
            "entropy": entropy.mean(),
            "temperature": temperature,
        }

    return update


def make_train(env, config):
    num_envs = config["NUM_ENVS"]
    buffer_size = config["BUFFER_SIZE"]
    warmup_iters = min(config["LEARNING_STARTS"] // num_envs, config["NUM_ITERS"])
    learn_iters = config["NUM_ITERS"] - warmup_iters

    def train(train_state, env_params, rng):
        sac, optimizer, _, _ = train_state
        action_space = env.action_space(env_params)
        _update = make_update(config, target_entropy(config, action_space))

        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(jax.random.split(_rng, num_envs), env_params)
        action_shape = () if sac.discrete else action_space.shape
        buffer = Batch(
            obs=jnp.zeros((buffer_size,) + obsv.shape[1:], obsv.dtype),
            action=jnp.zeros(
                (buffer_size,) + action_shape,
                jnp.int32 if sac.discrete else jnp.float32,
            ),
            reward=jnp.zeros(buffer_size),
            terminated=jnp.zeros(buffer_size),
            next_obs=jnp.zeros((buffer_size,) + obsv.shape[1:], obsv.dtype),
        )

        def _gradient_step(carry, key):
            sac, optimizer, buffer, count = carry
            key_sample, key_update = jax.random.split(key)
            index = jax.random.randint(key_sample, (config["BATCH_SIZE"],), 0, count)
            batch = jax.tree.map(lambda x: x[index], buffer)
            losses = _update(sac, optimizer, batch, key_update)
            return (sac, optimizer, buffer, count), losses

        def make_iter(learn):
            def _iter(carry, unused):
                sac, optimizer, env_state, obs, buffer, ptr, count, rng = carry
                rng, key_action, key_step, key_update = jax.random.split(rng, 4)
                action = sac.actor(obs).sample(seed=key_action)
                obsv, env_state, reward, terminated, _, info = env.step(
                    jax.random.split(key_step, num_envs), env_state, action, env_params
                )
                index = (ptr + jnp.arange(num_envs)) % buffer_size
                transition = Batch(
                    obs,
                    action,
                    reward,
                    terminated.astype(jnp.float32),
                    info["next_obs"],
                )
                buffer = jax.tree.map(
                    lambda b, x: b.at[index].set(x.astype(b.dtype)), buffer, transition
                )
                ptr = (ptr + num_envs) % buffer_size
                count = jnp.minimum(count + num_envs, buffer_size)
                losses = None
                if learn:
                    keys = jax.random.split(key_update, config["UPDATES_PER_ITER"])
                    _, losses = nnx.scan(_gradient_step)(
                        (sac, optimizer, buffer, count), keys
                    )
                    losses = jax.tree.map(jnp.mean, losses)
                metric = {
                    "returned_episode_returns": info["returned_episode_returns"],
                    "returned_episode": info["returned_episode"],
                }
                carry = (sac, optimizer, env_state, obsv, buffer, ptr, count, rng)
                return carry, (metric, losses)

            return _iter

        carry = (
            sac,
            optimizer,
            env_state,
            obsv,
            buffer,
            jnp.int32(0),
            jnp.int32(0),
            rng,
        )
        carry, (warmup_metric, _) = nnx.scan(make_iter(False), length=warmup_iters)(
            carry, None
        )
        carry, (learn_metric, losses) = nnx.scan(make_iter(True), length=learn_iters)(
            carry, None
        )
        metric = jax.tree.map(
            lambda a, b: jnp.concatenate([a, b]), warmup_metric, learn_metric
        )
        _, _, env_state, obsv, _, _, _, rng = carry
        runner_state = (train_state, env_state, obsv, rng)
        return {"runner_state": runner_state, "metrics": metric, "losses": losses}

    return train


def make_train_state(config, env, env_params, rngs):
    action_space = env.action_space(env_params)
    discrete = isinstance(action_space, spaces.Discrete)
    obs_shape = env.observation_space(env_params).shape
    if discrete:
        action_dim, low, high = action_space.n, (), ()
    else:
        action_dim = action_space.shape[0]
        low = np.broadcast_to(np.asarray(action_space.low), action_space.shape)
        high = np.broadcast_to(np.asarray(action_space.high), action_space.shape)
        low, high = tuple(low.tolist()), tuple(high.tolist())
    sac = SAC(
        obs_shape[0],
        action_dim,
        config["HIDDEN_DIM"],
        config["ACTIVATION"],
        discrete,
        low,
        high,
        config["INIT_TEMPERATURE"],
        rngs=rngs,
    )
    optimizer = SACOptimizer(sac, config["LR"])
    normalize_vec_obs = NormalizeVecObs(jnp.zeros(obs_shape))
    normalize_vec_reward = NormalizeVecReward(
        jnp.zeros(config["NUM_ENVS"]), config["GAMMA"]
    )
    return (sac, optimizer, normalize_vec_obs, normalize_vec_reward)


def make_batched_train(env, env_params, config):
    train = make_train(env, config)
    inner = nnx.vmap(train, in_axes=(0, env_params.config_vmap_axes(), 0), out_axes=0)
    return nnx.vmap(inner, in_axes=(0, env_params.seed_vmap_axes(), 0), out_axes=0)


def make_batched_train_state(config, env, env_params, keys):
    @nnx.vmap
    def build_seed(seed_keys):
        @nnx.vmap
        def build_config(key):
            return make_train_state(config, env, env_params, nnx.Rngs(params=key))

        return build_config(seed_keys)

    return build_seed(keys)
