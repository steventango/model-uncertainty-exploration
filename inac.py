"""In-Sample Actor-Critic (InAC; Xiao et al., 2023), ported from
https://github.com/steventango/inacax (core/agent/in_sample.py, core/network/*).

Two entry points share one update:
- ``make_offline_train``: gradient steps on a fixed dataset (growing batch).
- ``make_train``: online replay loop against an env (the ModelEnv solver);
  same signature/return as ``ppo.make_train``.

The train state is a 4-tuple ``(agent, optimizers, normalize_vec_obs,
normalize_vec_reward)`` matching PPO's, and ``InACAgent.__call__`` returns
``(pi, value)`` like ``networks.ActorCritic``, so ``ppo.make_rollout`` and the
evaluation / checkpoint helpers work unchanged. The normalizers are unused
(``NORMALIZE_ENV=False``).
"""

import distrax
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from gymnax.environments import spaces
from jax.nn import initializers

from normalization import NormalizeVecObs, NormalizeVecReward


class MLP(nnx.Module):
    """ReLU MLP with xavier-uniform kernels and zero biases (inacax FCNetwork)."""

    def __init__(self, in_dim, hidden_dim, out_dim, *, rngs: nnx.Rngs):
        dims = (in_dim, hidden_dim, hidden_dim)
        self.layers = nnx.List(
            [
                nnx.Linear(
                    d_in,
                    d_out,
                    kernel_init=initializers.xavier_uniform(),
                    bias_init=initializers.zeros,
                    rngs=rngs,
                )
                for d_in, d_out in zip(dims[:-1], dims[1:])
            ]
        )
        self.head = nnx.Linear(
            hidden_dim,
            out_dim,
            kernel_init=initializers.xavier_uniform(),
            bias_init=initializers.zeros,
            rngs=rngs,
        )

    def __call__(self, x):
        for layer in self.layers:
            x = nnx.relu(layer(x))
        return self.head(x)


class DiscretePolicy(nnx.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, *, rngs: nnx.Rngs):
        self.net = MLP(obs_dim, hidden_dim, action_dim, rngs=rngs)

    def __call__(self, obs):
        return distrax.Categorical(logits=self.net(obs))


class GaussianPolicy(nnx.Module):
    """tanh-squashed mean, state-independent sigmoid-bounded log-std (MLPCont)."""

    min_log_std = -6.0
    max_log_std = 0.0

    def __init__(self, obs_dim, action_dim, hidden_dim, action_range, *, rngs):
        self.net = MLP(obs_dim, hidden_dim, action_dim, rngs=rngs)
        self.log_std_logits = nnx.Param(jnp.zeros(action_dim))
        self.action_range = action_range

    def __call__(self, obs):
        mu = jnp.tanh(self.net(obs)) * self.action_range
        log_std = self.min_log_std + jax.nn.sigmoid(self.log_std_logits.get_value()) * (
            self.max_log_std - self.min_log_std
        )
        return distrax.MultivariateNormalDiag(
            mu, jnp.broadcast_to(jnp.exp(log_std), mu.shape)
        )


class DoubleQ(nnx.Module):
    """Twin critics. Discrete: Q(s, .) heads indexed by action; continuous: Q(s, a)."""

    def __init__(self, obs_dim, action_dim, hidden_dim, discrete, *, rngs):
        self.discrete = discrete
        in_dim, out_dim = (
            (obs_dim, action_dim) if discrete else (obs_dim + action_dim, 1)
        )
        self.q1 = MLP(in_dim, hidden_dim, out_dim, rngs=rngs)
        self.q2 = MLP(in_dim, hidden_dim, out_dim, rngs=rngs)

    def __call__(self, obs, action):
        if self.discrete:
            idx = action[..., None].astype(jnp.int32)
            q1 = jnp.take_along_axis(self.q1(obs), idx, axis=-1)[..., 0]
            q2 = jnp.take_along_axis(self.q2(obs), idx, axis=-1)[..., 0]
        else:
            x = jnp.concatenate([obs, action], axis=-1)
            q1 = self.q1(x)[..., 0]
            q2 = self.q2(x)[..., 0]
        return jnp.minimum(q1, q2), q1, q2


class InACAgent(nnx.Module):
    def __init__(
        self, obs_dim, action_dim, hidden_dim, discrete, action_range=1.0, *, rngs
    ):
        if discrete:
            self.pi = DiscretePolicy(obs_dim, action_dim, hidden_dim, rngs=rngs)
            self.beh_pi = DiscretePolicy(obs_dim, action_dim, hidden_dim, rngs=rngs)
        else:
            self.pi = GaussianPolicy(
                obs_dim, action_dim, hidden_dim, action_range, rngs=rngs
            )
            self.beh_pi = GaussianPolicy(
                obs_dim, action_dim, hidden_dim, action_range, rngs=rngs
            )
        self.q = DoubleQ(obs_dim, action_dim, hidden_dim, discrete, rngs=rngs)
        self.value_net = MLP(obs_dim, hidden_dim, 1, rngs=rngs)
        self.q_target = nnx.clone(self.q)
        self.discrete = discrete

    def __call__(self, obs):
        """(pi, value) interface of ``networks.ActorCritic`` for ``make_rollout``."""
        return self.pi(obs), jnp.zeros(obs.shape[:-1])


class InACOptimizers(nnx.Module):
    def __init__(self, agent: InACAgent, lr: float):
        self.pi = nnx.Optimizer(agent.pi, optax.adam(lr), wrt=nnx.Param)
        self.beh_pi = nnx.Optimizer(agent.beh_pi, optax.adam(lr), wrt=nnx.Param)
        self.q = nnx.Optimizer(agent.q, optax.adam(lr), wrt=nnx.Param)
        self.value = nnx.Optimizer(agent.value_net, optax.adam(lr), wrt=nnx.Param)


def make_train_state(config, env, env_params, rngs):
    action_space = env.action_space(env_params)
    discrete = isinstance(action_space, spaces.Discrete)
    action_dim = action_space.n if discrete else action_space.shape[0]
    obs_shape = env.observation_space(env_params).shape
    agent = InACAgent(
        obs_shape[0],
        action_dim,
        config["HIDDEN_DIM"],
        discrete,
        action_range=1.0 if discrete else float(jnp.max(action_space.high)),
        rngs=rngs,
    )
    optimizers = InACOptimizers(agent, config["LR"])
    normalize_vec_obs = NormalizeVecObs(jnp.zeros(obs_shape))
    normalize_vec_reward = NormalizeVecReward(
        jnp.zeros(config["NUM_ENVS"]), config["GAMMA"]
    )
    return (agent, optimizers, normalize_vec_obs, normalize_vec_reward)


def _polyak_update(source: nnx.Module, target: nnx.Module, polyak: float):
    """target <- polyak * target + (1 - polyak) * source."""
    new = optax.incremental_update(
        nnx.state(source, nnx.Param), nnx.state(target, nnx.Param), 1.0 - polyak
    )
    nnx.update(target, new)


def inac_update(agent: InACAgent, optimizers: InACOptimizers, batch, key, config):
    """One InAC step (inacax ``_update``): beta, V, Q, pi, then target sync.

    ``batch`` is ``(obs, action, reward, terminated, next_obs)``. Updates the
    modules in place and returns a dict of scalar losses/diagnostics.
    """
    obs, action, reward, terminated, next_obs = batch
    tau, gamma = config["TAU"], config["GAMMA"]
    key_v, key_q = jax.random.split(key)

    # Behaviour policy: maximum likelihood on dataset actions.
    def beta_loss_fn(beh_pi):
        return -beh_pi(obs).log_prob(action).mean()

    loss_beta, grads = nnx.value_and_grad(beta_loss_fn)(agent.beh_pi)
    optimizers.beh_pi.update(agent.beh_pi, grads)

    # V(s) -> E_{a~pi}[min Q_targ(s, a) - tau log pi(a|s)].
    pi_s = agent.pi(obs)
    a_pi, logp_pi = pi_s.sample_and_log_prob(seed=key_v)
    min_q_pi, _, _ = agent.q_target(obs, a_pi)
    v_target = jax.lax.stop_gradient(min_q_pi - tau * logp_pi)

    def value_loss_fn(value_net):
        v = value_net(obs)[..., 0]
        return (0.5 * (v - v_target) ** 2).mean(), v

    (loss_v, v), grads = nnx.value_and_grad(value_loss_fn, has_aux=True)(
        agent.value_net
    )
    optimizers.value.update(agent.value_net, grads)

    # Q(s, a) -> r + gamma (1 - term) E_{a'~pi}[min Q_targ(s', a') - tau log pi(a'|s')].
    next_a, next_logp = agent.pi(next_obs).sample_and_log_prob(seed=key_q)
    min_q_next, _, _ = agent.q_target(next_obs, next_a)
    q_target = jax.lax.stop_gradient(
        reward + gamma * (1.0 - terminated) * (min_q_next - tau * next_logp)
    )

    def q_loss_fn(q):
        min_q, q1, q2 = q(obs, action)
        loss = 0.5 * (
            (0.5 * (q_target - q1) ** 2).mean() + (0.5 * (q_target - q2) ** 2).mean()
        )
        return loss, min_q

    (loss_q, min_q), grads = nnx.value_and_grad(q_loss_fn, has_aux=True)(agent.q)
    optimizers.q.update(agent.q, grads)

    # Policy: in-sample softmax weights exp((Q - V)/tau - log beta), clipped.
    min_q_new, _, _ = agent.q(obs, action)
    value_new = agent.value_net(obs)[..., 0]
    beh_logp = agent.beh_pi(obs).log_prob(action)
    weight = jax.lax.stop_gradient(
        jnp.clip(
            jnp.exp((min_q_new - value_new) / tau - beh_logp),
            config["EPS"],
            config["EXP_THRESHOLD"],
        )
    )

    def pi_loss_fn(pi):
        return -(weight * pi(obs).log_prob(action)).mean()

    loss_pi, grads = nnx.value_and_grad(pi_loss_fn)(agent.pi)
    optimizers.pi.update(agent.pi, grads)

    _polyak_update(agent.q, agent.q_target, config["POLYAK"])

    return {
        "beta": loss_beta,
        "actor": loss_pi,
        "critic": loss_q,
        "value": loss_v,
        "q_info": min_q.mean(),
        "v_info": v.mean(),
        "logp_info": logp_pi.mean(),
    }


def _as_batch(obs, action, reward, terminated, next_obs, discrete):
    if discrete and action.ndim > reward.ndim:
        # collate_rollout stores discrete actions as (N, 1).
        action = action[..., 0]
    return (
        obs,
        action,
        reward.astype(jnp.float32),
        terminated.astype(jnp.float32),
        next_obs,
    )


def sample_batch(buffer, count, key, batch_size, discrete):
    """Uniformly sample ``batch_size`` rows from the first ``count`` of ``buffer``.

    ``buffer`` is a ``ppo.Transition`` (real-env dataset) or a dict with the
    same field names (``make_train``'s replay).
    """
    idx = jax.random.randint(key, (batch_size,), 0, count)
    if isinstance(buffer, dict):
        obs, action, reward = buffer["obs"], buffer["action"], buffer["reward"]
        terminated, next_obs = buffer["terminated"], buffer["next_obs"]
    else:
        obs, action, reward = buffer.obs, buffer.action, buffer.reward
        terminated, next_obs = buffer.terminated, buffer.info["next_obs"]
    return _as_batch(
        obs[idx], action[idx], reward[idx], terminated[idx], next_obs[idx], discrete
    )


def _scan_updates(agent, optimizers, buffer, count, key, num_updates, config):
    discrete = agent.discrete

    def _step(carry, key):
        agent, optimizers = carry
        key_sample, key_update = jax.random.split(key)
        batch = sample_batch(buffer, count, key_sample, config["BATCH_SIZE"], discrete)
        losses = inac_update(agent, optimizers, batch, key_update, config)
        return (agent, optimizers), losses

    keys = jax.random.split(key, num_updates)
    _, losses = nnx.scan(_step)((agent, optimizers), keys)
    return losses


def tail_mean(series):
    """Mean of the last 10% of a per-update loss series."""
    return jnp.mean(series[-max(1, series.shape[0] // 10) :])


def make_offline_train(config):
    """``train(train_state, dataset, count, rng) -> (train_state, losses)``:
    ``OFFLINE_UPDATES`` InAC steps on the first ``count`` rows of ``dataset``."""

    def train(train_state, dataset, count, rng):
        agent, optimizers, _, _ = train_state
        losses = _scan_updates(
            agent, optimizers, dataset, count, rng, config["OFFLINE_UPDATES"], config
        )
        return train_state, losses

    return train


def make_train(env, config):
    """InAC trained online against ``env`` (vectorized, ``NUM_ENVS`` copies).

    The first ``LEARNING_STARTS // NUM_ENVS`` iterations only collect; each
    later iteration steps every env once, writes the transitions into a
    circular ``BUFFER_SIZE`` replay, then runs ``UPDATES_PER_ITER`` updates.
    Metrics are ``(NUM_ITERS, NUM_ENVS)``, i.e. PPO's layout with NUM_STEPS=1.
    """
    num_envs = config["NUM_ENVS"]
    buffer_size = config["BUFFER_SIZE"]
    warmup_iters = min(config["LEARNING_STARTS"] // num_envs, config["NUM_ITERS"])
    learn_iters = config["NUM_ITERS"] - warmup_iters

    def train(train_state, env_params, rng):
        agent, optimizers, _, _ = train_state

        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(jax.random.split(_rng, num_envs), env_params)
        rng, _rng = jax.random.split(rng)
        dummy_action = agent.pi(obsv).sample(seed=_rng)
        buffer = {
            "obs": jnp.zeros((buffer_size,) + obsv.shape[1:], obsv.dtype),
            "action": jnp.zeros(
                (buffer_size,) + dummy_action.shape[1:], dummy_action.dtype
            ),
            "reward": jnp.zeros(buffer_size),
            "terminated": jnp.zeros(buffer_size, dtype=bool),
            "next_obs": jnp.zeros((buffer_size,) + obsv.shape[1:], obsv.dtype),
        }

        def make_iter(learn):
            def _iter(carry, unused):
                agent, optimizers, env_state, obs, buffer, ptr, count, rng = carry
                rng, key_action, key_step, key_update = jax.random.split(rng, 4)
                action = agent.pi(obs).sample(seed=key_action)
                obsv, env_state, reward, terminated, truncated, info = env.step(
                    jax.random.split(key_step, num_envs), env_state, action, env_params
                )
                idx = (ptr + jnp.arange(num_envs)) % buffer_size
                new = {
                    "obs": obs,
                    "action": action,
                    "reward": reward,
                    "terminated": terminated,
                    "next_obs": info["next_obs"],
                }
                buffer = jax.tree.map(lambda b, x: b.at[idx].set(x), buffer, new)
                ptr = (ptr + num_envs) % buffer_size
                count = jnp.minimum(count + num_envs, buffer_size)
                if learn:
                    losses = _scan_updates(
                        agent,
                        optimizers,
                        buffer,
                        count,
                        key_update,
                        config["UPDATES_PER_ITER"],
                        config,
                    )
                    losses = jax.tree.map(jnp.mean, losses)
                else:
                    losses = None
                metric = {
                    "returned_episode_returns": info["returned_episode_returns"],
                    "returned_episode": info["returned_episode"],
                }
                carry = (agent, optimizers, env_state, obsv, buffer, ptr, count, rng)
                return carry, (metric, losses)

            return _iter

        carry = (
            agent,
            optimizers,
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
