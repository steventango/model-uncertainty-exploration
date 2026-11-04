import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from gymnax.environments import environment

from agents.base import Config, Env, Experience, PolicyState, make_batched_policy
from agents.model_free import ModelFreeAgent
from batching import vsplit
from config import RNDConfig
from logger import ExperimentLogger
from networks import Critic
from normalization import NormalizeVecReward
from ppo import calculate_gae
from rnd import RND, make_rnd, make_train_rnd


class RNDModelFreeState(NamedTuple):
    policy: PolicyState
    rnd: RND
    # Value of the intrinsic return, on PPO-normalized observations.
    int_critic: Critic
    int_critic_optimizer: nnx.Optimizer
    # Running std of the non-episodic intrinsic return (one real env).
    int_reward_norm: NormalizeVecReward


def normalize_intrinsic(reward_norm, reward):
    """Divide (T, E) bonuses by the running std of the non-episodic intrinsic return."""

    def step(reward_norm, r):
        never = jnp.zeros_like(r, dtype=bool)
        return reward_norm, reward_norm(r, never, never)

    _, normalized = nnx.scan(step)(reward_norm, reward)
    return normalized


class RNDModelFreeAgent(ModelFreeAgent):
    """PPO+RND (Burda et al., 2018): one policy on extrinsic + intrinsic advantages.

    The intrinsic return gets its own critic and is non-episodic (ignores
    terminations and truncations).
    """

    def __init__(
        self,
        config: Config,
        env: Env,
        env_params: environment.EnvParams,
        loggers: list[ExperimentLogger],
        solver: str,
        rnd_config: RNDConfig,
    ):
        super().__init__(config, env, env_params, loggers, solver)
        self.rnd_config = rnd_config
        train_rnd = make_train_rnd(rnd_config)

        def train(
            train_state,
            rnd,
            int_critic,
            int_critic_optimizer,
            int_reward_norm,
            traj_batch,
            buffer,
            count,
            rng,
        ):
            rng_ppo, rng_rnd = jax.random.split(rng)
            next_obs = traj_batch.info["next_obs"]
            rnd.update_stats(buffer.info["next_obs"], count)
            raw_bonus = rnd(next_obs)
            bonus = normalize_intrinsic(int_reward_norm, raw_bonus)
            if config["NORMALIZE_ENV"]:
                next_obs = train_state[2].normalize(next_obs)
            never = jnp.zeros_like(traj_batch.terminated)
            int_batch = traj_batch._replace(
                reward=bonus,
                value=int_critic(traj_batch.obs),
                next_value=int_critic(next_obs),
                terminated=never,
                truncated=never,
            )
            int_advantages, int_targets = calculate_gae(int_batch, self.online_config)
            _, (total, (value, actor, entropy)) = self.ppo_update(
                train_state,
                traj_batch,
                rng_ppo,
                extra_advantages=rnd_config.int_coef * int_advantages,
            )

            def critic_step(carry, _):
                critic, optimizer = carry
                loss, grads = nnx.value_and_grad(
                    lambda c: (
                        0.5 * jnp.mean(jnp.square(c(traj_batch.obs) - int_targets))
                    )
                )(critic)
                optimizer.update(critic, grads)
                return (critic, optimizer), loss

            _, int_value = nnx.scan(critic_step, length=config["UPDATE_EPOCHS"])(
                (int_critic, int_critic_optimizer), None
            )
            rnd_losses = train_rnd(rnd, buffer.info["next_obs"], count, rng_rnd)
            losses = {
                "total": total,
                "value": value,
                "actor": actor,
                "entropy": entropy,
                "int_value": int_value,
                "rnd_loss": rnd_losses[-1],
                "bonus": raw_bonus,
            }
            return jax.tree_util.tree_map(jnp.mean, losses)

        self._train_rnd = nnx.jit(
            nnx.vmap(train, in_axes=(0, 0, 0, 0, 0, 0, 0, None, 0))
        )

    def init(self, key: jax.Array) -> RNDModelFreeState:
        policy_key, rnd_key = vsplit(key)
        policy = make_batched_policy(self.config, self.env, self.env_params, policy_key)
        obs_dim = self.env.observation_space(self.env_params).shape[0]

        @nnx.vmap
        def build(key):
            rngs = nnx.Rngs(params=key)
            rnd = make_rnd(self.rnd_config, obs_dim, rngs)
            critic = Critic(obs_dim, self.rnd_config.hidden_dim, rngs=rngs)
            optimizer = nnx.Optimizer(
                critic, optax.adam(self.rnd_config.lr), wrt=nnx.Param
            )
            reward_norm = NormalizeVecReward(jnp.zeros(1), self.config["GAMMA"])
            return rnd, critic, optimizer, reward_norm

        return RNDModelFreeState(policy, *build(rnd_key))

    def update(
        self, agent_state: RNDModelFreeState, experience: Experience, key: jax.Array
    ) -> RNDModelFreeState:
        t0 = time.perf_counter()
        losses = self._train_rnd(
            *agent_state,
            experience.latest,
            experience.buffer,
            jnp.asarray(experience.size),
            key,
        )
        self._log_losses(losses, t0, experience.size)
        return agent_state
