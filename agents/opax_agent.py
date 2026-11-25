import dataclasses
import math
import os
import time
from typing import Any, ClassVar, NamedTuple

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import struct
from gymnax.environments import environment, spaces

from agents.base import Env, Experience
from batching import seed_slice
from config import OpaxConfig
from logger import log_time
from opax.models.active_learning_model import ActiveLearningHUCRLModel
from opax.optimizers.dummy_policy_optimizer import BestSequences
from opax.optimizers.icem_trajectory_optimizer import ICemTO
from opax.utils.replay_buffer import Transition as OpaxTransition
from opax.utils.type_aliases import ModelProperties, default_model_properties
from ppo import Transition

# Floor on the OPAX normalizer's std (constant dims, e.g. before data varies).
NORMALIZER_EPS = 1e-6


class ActionCodec:
    """Converts actions between iCEM's plan space, the model's input and the env.

    Three spaces:
      plan  : what iCEM samples, unit scale [-1, 1], shape (..., act_dim + obs_dim).
              The first act_dim entries are the real action, the rest is HUCRL's eta.
      model : the applied action as the dynamics model sees it, shape (..., act_dim).
              Box -> unit scale [-1, 1]; Discrete -> centre of the chosen bin (act_dim = 1).
      env   : what env.step takes and the buffer stores.
              Box -> [low, high], shape (..., act_dim); Discrete -> int32 index, shape (...).

    Discrete actions use n equal bins over [-1, 1], so they stay a single ordered input
    (fine for CartPole / Acrobot / MountainCar). For unordered actions, switch snap and
    from_buffer to one-hot; nothing else changes.
    """

    def __init__(self, action_space: spaces.Discrete | spaces.Box):
        self.discrete = isinstance(action_space, spaces.Discrete)
        if self.discrete:
            self.n = int(action_space.n)
            self.act_dim = 1
        else:
            shape = tuple(action_space.shape)
            self.act_dim = math.prod(shape)
            self.low = jnp.broadcast_to(jnp.asarray(action_space.low, jnp.float32), shape)
            self.high = jnp.broadcast_to(
                jnp.asarray(action_space.high, jnp.float32), shape
            )
            if not (jnp.all(jnp.isfinite(self.low)) and jnp.all(jnp.isfinite(self.high))):
                raise ValueError("ActionCodec needs a Box action space with finite bounds")

    def _bin_index(self, u: jax.Array) -> jax.Array:
        i = jnp.floor((u + 1.0) / 2.0 * self.n)
        return jnp.clip(i, 0, self.n - 1).astype(jnp.int32)

    def _bin_center(self, i: jax.Array) -> jax.Array:
        return 2.0 * (i.astype(jnp.float32) + 0.5) / self.n - 1.0

    def snap(self, plan: jax.Array) -> jax.Array:
        """Plan space -> plan space with the real-action part replaced by the action that
        will actually be applied. Eta (entries past act_dim) passes through unchanged."""
        if not self.discrete:
            return plan
        center = self._bin_center(self._bin_index(plan[..., :1]))
        return plan.at[..., :1].set(center)

    def to_env(self, action: jax.Array) -> jax.Array:
        """Model space (..., act_dim) -> env action."""
        action = action[..., : self.act_dim]
        if self.discrete:
            return self._bin_index(action[..., 0])
        unit = jnp.clip(action, -1.0, 1.0)
        return self.low + (unit + 1.0) / 2.0 * (self.high - self.low)

    def from_buffer(self, action: jax.Array) -> jax.Array:
        """Env action in the buffer layout -> model space (..., act_dim).

        Buffer layout (data.collate_rollout / _batch_action): Box (..., act_dim);
        Discrete (..., 1) int indices.
        """
        if self.discrete:
            return self._bin_center(action[..., 0])[..., None]
        return 2.0 * (action - self.low) / (self.high - self.low) - 1.0


class OracleReward:
    """The reward model OPAX's dynamics model calls during planning.

    BayesianDynamicsModel/HUCRLModel call ``predict(obs, action, next_obs, rng)`` once per
    particle with unbatched obs (obs_dim,) and a model-space action (act_dim,) that is
    already snapped. The env's own reward oracle (``obs_to_reward_terminated``, from the
    state-reconstruction wrappers or the Brax wrapper) gives the reward; termination is
    dropped because iCEM ignores it.

    ``env_params`` must be a single (unbatched) params object, e.g.
    ``seed_slice(env_params, 0)``. The model's jitted evaluate functions capture this
    object at construction, outside any vmap over seeds, so it cannot be per-seed. That
    is fine because env params only differ per seed for DeepSea, which OPAX rejects.
    """

    def __init__(self, env: Env, env_params: environment.EnvParams, codec: ActionCodec):
        if not hasattr(env, "obs_to_reward_terminated"):
            raise ValueError(
                "OPAX needs an env with obs_to_reward_terminated (a state-reconstruction "
                "wrapper or a Brax wrapper that defines it)."
            )
        self.env = env
        self.env_params = env_params
        self.codec = codec

    def predict(
        self,
        obs: jax.Array,
        action: jax.Array,
        next_obs: jax.Array,
        rng: jax.Array | None = None,
    ) -> jax.Array:
        reward, _ = self.env.obs_to_reward_terminated(
            obs, self.codec.to_env(action), next_obs, self.env_params
        )
        return jnp.asarray(reward, jnp.float32)


class SnappedHUCRLModel(ActiveLearningHUCRLModel):
    """ActiveLearningHUCRLModel whose planning entry points see the applied action.

    iCEM samples continuous plans; ``evaluate`` (eval planner, mean model + oracle reward)
    and ``evaluate_for_exploration`` (explore planner, OPAX objective + optimism) first
    replace the real-action part of each plan step with ``codec.snap`` (the action the env
    will actually receive), so planning rolls out the same actions the model is trained
    on. Eta passes through unchanged.

    Done in ``_init_fn`` because every constructor in the parent chain calls it and
    rebuilds these two functions; the most-derived override runs last, so the wrap always
    survives. Each wrapper is jitted and assigned once per call, which keeps its identity
    stable: ``ICemTO._optimize_with_params`` takes ``evaluate_fn`` as a static jit arg.

    Only these two functions are snapped. Snap the action yourself before calling
    ``predict``, ``predict_with_mean`` or ``predict_with_uncertainty`` directly.
    """

    def __init__(self, codec: ActionCodec, *args, **kwargs):
        # Set before super().__init__: the parent constructors call _init_fn.
        self.codec = codec
        super().__init__(*args, **kwargs)

    def _init_fn(self):
        super()._init_fn()
        snap = self.codec.snap
        evaluate = self.evaluate
        evaluate_for_exploration = self.evaluate_for_exploration

        def snapped_evaluate(
            parameters,
            obs,
            action,
            rng,
            model_props: ModelProperties = default_model_properties,
            sampling_idx=None,
        ):
            return evaluate(
                parameters,
                obs,
                snap(action),
                rng,
                model_props=model_props,
                sampling_idx=sampling_idx,
            )

        def snapped_evaluate_for_exploration(
            parameters,
            obs,
            action,
            rng,
            model_props: ModelProperties = default_model_properties,
            sampling_idx=None,
        ):
            return evaluate_for_exploration(
                parameters,
                obs,
                snap(action),
                rng,
                model_props=model_props,
                sampling_idx=sampling_idx,
            )

        self.evaluate = jax.jit(snapped_evaluate)
        self.evaluate_for_exploration = jax.jit(snapped_evaluate_for_exploration)


def initial_model_props(
    obs_dim: int, act_dim: int, batch_shape: tuple[int, ...] = ()
) -> ModelProperties:
    #starts a model with identity transformations on the normalizer,
    #vmappable
    obs_shape = (*batch_shape, obs_dim)
    act_shape = (*batch_shape, act_dim)
    return ModelProperties(
        alpha=jnp.ones(obs_shape, jnp.float32),
        bias_obs=jnp.zeros(obs_shape, jnp.float32),
        bias_act=jnp.zeros(act_shape, jnp.float32),
        bias_out=jnp.zeros(obs_shape, jnp.float32),
        scale_obs=jnp.ones(obs_shape, jnp.float32),
        scale_act=jnp.ones(act_shape, jnp.float32),
        scale_out=jnp.ones(obs_shape, jnp.float32),
    )


class OpaxState(NamedTuple):
	model_params: Any
	model_opt_state: Any
	model_props: ModelProperties

@struct.dataclass(frozen=True)
class OpaxPolicy:
    model_params: Any
    model_props: ModelProperties
    # Static (part of the pytree structure, not a leaf) so the rollout can branch on it
    # with a plain `if` and jit compiles explore and eval rollouts separately.
    evaluation: bool = struct.field(pytree_node=False, default=False)

class OpaxAgent:
    """OPAX: optimistic active exploration with an MPC (iCEM) planner.

    Exploration plans with the OPAX objective (sum log(1 + sigma_ep^2 / sigma_al^2)) and
    HUCRL optimism; evaluation plans for the env's oracle reward with the mean model.
    The model is a probabilistic ensemble retrained on the whole buffer every update.
    """

    on_policy: ClassVar[bool] = False

    def __init__(self, config: OpaxConfig, env, env_params, obs_dim, loggers):
        # Env params are identical across seeds for every env OPAX supports (only
        # DeepSea differs, and it is rejected), so one slice serves all seeds.
        single_env_params = seed_slice(env_params, 0)
        action_remapper = ActionCodec(env.action_space(single_env_params))
        self.act_dim = action_remapper.act_dim  # 1 for Discrete (bin centre), not n
        self.obs_dim = obs_dim
        self.planning_dim = self.act_dim + obs_dim  # real action, then HUCRL's eta per obs dim

        self.codec = action_remapper
        self.model = SnappedHUCRLModel(
            codec=action_remapper,
            obs_dim=obs_dim,
            act_dim=self.act_dim,
            reward_model=OracleReward(env, single_env_params, action_remapper),
            model_class="ProbabilisticEnsembleModel",
            num_ensemble=config.num_ensembles,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            deterministic=config.deterministic_env,
            pred_diff=config.pred_diff,
            beta=config.beta,
        )
    
        self.optimizer = ICemTO(
            horizon=config.horizon,
            action_dim=(self.planning_dim,),
            dynamics_model_list=[self.model],
            n_particles=config.n_particles,
            num_samples=config.num_samples,
            num_elites=config.num_elites,
            init_std=config.init_std,
            alpha=config.alpha,
            num_steps=config.num_steps,
            elite_set_fraction=config.elite_set_fraction,
        )

        self.config = config
        self.loggers = loggers
        # One compiled update for all seeds: per-seed core vmapped over the seed axis
        # (buffer arrays, state and keys batched; size shared).
        self._update = jax.jit(jax.vmap(self._update_one, in_axes=(0, 0, 0, 0, None, 0)))

    def init(self, keys: jax.Array) -> OpaxState:
        """keys: (B,) seed keys -> OpaxState with a leading B axis on every leaf."""
        return jax.vmap(self.initialize_model)(keys)

    #initializes an OpaxState with identity scaling and default parameters
    def initialize_model(self, key: jax.Array):
        props = initial_model_props(self.obs_dim, self.act_dim, ())
        params = self.model.init_params(key)
        opt_state = self.model.init_opt_state(params)
        return OpaxState(params, opt_state, props)
        
        
    def explore_policy(self, agent_state: OpaxState) -> "OpaxPolicy":
        return self.policy(agent_state, evaluation=False)

    def eval_policy(self, agent_state: OpaxState) -> "OpaxPolicy":
        return self.policy(agent_state, evaluation=True)

    def policy(self, agent_state: OpaxState, evaluation: bool) -> "OpaxPolicy":
        #no planning here, just a way to 
        return OpaxPolicy(
            model_params=agent_state.model_params,
            model_props=agent_state.model_props,
            evaluation=evaluation,
        )




    def make_rollout(self, config, env, env_params, training=True):
        """Real-env rollout that plans with iCEM at every step (MPC).

        Same contract as ppo.make_rollout, so run_online and evaluation can use it:
        runner_state = (policy, env_state, obs, rng) for ONE seed (the callers nnx.vmap
        over seeds), obs (E, obs_dim) with E envs batched by VecEnv. Returns the updated
        runner_state and a ppo.Transition with (T, E, ...) leaves, T = config["NUM_STEPS"].

        ``training`` is ignored: OPAX has no in-rollout normalizer updates, and whether to
        explore or exploit is ``policy.evaluation`` (static), not this flag.
        """
        model_optimizer = self.optimizer
        codec = self.codec
        horizon = model_optimizer.horizon
        planning_dim = self.planning_dim

        def rollout(runner_state):
            policy, env_state, obs, rng = runner_state
            num_envs = obs.shape[0]

            def plan_one(obs_e, warm_e, key_e):
                """Plan for one env. obs_e (obs_dim,), warm_e that env's BestSequences.

                The planner vmaps over a batch of observations but keeps only the first
                one's warm start, so it is called with a batch of 1 per env, and envs are
                vmapped outside. Returns the first planned step (planning_dim,) and the
                env's updated warm start.
                """
                particle_key, optimizer_key = jax.random.split(key_e)
                plan_kwargs = {
                    "dynamics_params": policy.model_params,
                    "obs": obs_e[None],
                    "best_sequences": warm_e,
                    "key": particle_key[None],
                    "optimizer_key": optimizer_key[None],
                    "model_props": policy.model_props,
                }
                # Static flag: only one planner is traced into each compiled rollout.
                # Keyword-only call: optimize is a partial with model_index bound by name.
                if policy.evaluation:
                    sequence, _, warm_e = model_optimizer.optimize(**plan_kwargs)
                else:
                    sequence, _, warm_e = model_optimizer.optimize_for_exploration(
                        **plan_kwargs
                    )
                return sequence[0, 0], warm_e

            def env_step(carry, unused):
                warm, env_state, obs, rng = carry
                rng, plan_rng, step_rng = jax.random.split(rng, 3)

                # PLAN: one independent iCEM optimization per env.
                plan_keys = jax.random.split(plan_rng, num_envs)
                first_step, warm = jax.vmap(plan_one)(obs, warm, plan_keys)

                # ACT: drop eta, snap to the applied action, map to env scale.
                action = codec.to_env(codec.snap(first_step))

                # STEP ENV (VecEnv is batched over E).
                step_keys = jax.random.split(step_rng, num_envs)
                next_obs, env_state, reward, terminated, truncated, info = env.step(
                    step_keys, env_state, action, env_params
                )

                # The env auto-resets on done, so that env's old plan no longer applies.
                done = jnp.logical_or(terminated, truncated)
                warm = jax.tree.map(
                    lambda w: jnp.where(
                        done.reshape((num_envs,) + (1,) * (w.ndim - 1)), 0.0, w
                    ),
                    warm,
                )

                # PPO-only fields are zero; obs/action/reward equal their raw versions.
                zeros = jnp.zeros_like(reward)
                transition = Transition(
                    terminated,
                    truncated,
                    action,
                    zeros,  # value
                    zeros,  # next_value
                    reward,
                    zeros,  # log_prob
                    obs,
                    info,
                    obs,  # raw_obs
                    action,  # raw_action
                    reward,  # raw_reward
                )
                return (warm, env_state, next_obs, rng), transition

            # Fresh warm start for every rollout: zeros of the planner's sequence shape,
            # one per env (eval keeps one sequence per model; there is one model).
            warm = BestSequences(
                evaluation_sequences=jnp.zeros(
                    (num_envs, 1, horizon, planning_dim), jnp.float32
                ),
                exploration_sequence=jnp.zeros(
                    (num_envs, horizon, planning_dim), jnp.float32
                ),
            )
            (_, env_state, obs, rng), traj_batch = jax.lax.scan(
                env_step, (warm, env_state, obs, rng), None, length=config["NUM_STEPS"]
            )
            return (policy, env_state, obs, rng), traj_batch

        return rollout

    def update(
        self, agent_state: OpaxState, experience: Experience, key: jax.Array
    ) -> OpaxState:
        """Refit the ensemble and the OPAX normalizer on the whole buffer, per seed.

        key: (B,) per-seed keys (run_online passes vsplit(key)).
        """
        step = experience.size
        buffer = experience.buffer
        t0 = time.perf_counter()
        new_state, metrics = self._update(
            agent_state,
            buffer.obs,
            buffer.action,
            buffer.info["final_observation"],
            jnp.asarray(experience.size),
            key,
        )
        jax.block_until_ready(new_state)
        log_time(self.loggers, "time/model_train_s", t0, step)
        for b, logger in enumerate(self.loggers):
            logger.log_scalar("opax/train_nll", metrics["train_nll"][b, -1], step)
            logger.log_scalar("opax/grad_norm", metrics["grad_norm"][b, -1], step)
            logger.log_scalar("opax/val_mse", metrics["val_mse"][b], step)
            logger.log_scalar("opax/val_mse_no_change", metrics["val_mse_no_change"][b], step)
            logger.log_scalar("opax/calibration_alpha", metrics["alpha_mean"][b], step)
            logger.log_scalar(
                "opax/calibration_error", metrics["calibration_error"][b], step
            )
        return new_state

    def _update_one(
        self,
        state: OpaxState,
        obs: jax.Array,
        action: jax.Array,
        next_obs: jax.Array,
        size: jax.Array,
        key: jax.Array,
    ) -> tuple[OpaxState, dict[str, jax.Array]]:
        """One seed. obs/next_obs (N, obs_dim), action in buffer layout (N, act_dim) or
        (N, 1) ints for Discrete; only the first `size` rows are filled."""
        config = self.config
        reset_key, train_key, val_key = jax.random.split(key, 3)
        params, opt_state = state.model_params, state.model_opt_state
        if config.reset_model:  # static: OPAX retrains from scratch every update
            params = self.model.init_params(reset_key)
            opt_state = self.model.init_opt_state(params)

        # Model space: unit-scale (Box) or bin-centre (Discrete) actions. The model
        # predicts the change in obs when pred_diff, else the next obs.
        action = self.codec.from_buffer(action).astype(jnp.float32)
        target = next_obs - obs if config.pred_diff else next_obs

        # OPAX normalizer from the filled rows only (masked mean/std, as in
        # models/base.py:update_stats); recomputed from scratch, not running stats.
        mask = (jnp.arange(obs.shape[0]) < size)[:, None]

        def mean_std(x):
            mean = jnp.mean(x, axis=0, where=mask)
            std = jnp.maximum(jnp.std(x, axis=0, where=mask), NORMALIZER_EPS)
            return mean.astype(jnp.float32), std.astype(jnp.float32)

        props = state.model_props
        if config.normalize:
            bias_obs, scale_obs = mean_std(obs)
            bias_out, scale_out = mean_std(target)
        else:
            bias_obs, scale_obs = props.bias_obs, props.scale_obs
            bias_out, scale_out = props.bias_out, props.scale_out
        # Actions stay identity: model-space actions are already in [-1, 1], and a
        # data std would collapse to the floor whenever the buffer holds one action
        # value (e.g. a single discrete action), blowing up every other planned action.
        bias_act, scale_act = props.bias_act, props.scale_act

        # Training data is normalized here: BayesianDynamicsModel._train uses it as
        # given. (Planning passes raw obs; the model normalizes internally with props.)
        obs_n = (obs - bias_obs) / scale_obs
        act_n = (action - bias_act) / scale_act
        target_n = (target - bias_out) / scale_out
        batch_size = config.transitions_per_update

        def sample(k):
            idx = jax.random.randint(k, (batch_size,), 0, size)
            return obs_n[idx], act_n[idx], target_n[idx]

        def train_step(carry, step_key):
            params, opt_state = carry
            o, a, y = sample(step_key)
            tran = OpaxTransition(
                obs=o,
                action=a,
                next_obs=y,
                reward=jnp.zeros(batch_size),
                done=jnp.zeros(batch_size),
            )
            params, opt_state, _, summary = self.model._train_step(
                tran, params, opt_state
            )
            return (params, opt_state), (summary.model_likelihood, summary.grad_norm)

        (params, opt_state), (train_nll, grad_norm) = jax.lax.scan(
            train_step,
            (params, opt_state),
            jax.random.split(train_key, config.train_steps_per_model_update),
        )

        # Calibration once after training (it searches ~100 alphas), on a fresh sample.
        # In-sample: OPAX trains on the whole buffer, so there is no held-out split.
        val_o, val_a, val_y = sample(val_key)
        val_x = jnp.concatenate([val_o, val_a], axis=-1)
        alpha, calibration_error = self.model.model.calculate_calibration_alpha(
            params, val_x, val_y
        )
        # Sanity metric: the mean model should beat "obs does not change".
        mean_pred, _ = jnp.split(self.model.model._predict(params, val_x), 2, axis=-1)
        no_change = (jnp.zeros_like(bias_out) - bias_out) / scale_out
        if not config.pred_diff:
            no_change = (val_o * scale_obs + bias_obs - bias_out) / scale_out
        metrics = {
            "train_nll": train_nll,
            "grad_norm": grad_norm,
            "val_mse": jnp.mean(jnp.square(mean_pred.mean(axis=0) - val_y)),
            "val_mse_no_change": jnp.mean(jnp.square(no_change - val_y)),
            "alpha_mean": alpha.mean(),
            "calibration_error": calibration_error.mean(),
        }

        new_props = ModelProperties(
            alpha=alpha.astype(jnp.float32),
            bias_obs=bias_obs,
            bias_act=bias_act,
            bias_out=bias_out,
            scale_obs=scale_obs,
            scale_act=scale_act,
            scale_out=scale_out,
        )
        return OpaxState(params, opt_state, new_props), metrics

    def save(self, agent_state: OpaxState, seed_dir: str, b: int) -> None:
        """Model parameters and normalizer/calibration for seed b, under
        seed_dir/checkpoint/opax_model/{params,props}."""
        state = seed_slice(agent_state, b)
        base = os.path.join(os.path.abspath(seed_dir), "checkpoint", "opax_model")
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(os.path.join(base, "params"), state.model_params)
        checkpointer.save(
            os.path.join(base, "props"), dataclasses.asdict(state.model_props)
        )
        checkpointer.wait_until_finished()
        


