import jax
import jax.numpy as jnp
from flax import nnx


def make_batched_solve_vi(model_env, num_states, num_actions, horizon, gamma):
    """Exact finite-horizon value iteration in the learned categorical model.

    Observations are Discrete state indices, so the whole transition tensor
    ``probs[s, a, s']`` is enumerated from the model and backed up in closed form.
    Returns ``solve(models, env_params, keys, alpha, beta) -> (q_explore, q_eval)``,
    vmapped over the seed axis. The explore Q uses ``alpha * reward + beta * bonus``;
    the eval Q uses the reward alone.
    """
    real_env = model_env._real_env
    states = jnp.arange(num_states)
    zero_action = jnp.zeros_like(model_env._zero_action)

    def solve(model, env_params, key, alpha, beta):
        z = model.sample_index(key, model_env.samples)

        def one(obs, action):
            x = model.single_input(obs, action)
            if model_env.prediction_mode == "sample":
                probs = jax.nn.softmax(model.predict_sample(x, z[0]), -1)
            else:
                probs = model.predict_mean(x)
            bonus = model.uncertainty(x, z, model_env.explore_bonus)
            # Oracle reward / termination, as in ModelEnvironment.step_env. DeepSea's
            # depend only on (state, action), so time and the last action are inert.
            state = real_env.get_state(obs, zero_action, 0, params=env_params)
            _, _, reward, terminated, _ = real_env.step_env(
                jax.random.key(0), state, action, env_params
            )
            return probs, bonus, reward, terminated

        def per_action(a):
            return jax.lax.map(lambda o: one(o, jnp.int32(a)), states, batch_size=128)

        probs, bonus, reward, terminated = jax.tree.map(
            lambda *xs: jnp.stack(xs, 1), *[per_action(a) for a in range(num_actions)]
        )
        cont = 1.0 - terminated.astype(jnp.float32)

        def q_values(r):
            def backup(v, _):
                q = r + gamma * cont * (probs @ v)
                return q.max(-1), q

            _, qs = jax.lax.scan(backup, jnp.zeros(num_states), None, length=horizon)
            return qs[-1]

        return q_values(alpha * reward + beta * bonus), q_values(reward)

    return nnx.jit(nnx.vmap(solve, in_axes=(0, 0, 0, None, None)))
