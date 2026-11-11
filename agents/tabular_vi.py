import jax
import jax.numpy as jnp
from flax import nnx


def make_batched_solve_vi(model_env, num_cells, num_actions, horizon, gamma):
    real_env = model_env._real_env
    cells = jnp.eye(num_cells)

    def solve(model, env_params, key, alpha, beta):
        z = model.sample_index(key, model_env.samples)

        def one(obs, action):
            x = model.single_input(obs, action)
            if model_env.prediction_mode == "sample":
                probs = jax.nn.softmax(model.predict_sample(x, z[0]), -1)
            else:
                probs = model.predict_mean(x) + obs
            bonus = model.uncertainty(x, z, "eig")
            reward, terminated = real_env.obs_to_reward_terminated(
                obs, action, obs, env_params
            )
            return probs, bonus, reward, terminated

        def per_action(a):
            return jax.lax.map(
                lambda o: one(o, jnp.int32(a)), cells, batch_size=128
            )

        probs, bonus, reward, terminated = jax.tree.map(
            lambda *xs: jnp.stack(xs, 1), *[per_action(a) for a in range(num_actions)]
        )
        cont = 1.0 - terminated.astype(jnp.float32)

        def q_values(r):
            def backup(v, _):
                q = r + gamma * cont * (probs @ v)
                return q.max(-1), q

            _, qs = jax.lax.scan(backup, jnp.zeros(num_cells), None, length=horizon)
            return qs[-1]

        return q_values(alpha * reward + beta * bonus), q_values(reward)

    return nnx.jit(nnx.vmap(solve, in_axes=(0, 0, 0, None, None)))
