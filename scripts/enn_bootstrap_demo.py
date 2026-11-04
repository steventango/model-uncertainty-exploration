"""ENN bootstrap: fresh c every minibatch (old) vs fixed per-point c_i.

1-D regression from the GPJax regression tutorial: y = sin(4x) + cos(2x) + N(0, 0.3^2),
x ~ U(-3, 3). x is fed as ``obs`` and y as the obs delta, so the repo's ENN model and
train step are used unchanged. The old behaviour (Gaussian, sigma=1, fresh c) only swaps
the row ids that ``ENNModel.bootstrap_c`` is keyed by for fresh random ids every step.

    JAX_PLATFORMS=cpu uv run python scripts/enn_bootstrap_demo.py
"""

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ENNConfig, model_config_dict
from models.enn import _make_batched_model, _train_step

ARMS = (
    ("fresh c, gaussian (old)", True, "gaussian", "#eb6834"),
    ("fixed c_i, gaussian", False, "gaussian", "#2a78d6"),
    ("fixed c_i, bernoulli", False, "bernoulli", "#1baf7a"),
)


class Data(NamedTuple):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    terminated: jax.Array
    info: dict


def f(x):
    return jnp.sin(4 * x) + jnp.cos(2 * x)


def make_data(key, n, noise=0.3):
    kx, ky = jax.random.split(key)
    x = jax.random.uniform(kx, (n, 1), minval=-3.0, maxval=3.0)
    y = f(x) + noise * jax.random.normal(ky, x.shape)
    return Data(
        obs=x,
        action=jnp.zeros((n, 0)),
        reward=jnp.zeros(n),
        terminated=jnp.zeros(n, bool),
        info={"next_obs": x + y},
    )


def make_train(fresh_c, update_steps, minibatch_size):
    """models.enn._train_model, with optionally fresh bootstrap row ids every step."""

    def core(model, optimizer, metrics, dataset, pointer, rngs):
        model.update_stats(dataset, pointer)

        def step(state, _):
            model, optimizer, metrics, rngs = state
            indices = jax.random.randint(rngs(), (minibatch_size,), 0, pointer)
            minibatch = jax.tree_util.tree_map(lambda x: x[indices], dataset)
            c_rows = (
                jax.random.randint(rngs(), (minibatch_size,), 0, 2**31 - 1)
                if fresh_c
                else indices
            )
            metrics.reset()
            _train_step(model, optimizer, metrics, rngs, minibatch, c_rows)
            return (model, optimizer, metrics, rngs), None

        nnx.scan(step, length=update_steps)((model, optimizer, metrics, rngs), None)

    return nnx.jit(nnx.vmap(core, in_axes=(0, 0, 0, None, None, 0)))


def fit_and_predict(args, n, fresh_c, bootstrap, xtest, seed):
    keys = jax.random.split(jax.random.key(seed), args.num_seeds * 3).reshape(
        3, args.num_seeds
    )
    cfg = model_config_dict(
        ENNConfig(bootstrap=bootstrap), max_data=n, minibatch_size=args.minibatch
    )
    model, (optimizer, metrics) = _make_batched_model(
        cfg, 1, 1, 1, None, keys[0], False
    )
    rngs = nnx.vmap(lambda k: nnx.Rngs(k))(keys[1])
    data = make_data(jax.random.key(1000 + n), n)
    make_train(fresh_c, cfg["UPDATE_STEPS"], args.minibatch)(
        model, optimizer, metrics, data, n, rngs
    )

    @nnx.vmap
    def predict(model, key):
        z = model.sample_index(key, args.num_z)
        x = model.normalize_input(xtest)
        return model.denormalize_delta_obs(model.batch_predict_samples(x, z))[..., 0]

    return np.asarray(predict(model, keys[2])), data  # (seeds, num_z, len(xtest))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ns", type=int, nargs="+", default=[10, 30, 100, 300, 1000])
    p.add_argument("--num-seeds", type=int, default=3)
    p.add_argument("--num-z", type=int, default=256)
    p.add_argument("--minibatch", type=int, default=100)
    p.add_argument("--out", default="notes/enn_bootstrap_demo.png")
    args = p.parse_args()

    xtest = jnp.linspace(-3.5, 3.5, 500).reshape(-1, 1)
    inside = np.abs(np.asarray(xtest[:, 0])) <= 3.0
    std = {}
    examples = {}
    for name, fresh, bootstrap, _ in ARMS:
        for n in args.ns:
            samples, data = fit_and_predict(args, n, fresh, bootstrap, xtest, seed=n)
            s = samples.std(axis=1)[:, inside].mean(axis=1)
            std[name, n] = s
            print(
                f"{name:24s} n={n:5d}  in-dist epistemic std {s.mean():.4f} ± {s.std():.4f}"
            )
            if n == 100:
                examples[name] = (samples[0], data)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2), constrained_layout=True)
    x = np.asarray(xtest[:, 0])
    for ax, (name, (samples, data)), (*_, color) in zip(axes, examples.items(), ARMS):
        mu, sd = samples.mean(0), samples.std(0)
        ax.scatter(
            data.obs[:, 0],
            data.info["next_obs"][:, 0] - data.obs[:, 0],
            s=10,
            color="#52514e",
            alpha=0.6,
            label="data (n=100)",
        )
        ax.plot(x, f(x), color="#0b0b0b", lw=1, ls="--", label="f(x)")
        ax.plot(x, mu, color=color, lw=2, label="mean over z")
        ax.fill_between(
            x,
            mu - 2 * sd,
            mu + 2 * sd,
            color=color,
            alpha=0.25,
            lw=0,
            label="± 2 epistemic std",
        )
        ax.set_title(f"{name}, n=100")
        ax.set_xlabel("x")
        ax.set_ylim(-3, 3)
        ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax = axes[3]
    ns = np.array(args.ns)
    for name, *_, color in ARMS:
        s = np.stack([std[name, n] for n in ns])
        ax.errorbar(
            ns,
            s.mean(1),
            yerr=s.std(1),
            color=color,
            lw=2,
            marker="o",
            ms=8,
            capsize=3,
            label=name,
        )
    ref = np.stack([std[ARMS[1][0], n] for n in ns]).mean(1)
    ax.plot(
        ns, ref[0] * np.sqrt(ns[0] / ns), color="#52514e", ls=":", lw=1, label="∝ 1/√n"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n training points")
    ax.set_ylabel("epistemic std, mean over |x| ≤ 3")
    ax.set_title(f"spread across z vs data ({args.num_seeds} seeds)")
    ax.legend(fontsize=8, frameon=False)
    for a in axes:
        a.grid(alpha=0.2)
        a.spines[["top", "right"]].set_visible(False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
