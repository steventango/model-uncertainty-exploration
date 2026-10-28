"""Epinet heads on the regression.ipynb curve.

Targets are standardized to unit variance, prior scale 1, plus
(lambda / N) ||theta||^2 on the trainable non-prior weights. Adam, not AdamW.

Two heads:
  beta_nll: a mean and a softplus scale, trained with the beta-NLL (Seitzer et
    al. 2022) of the unperturbed label; EIG uses the learned scale.
  mse: deepmind/enn L2Loss + GaussianTargetNoise. A mean only, trained with MSE
    on y_i + sigma_n (b_i . z), where b_i is a unit vector fixed per data point;
    EIG uses the fixed noise sigma_n = NOISE / y_std.

Plain NLL never fits the mean here: the scale inflates first and the 1/sigma^2
weight on the mean gradient leaves loc stuck near a constant. Weighting each
point's NLL by stop_grad(sigma^(2 beta)) with beta = 1 gives loc an MSE gradient.
The beta_nll label is not perturbed: its scale head already models the noise, so a unit
perturbation only inflates sigma (diluting the EIG), and perturbing by the
predicted scale has no finite fixed point (sigma^2 = noise^2 + sigma^2).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from networks import ENN

N = 100
SUBSETS = (1, 3, 10, 30, 100)
NOISE = 0.3
LR = 1e-3
MIN_SCALE = 1e-3
BETA = 1.0
LAMBDAS = (0.0, 1.0, 1e-1, 1e-2, 1e-3, 1e-4)
INDEX_DIM = 8
HIDDEN = 32
LEARNABLE_HIDDEN = 32
PRIOR_HIDDEN = 16
REFERENCE_PRIOR_HIDDEN = (5, 5)
CONFIGS = (("ours", "beta_nll"), ("ours", "mse"), ("reference", "mse"))
TOTAL_STEPS = 10_000
N_EPISTEMIC = 128
OUT_STEM = Path(__file__).with_suffix("")

TRAINABLE = nnx.All(nnx.Param, nnx.Not(nnx.PathContains("prior")))


def make_data(key):
    key, noise_key = jax.random.split(key)
    x = jax.random.uniform(key, (N, 1), minval=-3.0, maxval=3.0)

    def latent(t):
        return jnp.sin(4 * t) + jnp.cos(2 * t)

    y = latent(x) + NOISE * jax.random.normal(noise_key, (N, 1))
    xtest = jnp.linspace(-6.0, 6.0, 400).reshape(-1, 1)
    ytest = latent(xtest)
    return x, y, xtest, ytest, latent


class ReferencePrior(nnx.Module):
    """deepmind/enn make_ensemble_prior: index_dim ReLU MLPs combined by z.

    EnsembleMLP init: truncated normal weights with std 1/sqrt(fan_in), the same
    for the first-layer bias, zeros for the later biases.
    """

    def __init__(self, in_features, out_features, index_dim, *, rngs):
        sizes = (in_features, *REFERENCE_PRIOR_HIDDEN, out_features)
        self.num_layers = len(sizes) - 1
        for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            init = nnx.initializers.truncated_normal(stddev=fan_in**-0.5)
            w = init(rngs.params(), (index_dim, fan_in, fan_out))
            if i == 0:
                b = init(rngs.params(), (index_dim, fan_out))
            else:
                b = jnp.zeros((index_dim, fan_out))
            setattr(self, f"w{i}", nnx.Param(w))
            setattr(self, f"b{i}", nnx.Param(b))

    def __call__(self, x, z, rngs=None):
        h = jnp.broadcast_to(x, (z.shape[0], x.shape[-1]))
        for i in range(self.num_layers):
            h = jnp.einsum("na,nab->nb", h, getattr(self, f"w{i}")[...])
            h = h + getattr(self, f"b{i}")[...]
            if i < self.num_layers - 1:
                h = jax.nn.relu(h)
        return jnp.einsum("no,n->o", h, z)


def make_enn(key, prior, out_features):
    key, prior_key = jax.random.split(key)
    enn = ENN(
        in_features=1,
        hidden_features=HIDDEN,
        learnable_hidden_features=LEARNABLE_HIDDEN,
        prior_hidden_features=PRIOR_HIDDEN,
        out_features=out_features,
        index_dim=INDEX_DIM,
        rngs=nnx.Rngs(params=key),
    )
    if prior == "reference":
        enn.epinet.prior = ReferencePrior(
            1, out_features, INDEX_DIM, rngs=nnx.Rngs(params=prior_key)
        )
    return enn


def param_sq(model):
    leaves = jax.tree.leaves(nnx.state(model, TRAINABLE))
    return sum(jnp.sum(p**2) for p in leaves)


def scale_of(raw):
    return jax.nn.softplus(raw) + MIN_SCALE


def unit_sphere(key, n):
    b = jax.random.normal(key, (n, INDEX_DIM))
    return b / jnp.linalg.norm(b, axis=-1, keepdims=True)


def train_block(model, optimizer, x, y, b, rng, l2_reg, n_steps, head, noise_std):
    def step(carry, _):
        model, optimizer, rng = carry
        rng, z_key = jax.random.split(rng)
        z = jax.random.normal(z_key, (INDEX_DIM,))
        target = jnp.reshape(y, (-1,))

        def loss_fn(model):
            pred = jax.vmap(lambda xi: model(xi, z)[1])(x)
            if head == "mse":
                perturbed = target + noise_std * (b @ z)
                loss = (pred[:, 0] - perturbed) ** 2
            else:
                loc, scale = pred[:, 0], scale_of(pred[:, 1])
                loss = 0.5 * jnp.log(2 * jnp.pi) + jnp.log(scale)
                loss = loss + 0.5 * ((target - loc) / scale) ** 2
                loss = jax.lax.stop_gradient(scale ** (2 * BETA)) * loss
            return loss.mean() + l2_reg * param_sq(model) / x.shape[0]

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return (model, optimizer, rng), loss

    (model, optimizer, rng), losses = nnx.scan(step, length=n_steps)(
        (model, optimizer, rng), None
    )
    return model, optimizer, rng, losses.mean()


def predict(model, xtest, key, head, noise_std):
    z = jax.random.normal(key, (N_EPISTEMIC, INDEX_DIM))

    def samples_for(zi):
        return jax.vmap(lambda xi: model(xi, zi)[1])(xtest)

    raw = jax.vmap(samples_for)(z)
    loc = raw[..., 0]
    if head == "mse":
        scale = jnp.full_like(loc, noise_std)
    else:
        scale = scale_of(raw[..., 1])
    eig = 0.5 * jnp.log1p(jnp.var(loc, axis=0) / jnp.mean(scale**2, axis=0))
    return jnp.mean(loc, axis=0), jnp.std(loc, axis=0), jnp.mean(scale, axis=0), eig


def weighted_mean(values, mask):
    weight = mask.astype(values.dtype) / jnp.sum(mask)
    return jnp.sum(jnp.squeeze(values) * weight)


def full_stats(loc, std, scale, eig, xtest, ytest, y_mean, y_std):
    loc_y = loc * y_std + y_mean
    in_support = jnp.abs(xtest[:, 0]) <= 3.0
    err = (loc_y - jnp.squeeze(ytest)) ** 2
    return {
        "loc_y": loc_y,
        "std_y": std * y_std,
        "rmse": jnp.sqrt(weighted_mean(err, in_support)),
        "sigma": weighted_mean(scale, in_support) * y_std,
        "eig_in": weighted_mean(eig, in_support),
        "eig": eig,
        "eig_out": weighted_mean(eig, ~in_support),
    }


def run_sweep(template, x, y, b, xtest, checkpoints, report, head, noise_std):
    runs = []
    for lam in LAMBDAS:
        model = nnx.clone(template)
        runs.append(
            {
                "lambda": lam,
                "model": model,
                "optimizer": nnx.Optimizer(model, optax.adam(LR), wrt=TRAINABLE),
                "rng": jax.random.key(0),
                "l2_reg": jnp.asarray(lam),
                "snapshots": {},
            }
        )
    train_jit = nnx.jit(train_block, static_argnames=("n_steps", "head"))
    steps = 0
    for target in checkpoints:
        block = target - steps
        for run in runs:
            model, optimizer, rng, mean_loss = train_jit(
                run["model"],
                run["optimizer"],
                x,
                y,
                b,
                run["rng"],
                run["l2_reg"],
                block,
                head,
                noise_std,
            )
            run["model"] = model
            run["optimizer"] = optimizer
            run["rng"] = rng
            run["last_loss"] = mean_loss
        steps = target
        print(f"step {steps}")
        for run in runs:
            loc, std, scale, eig = predict(
                run["model"], xtest, jax.random.key(1), head, noise_std
            )
            stats = report(loc, std, scale, eig)
            stats["loss"] = run["last_loss"]
            stats["sq"] = param_sq(run["model"])
            run["snapshots"][steps] = stats
            label = f"{run['lambda']:.0e}"
            print(
                f"  λ={label:7s}  loss {float(stats['loss']):.3f}  "
                f"rmse {float(stats['rmse']):.3f}  "
                f"σ {float(stats['sigma']):.3f}  "
                f"eig in/out {float(stats['eig_in']):.3f}/"
                f"{float(stats['eig_out']):.3f}  "
                f"||θ||^2 {float(stats['sq']):.1f}"
            )
    return runs


def main():
    print("devices", jax.devices())
    key = jax.random.key(123)
    key, data_key, init_key, b_key = jax.random.split(key, 4)
    x, y, xtest, ytest, latent = make_data(data_key)
    y_mean, y_std = jnp.mean(y), jnp.std(y)
    y_n = (y - y_mean) / y_std
    # One b_i per data point, shared by every subset that contains it.
    b = unit_sphere(b_key, N)
    noise_std = NOISE / y_std
    for prior, head in CONFIGS:
        plot_config(
            prior, head, init_key, x, y_n, b, noise_std, xtest, ytest, y_mean, y_std
        )


def plot_config(
    prior, head, init_key, x, y_n, b, noise_std, xtest, ytest, y_mean, y_std
):
    template = make_enn(init_key, prior, 1 if head == "mse" else 2)
    _, _, _, prior_eig = predict(template, xtest, jax.random.key(1), head, noise_std)

    sweeps = {}
    for n in SUBSETS:
        print(f"{prior} prior, {head}, {n} points")
        sweeps[n] = run_sweep(
            template,
            x[:n],
            y_n[:n],
            b[:n],
            xtest,
            (1_000, TOTAL_STEPS),
            lambda loc, std, scale, eig: full_stats(
                loc, std, scale, eig, xtest, ytest, y_mean, y_std
            ),
            head,
            noise_std,
        )

    grid = jnp.squeeze(xtest)
    fig, axes = plt.subplots(
        len(SUBSETS),
        len(LAMBDAS),
        figsize=(16, 2.4 * len(SUBSETS)),
        sharex=True,
        sharey="row",
    )
    for row, n in enumerate(SUBSETS):
        for col, run in enumerate(sweeps[n]):
            done = run["snapshots"][TOTAL_STEPS]
            ax = axes[row, col]
            ax.plot(grid, prior_eig, color="0.5", linestyle="--", linewidth=1)
            ax.plot(grid, done["eig"], color="C0", linewidth=1.5)
            ax.plot(x[:n, 0], jnp.zeros(n), "|", color="C3", markersize=8)
            ax.set_title(
                f"λ={run['lambda']:.0e}  eig in/out "
                f"{float(done['eig_in']):.2f}/{float(done['eig_out']):.2f}",
                fontsize=8,
            )
        axes[row, 0].set_ylabel(f"{n} points, eig")
    fig.suptitle(f"{prior} prior, {head}")
    fig.tight_layout()
    out_path = OUT_STEM.with_name(f"{OUT_STEM.name}_{prior}_{head}.png")
    fig.savefig(out_path, dpi=140)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
