"""Gaussian Process world model using gpjax pathwise posterior sampling.

Architecture:
- Per-output independent GP with ARD-RBF kernel (zero mean, Gaussian likelihood).
- Conditioned on ALL collected transitions (no inducing-point subset).
- Training: gpjax.fit optimises hyperparameters; PathwiseSample is then built
  directly (Wilson 2020) without calling sample_approx, so all components are
  plain arrays ready for split/merge storage in nnx.Variables.
- Cache Variables are allocated to MAX_DATA capacity; valid slots [:N] hold
  fitted values, remaining slots are zeroed.
- Because N = pointer grows each rollout, _make_batched_train_model uses
  lru_cache to compile a fresh kernel per unique N.
"""

import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import optax
from flax import nnx
import gpjax
from gpjax.gps import ConjugatePosterior, add_jitter
from gpjax.kernels.base import _val
import paramax

from models.base import WorldModel, register_model


# ---------------------------------------------------------------------------
# PathwiseSample
# ---------------------------------------------------------------------------


class PathwiseSample(eqx.Module):
    """Wilson et al. (2020) pathwise posterior sample as a callable equinox pytree.

    Holds the RFF kernel, Fourier weights, canonical weights, and conditioning
    inputs.  Evaluation is O(M) per query — no Cholesky at prediction time.
    Being an equinox Module makes the whole object tree-flattenable for
    split/merge storage: array leaves go into nnx.Variables; the static treedef
    (BasisFunctionComputation, num_basis_fns, kernel class) is shared across seeds.
    """

    rff: gpjax.kernels.RFF
    fourier_weights: jax.Array  # (S, 2F)
    canonical_weights: jax.Array  # (M, S)
    X_cond: jax.Array  # (M, D)

    def __call__(self, x: jax.Array) -> jax.Array:
        """x: (N, D) → (N, S)."""
        F = self.rff.num_basis_fns
        scale = jnp.sqrt(_val(self.rff.base_kernel.variance) / F)
        phi = self.rff.compute_features(x) * scale  # (N, 2F)
        weight_space = jnp.inner(phi, self.fourier_weights)  # (N, S)
        cross = self.rff.base_kernel.cross_covariance(x, self.X_cond)  # (N, M)
        return weight_space + jnp.matmul(cross, self.canonical_weights)


# ---------------------------------------------------------------------------
# GPModel
# ---------------------------------------------------------------------------


class GPModel(WorldModel):
    """GP-based dynamics model with pathwise posterior sampling (Wilson et al., 2020).

    Cache Variables (X_cond, rff_freq, fourier_weights, canonical_weights, alpha) are
    sized to max_data.  After each training call with N = pointer data points, slots
    [:N] hold the fitted values; slots [N:] are zeroed and contribute nothing to
    prediction because canonical_weights[N:] * k(x, X_cond[N:]) adds zero.
    """

    def __init__(
        self,
        in_features: int,
        obs_dim: int,
        out_features: int,
        max_data: int,
        num_features: int,
        num_samples: int,
        lr: float,
        act_dim=None,
        eps: float = 1e-8,
        predict_reward_terminated: bool = True,
    ):
        super().__init__(
            in_features=in_features,
            obs_dim=obs_dim,
            act_dim=act_dim,
            eps=eps,
            predict_reward_terminated=predict_reward_terminated,
        )
        self.out_features = out_features
        self.max_data = max_data
        self.num_features = num_features
        self.num_samples = num_samples
        self.lr = lr

        self.lengthscale = nnx.Variable(
            jnp.ones((out_features, in_features), dtype=jnp.float32)
        )
        self.signal = nnx.Variable(jnp.ones((out_features,), dtype=jnp.float32))
        self.noise = nnx.Variable(jnp.full((out_features,), 0.1, dtype=jnp.float32))

        F, M, S = num_features, max_data, num_samples
        self.X_cond = nnx.Variable(jnp.zeros((M, in_features), dtype=jnp.float32))
        self.rff_freq = nnx.Variable(
            jnp.zeros((out_features, F, in_features), dtype=jnp.float32)
        )
        self.fourier_weights = nnx.Variable(
            jnp.zeros((out_features, S, 2 * F), dtype=jnp.float32)
        )
        self.canonical_weights = nnx.Variable(
            jnp.zeros((out_features, M, S), dtype=jnp.float32)
        )
        self.alpha = nnx.Variable(jnp.zeros((out_features, M), dtype=jnp.float32))

    # ------------------------------------------------------------------
    # WorldModel protocol
    # ------------------------------------------------------------------

    def sample_index(self, key, num_samples: int):
        return jax.random.randint(key, (num_samples,), 0, self.num_samples)

    def predict_mean(self, x):
        """Posterior mean. x: (in_features,) → (out_features,)."""
        x_2d = x[None, :]  # (1, D)
        means = []
        for j in range(self.out_features):
            kernel_j = gpjax.kernels.RBF(
                lengthscale=self.lengthscale.value[j], variance=self.signal.value[j]
            )
            cross_cov = kernel_j.cross_covariance(x_2d, self.X_cond.value)  # (1, M)
            means.append((cross_cov @ self.alpha.value[j])[0])
        return jnp.stack(means)

    def predict_sample(self, x, index):
        return self.predict_samples(x, index)[index]

    def predict_samples(self, x, index):
        """Evaluate all S cached pathwise samples at x. x: (in_features,) → (S, out_features)."""
        F = self.num_features
        x_2d = x[None, :]  # (1, D)
        samples = []
        for j in range(self.out_features):
            kernel_j = gpjax.kernels.RBF(
                lengthscale=self.lengthscale.value[j], variance=self.signal.value[j]
            )
            sample_j = PathwiseSample(
                rff=gpjax.kernels.RFF(
                    base_kernel=kernel_j,
                    num_basis_fns=F,
                    frequencies=self.rff_freq.value[j],
                ),
                fourier_weights=self.fourier_weights.value[j],
                canonical_weights=self.canonical_weights.value[j],
                X_cond=self.X_cond.value,
            )
            samples.append(sample_j(x_2d)[0])  # (S,)
        return jnp.stack(samples, axis=-1)  # (S, out_features)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _train_model(
    model: GPModel,
    train_state,
    dataset,
    update_steps: int,
    N: int,  # concrete Python int — number of collected transitions
    minibatch_size: int,
    rngs: nnx.Rngs,
):
    """Fit GP to N transitions, then build and cache PathwiseSample components.

    N is a Python int captured via lru_cache so JAX traces static shapes for
    the Cholesky (N×N) and the gpjax scan inside gpjax.fit.
    """
    F = model.num_features
    S = model.num_samples
    M = model.max_data
    LR = model.lr
    JITTER = 1e-2  # absolute jitter, matches gpjax's add_jitter usage in sample_approx

    model.update_stats(dataset, N)

    # All N collected transitions as conditioning points
    raw_obs = dataset.obs[:N]
    raw_action = dataset.action[:N]
    raw_next_obs = dataset.info["next_obs"][:N]
    raw_reward = dataset.reward[:N]
    raw_terminated = dataset.terminated[:N]

    X_cond = model.normalize_input(model.build_input(raw_obs, raw_action))  # (N, D)
    model.X_cond.value = model.X_cond.value.at[:N].set(X_cond)

    Y = model.build_targets(raw_obs, raw_next_obs, raw_reward, raw_terminated)

    nmll = lambda m, d: -gpjax.objectives.conjugate_mll(m, d)
    all_loss = []

    for j in range(model.out_features):
        Y_j = Y[:, j : j + 1]  # (N, 1) — gpjax expects 2D targets

        ls_j = model.lengthscale.value[j]
        sig_j = model.signal.value[j]
        noise_j = model.noise.value[j]

        kernel_j = gpjax.kernels.RBF(
            lengthscale=ls_j, variance=sig_j, n_dims=X_cond.shape[-1]
        )
        likelihood_j = gpjax.likelihoods.Gaussian(num_datapoints=N, obs_stddev=noise_j)
        prior_j = gpjax.gps.Prior(
            mean_function=gpjax.mean_functions.Zero(), kernel=kernel_j
        )
        posterior_j = ConjugatePosterior(
            prior=prior_j, likelihood=likelihood_j, jitter=JITTER
        )
        train_data_j = gpjax.Dataset(X=X_cond, y=Y_j)

        key_fit = rngs()
        tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(LR))
        opt_posterior_j, loss_hist_j = gpjax.fit(
            model=posterior_j,
            objective=nmll,
            train_data=train_data_j,
            optim=tx,
            num_iters=update_steps,
            key=key_fit,
            verbose=False,
            safe=False,
        )
        all_loss.append(loss_hist_j)

        # Write back optimised hyperparameters, clamped to sane ranges
        unwrapped = paramax.unwrap(opt_posterior_j)
        ls_opt = jnp.clip(unwrapped.prior.kernel.lengthscale, 1e-2, 20.0)
        sig_opt = jnp.clip(unwrapped.prior.kernel.variance, 1e-2, 20.0)
        noise_opt = jnp.clip(unwrapped.likelihood.obs_stddev, 0.01, 20.0)
        model.lengthscale.value = model.lengthscale.value.at[j].set(ls_opt)
        model.signal.value = model.signal.value.at[j].set(sig_opt)
        model.noise.value = model.noise.value.at[j].set(noise_opt)

        # Build posterior with clipped params for consistent cache construction
        kernel_c = gpjax.kernels.RBF(
            lengthscale=ls_opt, variance=sig_opt, n_dims=X_cond.shape[-1]
        )
        likelihood_c = gpjax.likelihoods.Gaussian(
            num_datapoints=N, obs_stddev=noise_opt
        )
        prior_c = gpjax.gps.Prior(
            mean_function=gpjax.mean_functions.Zero(), kernel=kernel_c
        )
        posterior_c = ConjugatePosterior(
            prior=prior_c, likelihood=likelihood_c, jitter=JITTER
        )

        # Shared Cholesky for both alpha (predict_mean) and canonical_weights (samples)
        Kxx = posterior_c.prior.kernel.gram(X_cond).as_matrix()
        Sigma = add_jitter(Kxx, noise_opt**2 + JITTER)
        L = jnp.linalg.cholesky(Sigma)
        alpha_j = jsp.linalg.cho_solve((L, True), Y_j[:, 0])  # (N,)

        # Wilson (2020) pathwise conditioning — build PathwiseSample directly
        key_rff, key_fw, key_eps = jax.random.split(rngs(), 3)
        rff_j = gpjax.kernels.RFF(base_kernel=kernel_c, num_basis_fns=F, key=key_rff)
        fourier_weights_j = jax.random.normal(key_fw, (S, 2 * F))
        Phi = rff_j.compute_features(X_cond) * jnp.sqrt(sig_opt / F)  # (N, 2F)
        eps_j = noise_opt * jax.random.normal(key_eps, (N, S))
        rhs = Y_j + eps_j - Phi @ fourier_weights_j.T  # (N, S)
        canonical_weights_j = jsp.linalg.cho_solve((L, True), rhs)  # (N, S)

        # Zero-pad to max_data so Variable shapes stay static across rollouts
        cw_full = jnp.zeros((M, S), dtype=jnp.float32).at[:N].set(canonical_weights_j)
        alpha_full = jnp.zeros((M,), dtype=jnp.float32).at[:N].set(alpha_j)
        model.canonical_weights.value = model.canonical_weights.value.at[j].set(cw_full)
        model.alpha.value = model.alpha.value.at[j].set(alpha_full)
        model.fourier_weights.value = model.fourier_weights.value.at[j].set(
            fourier_weights_j
        )
        model.rff_freq.value = model.rff_freq.value.at[j].set(rff_j.frequencies)

    loss_hist = jnp.stack(all_loss, axis=0).mean(axis=0)
    return {"loss": loss_hist}


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def _make_batched_model(
    model_config,
    in_features,
    obs_dim,
    out_features,
    act_dim,
    keys,
    predict_reward_terminated: bool = True,
):
    @nnx.vmap
    def build(key):
        return GPModel(
            in_features=in_features,
            obs_dim=obs_dim,
            out_features=out_features,
            max_data=model_config["MAX_DATA"],
            num_features=model_config["NUM_FEATURES"],
            num_samples=model_config["NUM_SAMPLES"],
            lr=model_config["LR"],
            act_dim=act_dim,
            predict_reward_terminated=predict_reward_terminated,
        )

    return build(keys), None


def _make_batched_train_model(update_steps, minibatch_size):
    @functools.lru_cache(maxsize=None)
    def _compiled(n):
        def core(model, train_state, dataset, rngs):
            return _train_model(
                model, train_state, dataset, update_steps, n, minibatch_size, rngs
            )

        return nnx.jit(nnx.vmap(core, in_axes=(0, None, 0, 0), out_axes=0))

    def train_fn(model, train_state, dataset, pointer, rngs):
        return _compiled(int(pointer))(model, train_state, dataset, rngs)

    return train_fn


register_model("gp")(
    {
        "make_batched_model": _make_batched_model,
        "make_batched_train_model": _make_batched_train_model,
    }
)
