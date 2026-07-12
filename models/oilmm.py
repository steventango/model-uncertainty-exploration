"""OILMM multi-output GP world model using gpjax pathwise posterior sampling.

Architecture:
- Orthogonal Instantaneous Linear Mixing Model (OILMM): P outputs modelled as
  y = H·x + ε  where x are m independent latent GPs and H = U·sqrt(S) is a
  learned orthogonal mixing matrix.
- The orthogonality of U ensures the projected noise is diagonal, so inference
  decomposes into m independent single-output conjugate GP problems.
- Each latent GP is conditioned on ALL collected transitions; pathwise posterior
  samples are built directly (Wilson 2020) so all components are plain arrays
  stored in nnx.Variables.
- Cache Variables are allocated to MAX_DATA capacity; valid slots [:N] hold
  fitted values, remaining slots are zeroed (same static-shape trick as gp.py).
- Because N = pointer grows each rollout, _make_batched_train_model uses
  lru_cache to compile a fresh kernel per unique N.

Computational complexity: O(N³·m) — independent of output count P.
"""

import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import optax
from flax import nnx
import gpjax
import gpjax.models as gpjax_models
from gpjax.models.oilmm import oilmm_mll
from gpjax.gps import add_jitter
from gpjax.kernels.base import _val

from models.base import WorldModel, register_model
from models.gp import PathwiseSample


JITTER = 1e-2  # matches gp.py


# ---------------------------------------------------------------------------
# OILMMWorldModel
# ---------------------------------------------------------------------------


class OILMMWorldModel(WorldModel):
    """Multi-output GP dynamics model via OILMM with pathwise posterior sampling.

    Outputs y ∈ ℝ^P are modelled as linear mixtures of m latent GPs:
        y(x) = H · f(x) + ε,  H = U·diag(sqrt(S)) ∈ ℝ^{P×m}
    where U has orthonormal columns and f_1, …, f_m are independent GPs.

    Training:
      1. Fit mixing matrix U, S and per-latent kernel hyperparameters jointly
         by maximising the OILMM marginal log-likelihood (projected closed-form
         expression) via lax.scan + optax, which runs inside nnx.jit/vmap.
      2. Project targets to latent space: y_lat = T @ y  (T = S^{-½} U^T).
      3. Build a Wilson-2020 PathwiseSample for each latent GP independently.

    Prediction:
      predict_mean(x) → H @ lat_mean   ∈ ℝ^P
      predict_samples(x, ·) → lat_samples @ H^T  ∈ ℝ^{S×P}

    Cache Variables (static shapes, allocated to max_data):
      mixing [P, m], lengthscale [m, D], signal [m], proj_noise [m],
      X_cond [M, D], rff_freq [m, F, D], fourier_weights [m, S, 2F],
      canonical_weights [m, M, S], alpha [m, M].
    """

    def __init__(
        self,
        in_features: int,
        obs_dim: int,
        out_features: int,  # P: obs_dim [+ 1 reward + 1 terminated]
        num_latent: int,  # m: number of latent GPs (≤ P)
        max_data: int,
        num_features: int,  # F: RFF basis functions per latent
        num_samples: int,  # S: number of pathwise samples
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
        self.num_latent = num_latent
        self.max_data = max_data
        self.num_features = num_features
        self.num_samples = num_samples
        self.lr = lr

        P, m = out_features, num_latent
        F, M, S = num_features, max_data, num_samples

        # Learned mixing matrix H [P, m], written back after each training call
        self.mixing = nnx.Variable(jnp.zeros((P, m), dtype=jnp.float32))

        # Per-latent hyperparameters (extracted from OILMM after training)
        self.lengthscale = nnx.Variable(jnp.ones((m, in_features), dtype=jnp.float32))
        self.signal = nnx.Variable(jnp.ones((m,), dtype=jnp.float32))
        # Projected noise stddev per latent (sqrt of projected_noise_variance)
        self.proj_noise_std = nnx.Variable(jnp.full((m,), 0.1, dtype=jnp.float32))

        # Single shared conditioning input set (same X for all latents)
        self.X_cond = nnx.Variable(jnp.zeros((M, in_features), dtype=jnp.float32))

        # Per-latent PathwiseSample components
        self.rff_freq = nnx.Variable(jnp.zeros((m, F, in_features), dtype=jnp.float32))
        self.fourier_weights = nnx.Variable(jnp.zeros((m, S, 2 * F), dtype=jnp.float32))
        self.canonical_weights = nnx.Variable(jnp.zeros((m, M, S), dtype=jnp.float32))
        self.alpha = nnx.Variable(jnp.zeros((m, M), dtype=jnp.float32))

    # ------------------------------------------------------------------
    # WorldModel protocol
    # ------------------------------------------------------------------

    def sample_index(self, key, num_samples: int):
        return jax.random.randint(key, (num_samples,), 0, self.num_samples)

    def predict_mean(self, x):
        """Posterior mean. x: (in_features,) → (out_features,)."""
        x_2d = x[None, :]  # (1, D)
        lat_means = []
        for i in range(self.num_latent):
            kernel_i = gpjax.kernels.RBF(
                lengthscale=self.lengthscale.value[i],
                variance=self.signal.value[i],
            )
            cross_cov = kernel_i.cross_covariance(x_2d, self.X_cond.value)  # (1, M)
            lat_means.append((cross_cov @ self.alpha.value[i])[0])  # scalar
        lat_mean = jnp.stack(lat_means)  # (m,)
        return self.mixing.value @ lat_mean  # (P,)

    def predict_sample(self, x, index):
        return self.predict_samples(x, index)[index]

    def predict_samples(self, x, index):
        """Evaluate all S cached pathwise samples. x: (in_features,) → (S, out_features)."""
        F = self.num_features
        x_2d = x[None, :]  # (1, D)
        lat_samples = []
        for i in range(self.num_latent):
            kernel_i = gpjax.kernels.RBF(
                lengthscale=self.lengthscale.value[i],
                variance=self.signal.value[i],
            )
            sample_i = PathwiseSample(
                rff=gpjax.kernels.RFF(
                    base_kernel=kernel_i,
                    num_basis_fns=F,
                    frequencies=self.rff_freq.value[i],
                ),
                fourier_weights=self.fourier_weights.value[i],
                canonical_weights=self.canonical_weights.value[i],
                X_cond=self.X_cond.value,
            )
            lat_samples.append(sample_i(x_2d)[0])  # (S,)
        lat = jnp.stack(lat_samples, axis=-1)  # (S, m)
        return lat @ self.mixing.value.T  # (S, P)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _train_model(
    model: OILMMWorldModel,
    train_state,
    dataset,
    update_steps: int,
    N: int,  # concrete Python int — number of collected transitions
    minibatch_size: int,
    rngs: nnx.Rngs,
):
    """Fit OILMM to N transitions, then build and cache PathwiseSample components.

    N is a Python int captured via lru_cache so JAX traces static shapes for
    the latent Cholesky (N×N) and the lax.scan training loop.

    Steps:
      1. Normalise data and assemble X_cond (N,D), Y (N,P).
      2. Fit OILMM jointly (mixing matrix + per-latent kernels) via lax.scan.
      3. Extract H, per-latent ls/var/proj_noise; write to model Variables.
      4. Project Y to latent space: Y_lat = T @ Y^T  → (N,m).
      5. Per latent i: Cholesky → alpha_i; pathwise → canonical_weights_i.
    """
    F = model.num_features
    S = model.num_samples
    M = model.max_data
    m = model.num_latent
    P = model.out_features
    LR = model.lr

    # ------------------------------------------------------------------ #
    # Step 1 — normalise and assemble conditioning data                   #
    # ------------------------------------------------------------------ #
    model.update_stats(dataset, N)

    raw_obs = dataset.obs[:N]
    raw_action = dataset.action[:N]
    raw_next_obs = dataset.info["next_obs"][:N]
    raw_reward = dataset.reward[:N]
    raw_terminated = dataset.terminated[:N]

    X_cond = model.normalize_input(model.build_input(raw_obs, raw_action))  # (N, D)
    model.X_cond.value = model.X_cond.value.at[:N].set(X_cond)

    Y = model.build_targets(raw_obs, raw_next_obs, raw_reward, raw_terminated)
    # Y: (N, P)

    D = X_cond.shape[-1]  # static Python int under JAX tracing

    # ------------------------------------------------------------------ #
    # Step 2 — build and fit OILMM jointly                                #
    # ------------------------------------------------------------------ #
    # Pass a list of kernels to avoid copy.deepcopy under JAX tracing.
    key_oilmm = rngs()
    kernels = [gpjax.kernels.RBF(n_dims=D) for _ in range(m)]
    oilmm = gpjax_models.OILMMModel(
        num_outputs=P,
        num_latent_gps=m,
        key=key_oilmm,
        kernel=kernels,
    )
    train_data = gpjax.Dataset(X=X_cond, y=Y)

    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(LR))
    opt_state = tx.init(eqx.filter(oilmm, eqx.is_array))

    def body(carry, _):
        mdl, state = carry
        loss, grads = eqx.filter_value_and_grad(lambda m2: -oilmm_mll(m2, train_data))(
            mdl
        )
        updates, new_state = tx.update(grads, state, eqx.filter(mdl, eqx.is_array))
        new_mdl = eqx.apply_updates(mdl, updates)
        return (new_mdl, new_state), loss

    (oilmm_opt, _), loss_hist = jax.lax.scan(
        body, (oilmm, opt_state), None, length=update_steps
    )

    # ------------------------------------------------------------------ #
    # Step 3 — extract learned mixing matrix and per-latent hyperparams   #
    # ------------------------------------------------------------------ #
    H = oilmm_opt.mixing_matrix.H  # (P, m) — property handles _val internally
    T = oilmm_opt.mixing_matrix.T  # (m, P)
    proj_noise = oilmm_opt.mixing_matrix.projected_noise_variance  # (m,) variances

    model.mixing.value = H

    # ------------------------------------------------------------------ #
    # Step 4 — project targets to latent space                            #
    # ------------------------------------------------------------------ #
    Y_lat = (T @ Y.T).T  # (N, m)

    # ------------------------------------------------------------------ #
    # Step 5 — per-latent pathwise cache (mirrors gp.py loop over P)     #
    # ------------------------------------------------------------------ #
    for i in range(m):
        ls_i = jnp.clip(_val(oilmm_opt.latent_priors[i].kernel.lengthscale), 1e-2, 20.0)
        sig_i = jnp.clip(_val(oilmm_opt.latent_priors[i].kernel.variance), 1e-2, 20.0)
        # proj_noise[i] is a variance; take sqrt and clamp to get stddev
        noise_i = jnp.clip(jnp.sqrt(jnp.maximum(proj_noise[i], 1e-8)), 0.01, 20.0)

        model.lengthscale.value = model.lengthscale.value.at[i].set(ls_i)
        model.signal.value = model.signal.value.at[i].set(sig_i)
        model.proj_noise_std.value = model.proj_noise_std.value.at[i].set(noise_i)

        # Rebuild kernel with clamped params for consistent cache construction
        kernel_i = gpjax.kernels.RBF(lengthscale=ls_i, variance=sig_i, n_dims=D)

        # Shared Cholesky for alpha (mean) and canonical_weights (samples)
        Kxx = kernel_i.gram(X_cond).as_matrix()  # (N, N)
        Sigma = add_jitter(Kxx, noise_i**2 + JITTER)
        L = jnp.linalg.cholesky(Sigma)
        Y_i = Y_lat[:, i]  # (N,)
        alpha_i = jsp.linalg.cho_solve((L, True), Y_i)  # (N,)

        # Wilson (2020) pathwise conditioning for latent i
        key_rff, key_fw, key_eps = jax.random.split(rngs(), 3)
        rff_i = gpjax.kernels.RFF(base_kernel=kernel_i, num_basis_fns=F, key=key_rff)
        fourier_weights_i = jax.random.normal(key_fw, (S, 2 * F))
        Phi = rff_i.compute_features(X_cond) * jnp.sqrt(sig_i / F)  # (N, 2F)
        eps_i = noise_i * jax.random.normal(key_eps, (N, S))
        rhs = Y_i[:, None] + eps_i - Phi @ fourier_weights_i.T  # (N, S)
        canonical_weights_i = jsp.linalg.cho_solve((L, True), rhs)  # (N, S)

        # Zero-pad to max_data so Variable shapes stay static across rollouts
        cw_full = jnp.zeros((M, S), dtype=jnp.float32).at[:N].set(canonical_weights_i)
        alpha_full = jnp.zeros((M,), dtype=jnp.float32).at[:N].set(alpha_i)
        model.canonical_weights.value = model.canonical_weights.value.at[i].set(cw_full)
        model.alpha.value = model.alpha.value.at[i].set(alpha_full)
        model.fourier_weights.value = model.fourier_weights.value.at[i].set(
            fourier_weights_i
        )
        model.rff_freq.value = model.rff_freq.value.at[i].set(rff_i.frequencies)

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
    num_latent = model_config.get("NUM_LATENT") or out_features

    @nnx.vmap
    def build(key):
        return OILMMWorldModel(
            in_features=in_features,
            obs_dim=obs_dim,
            out_features=out_features,
            num_latent=num_latent,
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


register_model("oilmm")(
    {
        "make_batched_model": _make_batched_model,
        "make_batched_train_model": _make_batched_train_model,
    }
)
