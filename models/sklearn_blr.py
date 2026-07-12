"""sklearn-trained / JAX-inference Bayesian linear regression with fixed RFF features.

Training is performed on the host (Python loop over seeds × outputs) using
sklearn's BayesianRidge or ARDRegression — both learn noise precision α and
weight precision λ (global or per-feature) via EM, unlike the JAX BLR whose
hyperparameters are fixed by config.

Inference (predict_mean, predict_sample, sample_index) is pure JAX and identical
in structure to BLRModel, enabling vmap'd rollouts.

Architecture:
  _prep   — nnx.jit(nnx.vmap): update normalisation stats, compute Phi and Y
             over all seeds in one jit'd call; shape-stable via masking.
  sklearn — host Python loop: fit one estimator per (seed, output dimension).
  _write  — nnx.jit(nnx.vmap): Cholesky-sample posterior weights and assign into
             model Variables.

See models/features.py for the RFF uncertainty caveat (degenerate epistemic
variance under extrapolation is a known limitation of fixed feature maps).
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from sklearn.linear_model import ARDRegression, BayesianRidge

from models.base import WorldModel, register_model
from models.features import RFFFeatures

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SklearnBLRModel(WorldModel):
    """BLR with fixed RFF features trained by sklearn; pure-JAX inference."""

    def __init__(
        self,
        in_features: int,
        obs_dim: int,
        out_features: int,
        num_features: int,
        num_samples: int,
        length_scale: float,
        key: jax.Array,
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
        self.num_features = num_features
        self.num_samples = num_samples
        self.features = RFFFeatures(key, in_features, num_features, length_scale)

        o = out_features
        F = num_features + 1
        S = num_samples

        self.w_mean = nnx.Variable(jnp.zeros((o, F)))
        self.w_samples = nnx.Variable(jnp.zeros((S, o, F)))
        self.noise = nnx.Variable(jnp.ones((o,)))

    def sample_index(self, key: jax.Array, num_samples: int) -> jax.Array:
        return jax.random.randint(key, (num_samples,), 0, self.num_samples)

    def predict_mean(self, x: jax.Array) -> jax.Array:
        """x: (in_features,) → (out_features,)."""
        return self.w_mean.value @ self.features(x)

    def predict_sample(self, x: jax.Array, index: jax.Array) -> jax.Array:
        return self.w_samples.value[index] @ self.features(x)

    # No variance override: base class computes empirical variance over
    # w_samples, which correctly reflects the full posterior spread for both
    # BayesianRidge (global λ) and ARDRegression (per-feature λ).


# ---------------------------------------------------------------------------
# JAX prep / write helpers (module-level so jit cache is shared across calls)
# ---------------------------------------------------------------------------


def _prep_single(model: SklearnBLRModel, dataset, pointer):
    """Update normalisation stats and compute masked Phi, Y for one seed."""
    model.update_stats(dataset, pointer)
    N = dataset.obs.shape[0]
    mask = (jnp.arange(N) < pointer).astype(jnp.float32)[:, None]
    X = model.normalize_input(model.build_input(dataset.obs, dataset.action))
    Phi = model.features(X) * mask  # (N, F)
    delta_obs = dataset.info["next_obs"] - dataset.obs
    delta_obs_norm = model.normalize_delta_obs(delta_obs)
    if model.predict_reward_terminated:
        reward_norm = model.normalize_reward(dataset.reward)[:, None]
        terminated = dataset.terminated[:, None].astype(jnp.float32)
        Y = jnp.concatenate([delta_obs_norm, reward_norm, terminated], axis=-1)
    else:
        Y = delta_obs_norm
    return Phi, Y * mask  # (N, F), (N, o)


_prep = nnx.jit(nnx.vmap(_prep_single, in_axes=(0, 0, None), out_axes=0))


def _write_single(
    model: SklearnBLRModel,
    coef: jax.Array,  # (o, F)
    cov: jax.Array,  # (o, F, F)
    noise: jax.Array,  # (o,)
    rngs: nnx.Rngs,
) -> None:
    """Cholesky-sample posterior weights and write into model Variables."""
    S = model.num_samples
    o = model.out_features
    F = model.num_features + 1

    # Use eigendecomposition instead of Cholesky: robust to near-zero eigenvalues
    # from ARD-pruned features (Cholesky would silently return NaN for non-PD input).
    eigenvalues, eigenvectors = jax.vmap(jnp.linalg.eigh)(cov)  # (o,F), (o,F,F)
    eigenvalues = jnp.maximum(eigenvalues, 1e-8)
    L = eigenvectors * jnp.sqrt(eigenvalues)[:, None, :]  # (o, F, F)
    z = jax.random.normal(rngs(), (S, o, F))
    # w[s, i] = coef[i] + L[i] @ z[s, i]
    w_samples = coef[None] + jnp.einsum("oij,soj->soi", L, z)  # (S, o, F)

    model.w_mean.value = coef
    model.w_samples.value = w_samples
    model.noise.value = noise


_write = nnx.jit(nnx.vmap(_write_single, in_axes=(0, 0, 0, 0, 0), out_axes=0))


# ---------------------------------------------------------------------------
# Factory
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
        return SklearnBLRModel(
            in_features=in_features,
            obs_dim=obs_dim,
            out_features=out_features,
            num_features=model_config["NUM_FEATURES"],
            num_samples=model_config["NUM_SAMPLES"],
            length_scale=model_config["LENGTH_SCALE"],
            key=key,
            act_dim=act_dim,
            predict_reward_terminated=predict_reward_terminated,
        )

    return build(keys), None


def _extract_sigma_full(est, _F: int) -> np.ndarray:
    """BayesianRidge: sigma_ is already (F, F)."""
    sigma = est.sigma_.astype(np.float32)
    return 0.5 * (sigma + sigma.T)


def _extract_sigma_ard(est, F: int) -> np.ndarray:
    """ARDRegression: sigma_ is (n_kept, n_kept); scatter into full (F, F).

    ARD prunes features whose precision λ_i exceeds threshold_lambda (default
    1e4). The posterior covariance is only computed for kept features. Pruned
    features have effectively zero posterior variance — their rows/cols in the
    full (F, F) matrix stay at zero, which the downstream Cholesky handles
    correctly (those weight samples will stay near zero as well).
    """
    keep = est.lambda_ < est.threshold_lambda  # (F,) bool
    full_sigma = np.zeros((F, F), dtype=np.float32)
    full_sigma[np.ix_(keep, keep)] = est.sigma_.astype(np.float32)
    return 0.5 * (full_sigma + full_sigma.T)


def _make_batched_train_model(estimator_cls, estimator_kwargs, extract_sigma_fn):
    """Return a train_fn that fits sklearn on host and writes params via JAX."""

    def train_fn(model, train_state, dataset, pointer, rngs):
        n = int(pointer)

        # --- JAX prep: normalise data, compute masked features (jit+vmap over seeds)
        Phi_jax, Y_jax = _prep(model, dataset, pointer)  # (B, N, F), (B, N, o)

        # Slice to valid rows on host; asarray blocks until JAX computation is done.
        # nan_to_num guards against IEEE NaN * 0 = NaN: any NaN in uninitialized
        # dataset rows survives the _prep_single mask and would crash sklearn.
        Phi_np = np.nan_to_num(np.asarray(Phi_jax)[:, :n, :], nan=0.0)  # (B, n, F)
        Y_np = np.nan_to_num(np.asarray(Y_jax)[:, :n, :], nan=0.0)  # (B, n, o)
        B, _, F = Phi_np.shape
        o = Y_np.shape[-1]

        # --- Host sklearn loop: one fit per (seed, output)
        coef_np = np.zeros((B, o, F), dtype=np.float32)
        cov_np = np.zeros((B, o, F, F), dtype=np.float32)
        noise_np = np.zeros((B, o), dtype=np.float32)
        lml_np = np.zeros(B, dtype=np.float32)

        for b in range(B):
            for i in range(o):
                est = estimator_cls(compute_score=True, **estimator_kwargs)
                est.fit(Phi_np[b], Y_np[b, :, i])
                coef_np[b, i] = est.coef_.astype(np.float32)
                # extract_sigma_fn handles the full vs pruned-feature covariance.
                # Add a small jitter for numerical stability before Cholesky.
                sigma = extract_sigma_fn(est, F)
                sigma += 1e-6 * np.eye(F, dtype=np.float32)
                cov_np[b, i] = sigma
                noise_np[b, i] = 1.0 / float(est.alpha_)
                score = float(est.scores_[-1])
                if np.isfinite(score):
                    lml_np[b] += score

        # --- JAX write: Cholesky, sample weights, assign Variables (jit+vmap)
        _write(
            model,
            jnp.asarray(coef_np),
            jnp.asarray(cov_np),
            jnp.asarray(noise_np),
            rngs,
        )

        # Return per-seed negative LML as the loss metric (shape (B, 1))
        return {"loss": jnp.asarray(-lml_np)[:, None]}

    return train_fn


def _make_skblr_train_model(update_steps, minibatch_size):
    # fit_intercept=False: rff_features prepends a bias column; sklearn's default
    # centering would zero it out and learn a separate intercept_ we never apply.
    # lambda_init/alpha_init: seed EM at blr_rff's fixed hyperparameters (LAM=0.01,
    # A0/B0=1.0) so the first iteration is well-conditioned even when F ≈ n.
    return _make_batched_train_model(
        BayesianRidge,
        {"fit_intercept": False, "lambda_init": 0.01, "alpha_init": 1.0},
        _extract_sigma_full,
    )


def _make_skard_train_model(update_steps, minibatch_size):
    # ARDRegression has no lambda_init/alpha_init; it prunes per-feature weights
    # so it's naturally robust to F ≈ n — no collapse initialisation needed.
    return _make_batched_train_model(
        ARDRegression,
        {"fit_intercept": False},
        _extract_sigma_ard,
    )


register_model("skblr")(
    {
        "make_batched_model": _make_batched_model,
        "make_batched_train_model": _make_skblr_train_model,
    }
)

register_model("skard")(
    {
        "make_batched_model": _make_batched_model,
        "make_batched_train_model": _make_skard_train_model,
    }
)
