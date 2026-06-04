"""Gaussian Process world models with optional Linear Coregionalization.

- Independent GPs (latent_dim=None): one GP per output dimension, no cross-output correlations
- LCM GPs (latent_dim>0): shared LCM kernel capturing cross-output correlations
"""

from __future__ import annotations

import gpjax as gpx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import optax as ox
from gpjax.linalg import Dense
from gpjax.linalg.operations import lower_cholesky, solve
from gpjax.linalg.utils import psd

from mue.types import ModelPrediction


class _MultiOutputZeroMean(gpx.mean_functions.Zero):
    """Multi-output zero mean for LCM: returns (N*P, 1)."""

    def __init__(self, num_outputs: int):
        super().__init__()
        self.num_outputs = num_outputs

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.zeros((x.shape[0] * self.num_outputs, 1))


class _Normalizer:
    """Mixin for input/output normalization."""

    _X_mean: jnp.ndarray | None
    _X_std: jnp.ndarray | None
    _Y_mean: jnp.ndarray | None
    _Y_std: jnp.ndarray | None

    def _init_norm(self) -> None:
        self._X_mean = None
        self._X_std = None
        self._Y_mean = None
        self._Y_std = None

    def _fit_norm(
        self, X: jnp.ndarray, Y: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        self._X_mean = X.mean(axis=0)
        self._X_std = jnp.maximum(X.std(axis=0), 1e-8)
        self._Y_mean = Y.mean(axis=0)
        self._Y_std = jnp.maximum(Y.std(axis=0), 1e-8)
        return (X - self._X_mean) / self._X_std, (Y - self._Y_mean) / self._Y_std

    def _normalize_X(self, X: jnp.ndarray) -> jnp.ndarray:
        assert self._X_mean is not None and self._X_std is not None
        return (X - self._X_mean) / self._X_std

    def _normalize_Y(self, Y: jnp.ndarray) -> jnp.ndarray:
        assert self._Y_mean is not None and self._Y_std is not None
        return (Y - self._Y_mean) / self._Y_std


def _extract_lcm_prediction(
    posterior,
    train_data: gpx.Dataset,
    X_test: jnp.ndarray,
    obs: jnp.ndarray,
    obs_dim: int,
    out_dim: int,
    Y_mean: jnp.ndarray | None = None,
    Y_std: jnp.ndarray | None = None,
) -> ModelPrediction:
    """Extract prediction from LCM posterior."""
    N = X_test.shape[0]
    P = out_dim
    N_train = train_data.n

    kernel = posterior.prior.kernel
    jitter = posterior.prior.jitter

    # Build regularized training kernel: K + noise*I + jitter*I
    Kxx = kernel.gram(train_data.X).to_dense()  # (N_train*P, N_train*P)
    noise = posterior.likelihood.obs_stddev[...] ** 2  # (P,)
    noise_diag = jnp.repeat(noise, N_train)
    Kxx_reg = Kxx + jnp.diag(noise_diag) + jitter * jnp.eye(N_train * P)

    # Cross-covariance and test diagonal
    Kxs = kernel.cross_covariance(train_data.X, X_test)  # (N_train*P, N*P)
    Kss_diag = kernel.diagonal(X_test).diagonal  # (N*P,)

    # Predictive mean: Kxs^T @ (Kxx_reg)^{-1} @ y
    y = train_data.y.T.ravel()[:, None]  # (N_train*P, 1) output-major
    alpha = jnp.linalg.solve(Kxx_reg, y)
    pred_mean = (Kxs.T @ alpha).ravel()  # (N*P,)

    # Predictive variance: diag(Kss) - diag(Kxs^T @ Kxx_reg^{-1} @ Kxs)
    v = jnp.linalg.solve(Kxx_reg, Kxs)  # (N_train*P, N*P)
    pred_var = Kss_diag - jnp.sum(Kxs * v, axis=0)  # (N*P,)
    pred_var = jnp.maximum(pred_var, 0.0)

    # Reshape: output-major (NP,) -> (N, P)
    mean_raw = pred_mean.reshape(P, N).T
    total_var = pred_var.reshape(P, N).T

    # Per-output noise
    noise_per_output = posterior.likelihood.obs_stddev[...] ** 2  # (P,)

    # Denormalize if provided
    if Y_mean is not None and Y_std is not None:
        mean_raw = mean_raw * Y_std[None, :] + Y_mean[None, :]
        total_var = total_var * Y_std[None, :] ** 2
        noise_per_output = noise_per_output * Y_std**2

    # Clamp variances
    total_var = jnp.maximum(total_var, 0.0)

    # Add obs back to state-delta dimensions
    mean = mean_raw.at[:, :obs_dim].add(obs)

    aleatoric_var = jnp.broadcast_to(noise_per_output[None, :], (N, P))
    epistemic_var = jnp.maximum(total_var - aleatoric_var, 0.0)

    # Build covariance matrices (diagonal)
    covariance = jnp.zeros((N, P, P))
    epistemic_covariance = jnp.zeros((N, P, P))
    aleatoric_covariance = jnp.zeros((N, P, P))
    for i in range(P):
        covariance = covariance.at[:, i, i].set(total_var[:, i])
        epistemic_covariance = epistemic_covariance.at[:, i, i].set(epistemic_var[:, i])
        aleatoric_covariance = aleatoric_covariance.at[:, i, i].set(aleatoric_var[:, i])

    return ModelPrediction(
        mean=mean,
        covariance=covariance,
        epistemic_covariance=epistemic_covariance,
        aleatoric_covariance=aleatoric_covariance,
    )


class GPModel(_Normalizer):
    """Multi-output Gaussian Process with optional Linear Coregionalization.

    When latent_dim=None: Independent GPs per output dimension (classic approach).
    When latent_dim>0: LCM kernel with rank-latent_dim coregionalization matrices
    to capture cross-output correlations.

    Predicts p(s', r | s, a) using [s, a] as input and [s' - s, r] as targets.

    Supports the FantasyModel protocol: condition_on returns a copy with
    augmented training data for information gain computation.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        learning_rate: float = 0.01,
        num_iters: int = 500,
        latent_dim: int | None = None,
        seed: int = 0,
    ):
        """Initialize GPModel.

        Args:
            obs_dim: Observation dimensionality.
            act_dim: Action dimensionality.
            learning_rate: Optimizer learning rate.
            num_iters: Number of optimization iterations (for independent GPs).
            latent_dim: Latent dimensionality for LCM kernel.
                - None: independent GPs (no cross-output correlations)
                - >0: LCM kernel with rank-latent_dim coregionalization
            seed: Random seed.
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.in_dim = obs_dim + act_dim
        self.out_dim = obs_dim + 1  # state deltas + reward
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.latent_dim = latent_dim
        self.key = jr.key(seed)
        self._init_norm()

        # Independent GPs mode
        self._posteriors: list[gpx.gps.ConjugatePosterior] = []
        self._train_data_list: list[gpx.Dataset] = []

        # LCM mode
        self._posterior: gpx.gps.ConjugatePosterior | None = None
        self._train_data: gpx.Dataset | None = None

    def _build_independent_gp(self, n: int) -> gpx.gps.ConjugatePosterior:
        """Build a single independent GP for one output."""
        kernel = gpx.kernels.Matern52(n_dims=self.in_dim)
        mean_function = gpx.mean_functions.Zero()
        prior = gpx.gps.Prior(kernel=kernel, mean_function=mean_function, jitter=1e-4)
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=n)
        return prior * likelihood

    def _build_lcm_posterior(self, N: int) -> gpx.gps.ConjugatePosterior:
        """Build LCM posterior."""
        assert self.latent_dim is not None and self.latent_dim > 0
        subkeys = jr.split(self.key, self.latent_dim)
        coreg = [
            gpx.parameters.CoregionalizationMatrix(
                num_outputs=self.out_dim, rank=1, key=subkeys[q]
            )
            for q in range(self.latent_dim)
        ]
        kernel = gpx.kernels.LCMKernel(
            kernels=[gpx.kernels.Matern52(n_dims=self.in_dim) for _ in range(self.latent_dim)],
            coregionalization_matrices=coreg,
        )
        mean_fn = _MultiOutputZeroMean(self.out_dim)
        prior = gpx.gps.Prior(mean_function=mean_fn, kernel=kernel, jitter=1e-2)
        likelihood = gpx.likelihoods.MultiOutputGaussian(
            num_datapoints=N,
            num_outputs=self.out_dim,
            obs_stddev=0.1,
        )
        return prior * likelihood

    def fit(
        self,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
        next_obs: jnp.ndarray,
        rewards: jnp.ndarray,
    ) -> dict[str, float]:
        X = jnp.concatenate([obs, actions], axis=-1)
        deltas = next_obs - obs
        Y = jnp.concatenate([deltas, rewards.reshape(-1, 1)], axis=-1)
        N = X.shape[0]

        X_norm, Y_norm = self._fit_norm(X, Y)

        if self.latent_dim is None:
            # Independent GPs
            return self._fit_independent(X_norm, Y_norm, N)
        else:
            # LCM GP
            return self._fit_lcm(X_norm, Y_norm, N)

    def _fit_independent(
        self, X_norm: jnp.ndarray, Y_norm: jnp.ndarray, N: int
    ) -> dict[str, float]:
        """Fit independent GPs per output dimension."""
        self._posteriors = []
        self._train_data_list = []
        total_loss = 0.0

        for i in range(self.out_dim):
            y = Y_norm[:, i : i + 1]
            dataset = gpx.Dataset(X=X_norm, y=y)
            posterior = self._build_independent_gp(n=N)

            def objective(post, data):
                return -gpx.objectives.conjugate_mll(post, data)

            self.key, subkey = jr.split(self.key)
            trained_posterior, history = gpx.fit(
                model=posterior,
                objective=objective,
                train_data=dataset,
                optim=ox.adam(self.learning_rate),
                num_iters=self.num_iters,
                verbose=False,
                key=subkey,
            )

            self._posteriors.append(trained_posterior)
            self._train_data_list.append(dataset)
            total_loss += float(history[-1])

        return {"model/loss": total_loss / self.out_dim}

    def _fit_lcm(
        self, X_norm: jnp.ndarray, Y_norm: jnp.ndarray, N: int
    ) -> dict[str, float]:
        """Fit LCM GP."""
        dataset = gpx.Dataset(X=X_norm, y=Y_norm)
        self.key, subkey = jr.split(self.key)
        posterior = self._build_lcm_posterior(N)

        opt_posterior, history = gpx.fit_scipy(
            model=posterior,
            objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
            train_data=dataset,
            trainable=gpx.parameters.Parameter,
        )

        self._posterior = opt_posterior
        self._train_data = dataset
        return {"model/loss": float(history[-1])}

    def predict(
        self,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> ModelPrediction:
        X = jnp.concatenate([obs, actions], axis=-1)
        X_norm = self._normalize_X(X)
        n = X.shape[0]

        if self.latent_dim is None:
            return self._predict_independent(X_norm, obs, n)
        else:
            return self._predict_lcm(X_norm, obs, n)

    def _predict_independent(
        self, X_norm: jnp.ndarray, obs: jnp.ndarray, n: int
    ) -> ModelPrediction:
        """Predict with independent GPs."""
        means = []
        total_variances = []
        epistemic_variances = []
        aleatoric_variances = []

        for i, (posterior, train_data) in enumerate(
            zip(self._posteriors, self._train_data_list)
        ):
            pred_dist = posterior.predict(
                X_norm, train_data, return_covariance_type="diagonal"
            )

            y_std_i = self._Y_std[i]
            y_mean_i = self._Y_mean[i]
            means.append(pred_dist.mean * y_std_i + y_mean_i)

            total_var = pred_dist.variance * y_std_i**2
            total_variances.append(total_var)

            noise_var = posterior.likelihood.obs_stddev**2 * y_std_i**2
            aleatoric_var = jnp.full_like(total_var, noise_var)
            aleatoric_variances.append(aleatoric_var)

            epistemic_var = jnp.maximum(total_var - noise_var, 0.0)
            epistemic_variances.append(epistemic_var)

        mean_raw = jnp.stack(means, axis=-1)
        total_var_stack = jnp.stack(total_variances, axis=-1)
        epistemic_var_stack = jnp.stack(epistemic_variances, axis=-1)
        aleatoric_var_stack = jnp.stack(aleatoric_variances, axis=-1)

        mean = mean_raw.at[:, : self.obs_dim].add(obs)

        covariance = jnp.zeros((n, self.out_dim, self.out_dim))
        epistemic_covariance = jnp.zeros((n, self.out_dim, self.out_dim))
        aleatoric_covariance = jnp.zeros((n, self.out_dim, self.out_dim))
        for i in range(self.out_dim):
            covariance = covariance.at[:, i, i].set(total_var_stack[:, i])
            epistemic_covariance = epistemic_covariance.at[:, i, i].set(
                epistemic_var_stack[:, i]
            )
            aleatoric_covariance = aleatoric_covariance.at[:, i, i].set(
                aleatoric_var_stack[:, i]
            )

        return ModelPrediction(
            mean=mean.astype(jnp.float32),
            covariance=covariance.astype(jnp.float32),
            epistemic_covariance=epistemic_covariance.astype(jnp.float32),
            aleatoric_covariance=aleatoric_covariance.astype(jnp.float32),
        )

    def _predict_lcm(
        self, X_norm: jnp.ndarray, obs: jnp.ndarray, n: int
    ) -> ModelPrediction:
        """Predict with LCM GP."""
        assert self._posterior is not None and self._train_data is not None
        return _extract_lcm_prediction(
            self._posterior,
            self._train_data,
            X_norm,
            obs,
            self.obs_dim,
            self.out_dim,
            self._Y_mean,
            self._Y_std,
        )

    def condition_on(
        self,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
        targets: jnp.ndarray,
    ) -> GPModel:
        """Return a new GPModel conditioned on fantasy observations."""
        fantasy_X = jnp.concatenate([obs, actions], axis=-1)
        fantasy_X_norm = self._normalize_X(fantasy_X)
        targets_norm = self._normalize_Y(targets)

        new_model = GPModel.__new__(GPModel)
        new_model.obs_dim = self.obs_dim
        new_model.act_dim = self.act_dim
        new_model.in_dim = self.in_dim
        new_model.out_dim = self.out_dim
        new_model.learning_rate = self.learning_rate
        new_model.num_iters = self.num_iters
        new_model.latent_dim = self.latent_dim
        new_model.key = self.key
        new_model._X_mean = self._X_mean
        new_model._X_std = self._X_std
        new_model._Y_mean = self._Y_mean
        new_model._Y_std = self._Y_std

        if self.latent_dim is None:
            return self._condition_on_independent(new_model, fantasy_X_norm, targets_norm)
        else:
            return self._condition_on_lcm(new_model, fantasy_X_norm, targets_norm)

    def _condition_on_independent(
        self, new_model: GPModel, fantasy_X_norm: jnp.ndarray, targets_norm: jnp.ndarray
    ) -> GPModel:
        """Condition independent GPs on fantasy data."""
        new_posteriors = []
        new_datasets = []

        for i, (posterior, dataset) in enumerate(
            zip(self._posteriors, self._train_data_list)
        ):
            y_fantasy = targets_norm[:, i : i + 1]
            assert dataset.X is not None and dataset.y is not None
            new_X = jnp.concatenate([dataset.X, fantasy_X_norm], axis=0)
            new_y = jnp.concatenate([dataset.y, y_fantasy], axis=0)
            new_dataset = gpx.Dataset(X=new_X, y=new_y)

            n_new = new_X.shape[0]
            prior = gpx.gps.Prior(
                kernel=posterior.prior.kernel,
                mean_function=posterior.prior.mean_function,
            )
            likelihood = gpx.likelihoods.Gaussian(
                num_datapoints=n_new,
                obs_stddev=posterior.likelihood.obs_stddev,
            )
            new_posterior = prior * likelihood

            new_posteriors.append(new_posterior)
            new_datasets.append(new_dataset)

        new_model._posteriors = new_posteriors
        new_model._train_data_list = new_datasets
        return new_model

    def _condition_on_lcm(
        self, new_model: GPModel, fantasy_X_norm: jnp.ndarray, targets_norm: jnp.ndarray
    ) -> GPModel:
        """Condition LCM GP on fantasy data."""
        assert self._posterior is not None and self._train_data is not None
        assert self._train_data.X is not None and self._train_data.y is not None
        new_X = jnp.concatenate([self._train_data.X, fantasy_X_norm], axis=0)
        new_Y = jnp.concatenate([self._train_data.y, targets_norm], axis=0)
        new_dataset = gpx.Dataset(X=new_X, y=new_Y)
        N_new = new_X.shape[0]

        prior = gpx.gps.Prior(
            kernel=self._posterior.prior.kernel,
            mean_function=self._posterior.prior.mean_function,
        )
        likelihood = gpx.likelihoods.MultiOutputGaussian(
            num_datapoints=N_new,
            num_outputs=self.out_dim,
            obs_stddev=self._posterior.likelihood.obs_stddev,
        )
        new_posterior = prior * likelihood

        new_model._posterior = new_posterior
        new_model._train_data = new_dataset
        return new_model
