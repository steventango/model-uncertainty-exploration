from __future__ import annotations

import gpjax as gpx
import jax.numpy as jnp
import jax.random as jr
import optax as ox

from mue.types import ModelPrediction


class GPModel:
    """Independent GP per output dimension for dynamics + reward prediction.

    Predicts p(s', r | s, a) by fitting one GP per output dimension,
    using [s, a] as input and [s' - s, r] as targets.

    Uses an ARD (Automatic Relevance Determination) RBF kernel with
    per-dimension lengthscales, input/output normalization, and learned
    observation noise.  Cross-output correlations are *not* captured;
    use ``LCMGPModel`` when those matter.

    Supports the ``FantasyModel`` protocol: calling ``condition_on`` returns
    a lightweight copy whose posteriors include the fantasy data, so
    sequential information gain can be computed without re-fitting.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        learning_rate: float = 0.01,
        num_iters: int = 500,
        seed: int = 0,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.in_dim = obs_dim + act_dim
        self.out_dim = obs_dim + 1  # state deltas + reward
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.key = jr.key(seed)

        self._posteriors: list[gpx.gps.ConjugatePosterior] = []
        self._train_data: list[gpx.Dataset] = []

        # Normalization statistics (set during fit)
        self._X_mean: jnp.ndarray | None = None
        self._X_std: jnp.ndarray | None = None
        self._Y_mean: jnp.ndarray | None = None
        self._Y_std: jnp.ndarray | None = None

    def _build_gp(self, n: int) -> gpx.gps.ConjugatePosterior:
        kernel = gpx.kernels.RBF(n_dims=self.in_dim)
        mean_function = gpx.mean_functions.Zero()
        prior = gpx.gps.Prior(kernel=kernel, mean_function=mean_function, jitter=1e-4)
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=n)
        return prior * likelihood

    def _normalize_X(self, X: jnp.ndarray) -> jnp.ndarray:
        assert self._X_mean is not None and self._X_std is not None
        return (X - self._X_mean) / self._X_std

    def _normalize_Y(self, Y: jnp.ndarray) -> jnp.ndarray:
        assert self._Y_mean is not None and self._Y_std is not None
        return (Y - self._Y_mean) / self._Y_std

    def _denormalize_Y_mean(self, mean: jnp.ndarray) -> jnp.ndarray:
        assert self._Y_mean is not None and self._Y_std is not None
        return mean * self._Y_std + self._Y_mean

    def _denormalize_Y_var(self, var: jnp.ndarray) -> jnp.ndarray:
        assert self._Y_std is not None
        return var * self._Y_std**2

    def fit(
        self,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
        next_obs: jnp.ndarray,
        rewards: jnp.ndarray,
    ) -> dict[str, float]:
        # Promote to float64 for numerical stability in Cholesky / MLL
        X = jnp.concatenate([obs, actions], axis=-1).astype(jnp.float64)
        deltas = (next_obs - obs).astype(jnp.float64)
        rewards_f64 = rewards.astype(jnp.float64)
        # Targets: [delta_0, ..., delta_d, reward]
        targets = jnp.concatenate([deltas, rewards_f64.reshape(-1, 1)], axis=-1)

        # Compute and store normalization statistics
        self._X_mean = X.mean(axis=0)
        self._X_std = jnp.maximum(X.std(axis=0), 1e-8)
        self._Y_mean = targets.mean(axis=0)
        self._Y_std = jnp.maximum(targets.std(axis=0), 1e-8)

        X_norm = self._normalize_X(X)
        targets_norm = self._normalize_Y(targets)

        self._posteriors = []
        self._train_data = []
        total_loss = 0.0

        for i in range(self.out_dim):
            y = targets_norm[:, i : i + 1]
            dataset = gpx.Dataset(X=X_norm, y=y)
            posterior = self._build_gp(n=X_norm.shape[0])

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
            self._train_data.append(dataset)
            total_loss += float(history[-1])

        return {"model/loss": total_loss / self.out_dim}

    def predict(
        self,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> ModelPrediction:
        X = jnp.concatenate([obs, actions], axis=-1).astype(jnp.float64)
        X_norm = self._normalize_X(X)
        n = X.shape[0]

        means = []
        total_variances = []
        epistemic_variances = []
        aleatoric_variances = []

        for i, (posterior, train_data) in enumerate(
            zip(self._posteriors, self._train_data)
        ):
            pred_dist = posterior.predict(
                X_norm, train_data, return_covariance_type="diagonal"
            )

            # Denormalize mean and variance back to original target space
            y_std_i = self._Y_std[i]
            y_mean_i = self._Y_mean[i]
            means.append(pred_dist.mean * y_std_i + y_mean_i)

            # Total predictive variance (epistemic + aleatoric) in original space
            total_var = pred_dist.variance * y_std_i**2
            total_variances.append(total_var)

            # Aleatoric = likelihood noise variance, scaled back to original space
            noise_var = posterior.likelihood.obs_stddev**2 * y_std_i**2
            aleatoric_var = jnp.full_like(total_var, noise_var)
            aleatoric_variances.append(aleatoric_var)

            # Epistemic = total - aleatoric (posterior kernel variance)
            epistemic_var = jnp.maximum(total_var - noise_var, 0.0)
            epistemic_variances.append(epistemic_var)

        # (batch, out_dim) where out_dim = obs_dim + 1
        mean_raw = jnp.stack(means, axis=-1)
        total_var_stack = jnp.stack(total_variances, axis=-1)
        epistemic_var_stack = jnp.stack(epistemic_variances, axis=-1)
        aleatoric_var_stack = jnp.stack(aleatoric_variances, axis=-1)

        # Add obs back to the state delta dimensions, leave reward as-is
        mean = mean_raw.at[:, : self.obs_dim].add(obs.astype(jnp.float64))

        # Build diagonal covariance matrices
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

    def condition_on(
        self,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
        targets: jnp.ndarray,
    ) -> GPModel:
        """Return a new GPModel conditioned on fantasy observations.

        Creates a lightweight copy that shares the trained kernel
        hyperparameters but has augmented training data. This is the GP
        analogue of "fantasizing": we pretend we observed ``targets`` at
        inputs ``(obs, actions)`` and update the posterior accordingly.

        Because the kernel hyperparameters are kept fixed (no re-fitting),
        this is O(1) in creation cost — the extra cost is deferred to the
        next ``predict`` call, which must invert a slightly larger kernel
        matrix.

        Args:
            obs: (n, obs_dim) observation inputs for the fantasy points.
            actions: (n, act_dim) action inputs for the fantasy points.
            targets: (n, out_dim) fantasy targets in the GP's target space
                (state deltas + reward), *not* absolute next states.

        Returns:
            A new GPModel whose posteriors include the fantasy data.
        """
        fantasy_X = jnp.concatenate([obs, actions], axis=-1).astype(jnp.float64)
        fantasy_X_norm = self._normalize_X(fantasy_X)
        targets_norm = self._normalize_Y(targets.astype(jnp.float64))

        new_model = GPModel.__new__(GPModel)
        new_model.obs_dim = self.obs_dim
        new_model.act_dim = self.act_dim
        new_model.in_dim = self.in_dim
        new_model.out_dim = self.out_dim
        new_model.learning_rate = self.learning_rate
        new_model.num_iters = self.num_iters
        new_model.key = self.key
        new_model._X_mean = self._X_mean
        new_model._X_std = self._X_std
        new_model._Y_mean = self._Y_mean
        new_model._Y_std = self._Y_std

        new_posteriors = []
        new_datasets = []

        for i, (posterior, dataset) in enumerate(
            zip(self._posteriors, self._train_data)
        ):
            y_fantasy = targets_norm[:, i : i + 1]

            assert dataset.X is not None and dataset.y is not None
            new_X = jnp.concatenate([dataset.X, fantasy_X_norm], axis=0)
            new_y = jnp.concatenate([dataset.y, y_fantasy], axis=0)
            new_dataset = gpx.Dataset(X=new_X, y=new_y)

            # Rebuild posterior with the same kernel hyperparameters but
            # updated datapoint count.
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
        new_model._train_data = new_datasets
        return new_model
