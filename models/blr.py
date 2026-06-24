import jax
import jax.numpy as jnp
import jax.scipy.linalg
from flax import nnx

from models.base import WorldModel, register_model


class BLRModel(WorldModel):
    def __init__(
        self,
        in_features: int,
        obs_dim: int,
        out_features: int,
        num_features: int,
        num_samples: int,
        lam: float,
        a0: float,
        b0: float,
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
        self.lam = lam
        self.a0 = a0
        self.b0 = b0
        self.length_scale = length_scale

        d = in_features
        o = out_features
        F = num_features + 1
        S = num_samples

        self.C = nnx.Variable(jnp.zeros((num_features, d), dtype=jnp.float32))
        self.n_valid = nnx.Variable(jnp.zeros((), dtype=jnp.int32))

        self.w_mean = nnx.Variable(jnp.zeros((o, F)))
        self.w_samples = nnx.Variable(jnp.zeros((S, o, F)))
        self.noise = nnx.Variable(jnp.ones((o,)))
        # Cholesky of Λ_n (lower triangular, shared across outputs).
        self.L = nnx.Variable(jnp.eye(F))

    def _features(self, x):
        """RBF features + bias; invalid centers zeroed out.  x: (..., d) → (..., num_features+1)."""
        diff = jnp.expand_dims(x, axis=-2) - self.C.value
        sq_dist = jnp.sum(diff**2, axis=-1)
        phi = jnp.exp(-sq_dist / (2 * self.length_scale**2))
        valid = jnp.arange(self.num_features) < self.n_valid.value
        rbf = phi * valid
        bias = jnp.ones(x.shape[:-1] + (1,))
        return jnp.concatenate([bias, rbf], axis=-1)

    def sample_index(self, key, num_samples: int):
        return jax.random.randint(key, (num_samples,), 0, self.num_samples)

    def predict_mean(self, x):
        """x: (in_features,) → (out_features,)."""
        return self.w_mean.value @ self._features(x)

    def predict_sample(self, x, index):
        return self.w_samples.value[index] @ self._features(x)

    def variance(self, x, z):
        """Closed-form posterior variance."""
        phi = self._features(x)
        v = jax.scipy.linalg.solve_triangular(self.L.value, phi, lower=True)
        return self.noise.value * (v @ v)


def _train_model(
    model: BLRModel,
    train_state,
    dataset,
    update_steps: int,
    minibatch_size: int,
    pointer,
    rngs: nnx.Rngs,
):
    F = model.num_features + 1
    S = model.num_samples
    lam = model.lam
    a_0 = model.a0
    b_0 = model.b0

    model.update_stats(dataset, pointer)

    N = dataset.obs.shape[0]
    mask = (jnp.arange(N) < pointer).astype(jnp.float32)[:, None]
    N_eff = mask.sum()

    X = model.normalize_input(model.build_input(dataset.obs, dataset.action))
    model.C.value = X
    model.n_valid.value = pointer.astype(jnp.int32)

    Phi = model._features(X)
    Phi = Phi * mask

    Y = model.build_targets(dataset.obs, dataset.info["next_obs"], dataset.reward, dataset.terminated) * mask

    gram = Phi.T @ Phi
    Phi_y = Phi.T @ Y
    YtY = jnp.sum(Y**2, axis=0)

    Lambda_n = gram + lam * jnp.eye(F)
    L = jnp.linalg.cholesky(Lambda_n)

    o = model.out_features
    mu_n = jnp.linalg.solve(Lambda_n, Phi_y).T

    a_n = a_0 + 0.5 * N_eff
    b_n = b_0 + 0.5 * (YtY - jnp.sum(mu_n * Phi_y.T, axis=-1))
    sigma2 = b_n / (a_n + 1.0)

    z = jax.random.normal(rngs(), (S * o, F))
    Linv_z = jax.scipy.linalg.solve_triangular(L.T, z.T, lower=False).T
    Linv_z = Linv_z.reshape(S, o, F)
    w_samples = mu_n[None] + jnp.sqrt(sigma2)[None, :, None] * Linv_z

    log_det_Lambda_n = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    log_det_Lambda_0 = F * jnp.log(lam)
    log_marginal_likelihood = (
        -0.5 * N_eff * jnp.log(2 * jnp.pi)
        + 0.5 * log_det_Lambda_0
        -0.5 * log_det_Lambda_n
        + a_0 * jnp.log(b_0)
        - a_n * jnp.log(b_n)
        + jax.scipy.special.gammaln(a_n)
        - jax.scipy.special.gammaln(a_0)
    ).sum()

    model.noise.value = sigma2
    model.w_mean.value = mu_n
    model.w_samples.value = w_samples
    model.L.value = L

    return {"loss": jnp.array([-log_marginal_likelihood])}


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
        return BLRModel(
            in_features=in_features,
            obs_dim=obs_dim,
            out_features=out_features,
            num_features=model_config["MAX_DATA"],
            num_samples=model_config["NUM_SAMPLES"],
            lam=model_config["LAM"],
            a0=model_config["A0"],
            b0=model_config["B0"],
            length_scale=model_config["LENGTH_SCALE"],
            key=key,
            act_dim=act_dim,
            predict_reward_terminated=predict_reward_terminated,
        )

    return build(keys), None


def _make_batched_train_model(update_steps, minibatch_size):
    def core(model, train_state, dataset, pointer, rngs):
        return _train_model(
            model, train_state, dataset, update_steps, minibatch_size, pointer, rngs
        )

    return nnx.jit(nnx.vmap(core, in_axes=(0, None, 0, None, 0), out_axes=0))


register_model("blr")(
    {
        "make_batched_model": _make_batched_model,
        "make_batched_train_model": _make_batched_train_model,
    }
)
