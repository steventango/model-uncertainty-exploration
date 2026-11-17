import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from models.base import head_out_features
from models.enn import ENNModel, _beta_nll
from networks import ENN

OBS, ACT = 3, 1


def _model(
    beta_nll=0.0, min_scale=1e-3, head="gaussian", predict_reward_terminated=False
):
    rngs = nnx.Rngs(0)
    in_features = OBS + ACT
    enn = ENN(
        in_features,
        16,
        8,
        4,
        head_out_features(head, OBS, predict_reward_terminated),
        4,
        num_layers=2,
        activation="silu",
        epinet_num_layers=1,
        epinet_activation="silu",
        rngs=rngs,
    )
    return ENNModel(
        enn,
        in_features,
        OBS,
        act_dim=ACT,
        predict_reward_terminated=predict_reward_terminated,
        head=head,
        min_scale=min_scale,
        beta_nll=beta_nll,
        rngs=rngs,
    )


def _raw_and_target(n=5, seed=0):
    k1, k2 = jax.random.split(jax.random.key(seed))
    return jax.random.normal(k1, (n, 2 * OBS)), jax.random.normal(k2, (n, OBS))


def test_beta_nll_at_zero_is_the_gaussian_nll():
    model = _model(beta_nll=0.0)
    y, target = _raw_and_target()
    dist = model.dist_from_output(y)
    # distrax includes 0.5 * d * log(2π); _beta_nll drops it and averages over dims.
    expected = (-dist.log_prob(target) - 0.5 * OBS * jnp.log(2 * jnp.pi)) / OBS
    assert jnp.allclose(_beta_nll(model, y, target), expected, atol=1e-5)


def test_beta_nll_weights_each_dim_by_sigma_to_the_2beta():
    beta = 0.5
    model = _model(beta_nll=beta)
    y, target = _raw_and_target()
    loc, scale = model._loc_scale(y)
    nll = 0.5 * ((target - loc) / scale) ** 2 + jnp.log(scale)
    expected = (scale ** (2 * beta) * nll).mean(-1)
    assert jnp.allclose(_beta_nll(model, y, target), expected, atol=1e-6)


def test_beta_one_gives_the_mean_the_mse_gradient():
    model = _model(beta_nll=1.0)
    y, target = _raw_and_target()
    grad = jax.grad(lambda y_: _beta_nll(model, y_, target).sum())(y)
    loc = y[..., :OBS]
    # The sigma^2 weight cancels the 1/sigma^2 in the squared error (mean over dims).
    assert jnp.allclose(grad[..., :OBS], (loc - target) / OBS, atol=1e-5)


def test_beta_weight_has_no_gradient():
    model = _model(beta_nll=1.0)
    y, target = _raw_and_target()
    loc, scale = model._loc_scale(y)
    # d/d raw_scale of sg(sigma^2) * nll = sigma^2 * d nll / d sigma * d sigma / d raw.
    dnll_dsigma = -((target - loc) ** 2) / scale**3 + 1 / scale
    dsigma_draw = jax.nn.sigmoid(y[..., OBS:])
    expected = scale**2 * dnll_dsigma * dsigma_draw / OBS
    grad = jax.grad(lambda y_: _beta_nll(model, y_, target).sum())(y)
    assert jnp.allclose(grad[..., OBS:], expected, atol=1e-4)


@pytest.mark.parametrize("kind", ["std", "eig"])
def test_uncertainty_shapes_and_sign(kind):
    model = _model()
    x = jax.random.normal(jax.random.key(1), (OBS + ACT,))
    z = model.sample_index(jax.random.key(2), 6)
    per_dim = model.uncertainty(x, z, kind, reduce_output=False)
    total = model.uncertainty(x, z, kind)
    assert per_dim.shape == (OBS,)
    assert total.shape == ()
    assert (per_dim >= 0).all()
    assert jnp.allclose(total, per_dim.sum(), atol=1e-6)


def test_std_uncertainty_is_the_spread_of_the_mean_across_indices():
    model = _model()
    x = jax.random.normal(jax.random.key(1), (OBS + ACT,))
    z = model.sample_index(jax.random.key(2), 6)
    loc, _ = model._loc_scale(model.predict_samples(x, z))
    assert jnp.allclose(
        model.uncertainty(x, z, "std", reduce_output=False), loc.std(axis=0), atol=1e-6
    )


def test_gaussian_head_requires_the_oracle_reward():
    with pytest.raises(NotImplementedError):
        head_out_features("gaussian", OBS, True)
    with pytest.raises(NotImplementedError):
        _model(predict_reward_terminated=True)
