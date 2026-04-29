import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpy as np
import pytest
from unittest.mock import patch

from code.inference.svi import run_svi, SVIResult


def _gaussian_model(obs):
    mu = numpyro.sample("mu", dist.Normal(0, 10))
    numpyro.sample("obs", dist.Normal(mu, 1), obs=obs)


def _gaussian_guide(obs):
    mu_loc = numpyro.param("mu_loc", jnp.array(0.0))
    mu_scale = numpyro.param("mu_scale", jnp.array(1.0), constraint=dist.constraints.positive)
    numpyro.sample("mu", dist.Normal(mu_loc, mu_scale))


def test_run_svi_returns_svi_result():
    obs = jnp.array([1.5, 2.0, 1.8])
    result = run_svi(_gaussian_model, _gaussian_guide, (obs,), n_steps=50)
    assert isinstance(result, SVIResult)


def test_run_svi_returns_correct_losses_length():
    obs = jnp.array([1.5, 2.0, 1.8])
    result = run_svi(_gaussian_model, _gaussian_guide, (obs,), n_steps=100)
    assert len(result.losses) == 100


def test_run_svi_loss_decreases():
    obs = jnp.ones(50) * 3.0
    result = run_svi(_gaussian_model, _gaussian_guide, (obs,), n_steps=500)
    assert result.losses[-1] < result.losses[0], (
        f"Loss did not decrease: {result.losses[0]:.2f} → {result.losses[-1]:.2f}"
    )


def test_run_svi_nan_raises_after_step_100():
    call_count = [0]
    original_update = None

    def model(obs):
        mu = numpyro.sample("mu", dist.Normal(0, 10))
        numpyro.sample("obs", dist.Normal(mu, 1), obs=obs)

    obs = jnp.array([1.0])
    with pytest.raises(RuntimeError, match="NaN loss at step"):
        run_svi(model, _gaussian_guide, (obs,), n_steps=200, _inject_nan_at_step=150)


def test_run_svi_nan_before_step_100_does_not_raise():
    obs = jnp.array([1.0])
    result = run_svi(
        _gaussian_model, _gaussian_guide, (obs,), n_steps=200, _inject_nan_at_step=50
    )
    assert len(result.losses) == 200
