from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import SVI, Trace_ELBO


@dataclass
class SVIResult:
    params: dict
    losses: np.ndarray  # (n_steps,)


def run_svi(
    model: Callable,
    guide: Callable,
    model_args: tuple,
    n_steps: int,
    lr: float = 0.01,
    seed: int = 0,
    clip_norm: float | None = None,
    _inject_nan_at_step: int | None = None,
) -> SVIResult:
    """
    Adam + ELBO SVI via NumPyro, compiled with lax.scan for GPU efficiency.
    clip_norm: if set, uses ClippedAdam to bound gradient norms.
    Raises RuntimeError("NaN loss at step {i}") if loss is NaN after step 100.
    _inject_nan_at_step: test-only hook to inject NaN at a specific step.
    """
    if clip_norm is not None:
        optimizer = numpyro.optim.ClippedAdam(step_size=lr, clip_norm=clip_norm)
    else:
        optimizer = numpyro.optim.Adam(step_size=lr)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    rng_key = jax.random.PRNGKey(seed)
    state = svi.init(rng_key, *model_args)

    def step_fn(state, i):
        state, loss = svi.update(state, *model_args)
        if _inject_nan_at_step is not None:
            loss = jnp.where(i == _inject_nan_at_step, jnp.nan, loss)
        return state, loss

    state, losses = jax.lax.scan(step_fn, state, jnp.arange(n_steps))
    losses = np.array(losses)

    nan_mask = np.isnan(losses)
    if np.any(nan_mask[100:]):
        first_nan = int(np.argmax(nan_mask[100:])) + 100
        raise RuntimeError(f"NaN loss at step {first_nan}")

    return SVIResult(params=svi.get_params(state), losses=losses)
