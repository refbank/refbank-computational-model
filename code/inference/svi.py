from dataclasses import dataclass
from typing import Callable

import jax
import numpy as np
import numpyro
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal

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
    _inject_nan_at_step: int | None = None,
) -> SVIResult:
    """
    Adam + ELBO SVI via NumPyro.
    Raises RuntimeError("NaN loss at step {i}") if loss is NaN after step 100.
    _inject_nan_at_step: test-only hook to inject NaN at a specific step.
    """
    optimizer = numpyro.optim.Adam(step_size=lr)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    rng_key = jax.random.PRNGKey(seed)
    state = svi.init(rng_key, *model_args)

    losses = np.empty(n_steps)
    for i in range(n_steps):
        state, loss = svi.update(state, *model_args)
        if _inject_nan_at_step is not None and i == _inject_nan_at_step:
            loss = float("nan")
        losses[i] = loss
        if np.isnan(loss) and i >= 100:
            raise RuntimeError(f"NaN loss at step {i}")

    return SVIResult(params=svi.get_params(state), losses=losses)
