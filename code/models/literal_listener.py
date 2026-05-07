from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from code.inference.svi import run_svi, SVIResult
from code.prep_data.pipeline import TrialBatch


def literal_listener_model(batch: TrialBatch) -> None:
    """
    NumPyro model for the literal L_0 listener.
    Plates over listeners for beta_l ~ LogNormal(mu_beta, sigma_beta)
    and over trials for the 12-way choice likelihood.

    Image embeddings are fixed (passed via batch.option_embs).  Use the
    image projection pre-step to improve alignment before fitting.
    """
    mu_beta = numpyro.sample("mu_beta", dist.Normal(0, 2))
    sigma_beta = numpyro.sample("sigma_beta", dist.HalfNormal(1))

    with numpyro.plate("listeners", batch.n_listeners):
        log_beta = numpyro.sample("log_beta", dist.Normal(mu_beta, sigma_beta))

    cos_sims = jnp.einsum("nd,nkd->nk", batch.utterance_emb, batch.option_embs)
    beta_per_trial = jnp.exp(log_beta[batch.listener_ids])  # (N,)
    logits = beta_per_trial[:, None] * cos_sims              # (N, 12)

    with numpyro.plate("trials", batch.utterance_emb.shape[0]):
        numpyro.sample(
            "selected",
            dist.Categorical(logits=logits),
            obs=batch.selected_idx,
        )


def literal_listener_guide(batch: TrialBatch) -> None:
    """Mean-field variational guide for literal_listener_model."""
    mu_beta_loc = numpyro.param("mu_beta_loc", jnp.array(0.0))
    mu_beta_scale = numpyro.param(
        "mu_beta_scale", jnp.array(1.0), constraint=dist.constraints.positive
    )
    sigma_beta_loc = numpyro.param("sigma_beta_loc", jnp.array(0.5))

    numpyro.sample("mu_beta", dist.Normal(mu_beta_loc, mu_beta_scale))
    numpyro.sample(
        "sigma_beta",
        dist.TransformedDistribution(
            dist.Normal(sigma_beta_loc, jnp.array(0.5)),
            dist.transforms.ExpTransform(),
        ),
    )

    log_beta_loc = numpyro.param("log_beta_loc", jnp.zeros(batch.n_listeners))
    log_beta_scale = numpyro.param(
        "log_beta_scale",
        jnp.ones(batch.n_listeners) * 0.5,
        constraint=dist.constraints.positive,
    )
    with numpyro.plate("listeners", batch.n_listeners):
        numpyro.sample("log_beta", dist.Normal(log_beta_loc, log_beta_scale))


@dataclass
class ListenerFit:
    beta_loc:   np.ndarray  # (n_listeners,) posterior mean of log beta_l
    beta_scale: np.ndarray  # (n_listeners,) posterior std of log beta_l
    mu_beta:    float
    sigma_beta: float


def fit_listener(
    batch: TrialBatch,
    n_steps: int = 5000,
    lr: float = 0.01,
    seed: int = 0,
) -> ListenerFit:
    """Fits literal_listener_model via SVI. Raises RuntimeError on NaN loss."""
    result: SVIResult = run_svi(
        literal_listener_model,
        literal_listener_guide,
        (batch,),
        n_steps=n_steps,
        lr=lr,
        seed=seed,
    )
    return ListenerFit(
        beta_loc=np.array(result.params["log_beta_loc"]),
        beta_scale=np.array(result.params["log_beta_scale"]),
        mu_beta=float(result.params["mu_beta_loc"]),
        sigma_beta=float(np.exp(result.params["sigma_beta_loc"])),
    )


def save_listener_fit(fit: ListenerFit, path: str) -> None:
    np.savez(
        path,
        beta_loc=fit.beta_loc,
        beta_scale=fit.beta_scale,
        mu_beta=np.array(fit.mu_beta),
        sigma_beta=np.array(fit.sigma_beta),
    )


def load_listener_fit(path: str) -> ListenerFit:
    d = np.load(path)
    return ListenerFit(
        beta_loc=d["beta_loc"],
        beta_scale=d["beta_scale"],
        mu_beta=float(d["mu_beta"]),
        sigma_beta=float(d["sigma_beta"]),
    )


def compute_success_probs(fit: ListenerFit, batch: TrialBatch) -> np.ndarray:
    """
    Returns (N,) array: s_t = L_0(target_idx | u, O, listener).
    Uses posterior mean beta (exp of beta_loc) for each listener.
    """
    beta = np.exp(fit.beta_loc)  # (n_listeners,)
    cos_sims = np.einsum(
        "nd,nkd->nk",
        np.array(batch.utterance_emb),
        np.array(batch.option_embs),
    )  # (N, 12)
    beta_per_trial = beta[np.array(batch.listener_ids)]  # (N,)
    logits = beta_per_trial[:, None] * cos_sims           # (N, 12)

    logits_shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits_shifted)
    probs /= probs.sum(axis=1, keepdims=True)

    target_probs = probs[np.arange(len(batch.target_idx)), np.array(batch.target_idx)]
    return target_probs
