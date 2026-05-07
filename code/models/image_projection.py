import dataclasses
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from code.inference.svi import run_svi, SVIResult
from code.prep_data.pipeline import TrialBatch


@dataclass
class ImageProjection:
    A:    np.ndarray  # (D, rank)
    B:    np.ndarray  # (rank, D)
    rank: int


def apply_projection(proj: ImageProjection, image_embs: np.ndarray) -> np.ndarray:
    """
    Apply the low-rank residual projection P = I + A @ B to image_embs.

    Returns L2-normalized projected embeddings with the same shape as image_embs.
    When A=0 and B=0 this reduces to normalizing the input.
    """
    projected = image_embs + (image_embs @ proj.B.T) @ proj.A.T  # (n, D)
    norms = np.linalg.norm(projected, axis=-1, keepdims=True)
    return projected / norms


def project_batch(proj: ImageProjection, batch: TrialBatch) -> TrialBatch:
    """
    Return a new TrialBatch with option_embs and all_image_embs replaced by
    their projected, L2-normalized counterparts.  All other fields are unchanged.
    """
    projected_all = apply_projection(proj, np.array(batch.all_image_embs))  # (n_img, D)
    new_option_embs = projected_all[np.array(batch.option_image_ids)]        # (N, 12, D)
    return dataclasses.replace(
        batch,
        option_embs=jnp.array(new_option_embs),
        all_image_embs=jnp.array(projected_all),
    )


def _image_projection_model(batch: TrialBatch, rank: int) -> None:
    D = batch.all_image_embs.shape[1]
    A = numpyro.param("proj_A", jnp.zeros((D, rank)))   # (D, rank)
    B = numpyro.param("proj_B", jnp.zeros((rank, D)))   # (rank, D)
    log_beta = numpyro.param("log_beta", jnp.array(0.0))

    projected = batch.all_image_embs + (batch.all_image_embs @ B.T) @ A.T  # (n_img, D)
    norms = jnp.linalg.norm(projected, axis=-1, keepdims=True)
    projected_norm = projected / norms

    option_embs = projected_norm[batch.option_image_ids]  # (N, 12, D)
    cos_sims = jnp.einsum("nd,nkd->nk", batch.utterance_emb, option_embs)
    logits = jnp.exp(log_beta) * cos_sims

    with numpyro.plate("trials", batch.utterance_emb.shape[0]):
        numpyro.sample("selected", dist.Categorical(logits=logits), obs=batch.selected_idx)


def _null_guide(*args, **kwargs) -> None:
    pass


def fit_image_projection(
    batch: TrialBatch,
    rank: int = 8,
    n_steps: int = 3000,
    lr: float = 0.01,
    seed: int = 0,
) -> ImageProjection:
    """
    Fit a low-rank residual image projection P = I + A @ B from listener choice data.

    Intended to be called on rep-1 trials only, so the learned projection is
    grounded in the most image-descriptive language rather than game-specific
    shorthand that develops in later rounds.

    Raises ValueError if batch.option_image_ids or batch.all_image_embs is None.
    """
    if batch.option_image_ids is None or batch.all_image_embs is None:
        raise ValueError(
            "batch.option_image_ids and batch.all_image_embs must be set; "
            "use build_trial_batch_from_df to build the batch"
        )

    def model(b):
        return _image_projection_model(b, rank=rank)

    result: SVIResult = run_svi(
        model,
        _null_guide,
        (batch,),
        n_steps=n_steps,
        lr=lr,
        seed=seed,
    )
    return ImageProjection(
        A=np.array(result.params["proj_A"]),
        B=np.array(result.params["proj_B"]),
        rank=rank,
    )


def save_image_projection(proj: ImageProjection, path: str) -> None:
    np.savez(path, A=proj.A, B=proj.B, rank=np.array(proj.rank))


def load_image_projection(path: str) -> ImageProjection:
    d = np.load(path)
    return ImageProjection(A=d["A"], B=d["B"], rank=int(d["rank"]))
