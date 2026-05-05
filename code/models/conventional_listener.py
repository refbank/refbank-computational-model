import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from code.models.conventional_speaker import compute_sigma2_t
from code.prep_data.pipeline import TrialBatch


def listener_convention_model(
    batch: TrialBatch,
    s_t_binary: jnp.ndarray,
    mu_i_init: np.ndarray,
) -> None:
    """
    NumPyro model for listener-side conventions.
    Same structure as speaker_convention_model; s_t is binary correctness.
    """
    D = mu_i_init.shape[1]

    sigma_game = numpyro.sample("sigma_game", dist.HalfNormal(1.0))
    sigma_min = numpyro.sample("sigma_min", dist.HalfNormal(1.0))
    sigma_delta = numpyro.sample("sigma_delta", dist.HalfNormal(1.0))
    sigma_max = numpyro.deterministic("sigma_max", sigma_min + sigma_delta)

    mu_i = numpyro.sample(
        "mu_i",
        dist.Normal(jnp.array(mu_i_init), jnp.ones((batch.n_images, D))).to_event(1),
    )

    with numpyro.plate("games", batch.n_games):
        with numpyro.plate("images_in_games", batch.n_images):
            delta_gi = numpyro.sample(
                "delta_gi",
                dist.Normal(jnp.zeros(D), sigma_game).to_event(1),
            )

    delta_gi_t = jnp.moveaxis(delta_gi, 1, 0)  # (n_games, n_images, D)
    m_gi = mu_i[None, :, :] + delta_gi_t

    m_per_trial = m_gi[batch.game_ids, batch.image_ids]  # (N, D)
    sigma2_t = jnp.maximum(compute_sigma2_t(s_t_binary, sigma_min, sigma_max), 1e-6)
    sigma_t = jnp.sqrt(sigma2_t)

    with numpyro.plate("trials", batch.utterance_emb.shape[0]):
        numpyro.sample(
            "utterance_emb",
            dist.Normal(m_per_trial, sigma_t[:, None]).to_event(1),
            obs=batch.utterance_emb,
        )
