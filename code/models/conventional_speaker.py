from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from code.inference.svi import run_svi
from code.models.literal_listener import ListenerFit, compute_success_probs
from code.prep_data.pipeline import TrialBatch


def compute_sigma2_t(
    s_t: jnp.ndarray,
    sigma_min: float,
    sigma_max: float,
    eps: float = 0.01,
) -> jnp.ndarray:
    """Accuracy-dependent emission variance; sigma_min at s_t=1, grows as s_t→0."""
    s_clipped = jnp.clip(s_t, eps, 1.0)
    return sigma_min + (sigma_max - sigma_min) * (1.0 - s_clipped) / s_clipped


def compute_r1_average(batch: TrialBatch) -> np.ndarray:
    """
    Returns (n_images, D) mean of utterance_emb for rep_num == 1 trials.
    Raises ValueError if any image has no rep_num == 1 trials.
    """
    utt = np.array(batch.utterance_emb)
    rep = np.array(batch.rep_num)
    image_ids = np.array(batch.image_ids)

    mask = rep == 1
    result = np.zeros((batch.n_images, utt.shape[1]), dtype=np.float32)
    for img in range(batch.n_images):
        rows = utt[mask & (image_ids == img)]
        if len(rows) == 0:
            raise ValueError(
                f"Image {img} has no rep_num == 1 trials; cannot compute r1 average"
            )
        result[img] = rows.mean(axis=0)
    return result


def speaker_convention_model(
    batch: TrialBatch,
    s_t: jnp.ndarray,
    mu_i_init: np.ndarray,
) -> None:
    """
    NumPyro model for speaker-side conventions.
    Latents: sigma_game, sigma_min, sigma_max, mu_i, delta_gi.
    Convention: m_{g,i} = mu_i[image] + delta_gi[game, image].
    Likelihood: utterance_emb ~ N(m_{g,i_t}, sigma2_t * I).
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

    # delta_gi shape: (n_images, n_games, D) due to plate nesting; transpose to (n_games, n_images, D)
    delta_gi_t = jnp.moveaxis(delta_gi, 1, 0)
    m_gi = mu_i[None, :, :] + delta_gi_t  # (n_games, n_images, D)

    m_per_trial = m_gi[batch.game_ids, batch.image_ids]  # (N, D)
    sigma2_t = jnp.maximum(compute_sigma2_t(s_t, sigma_min, sigma_max), 1e-6)
    sigma_t = jnp.sqrt(sigma2_t)

    with numpyro.plate("trials", batch.utterance_emb.shape[0]):
        numpyro.sample(
            "utterance_emb",
            dist.Normal(m_per_trial, sigma_t[:, None]).to_event(1),
            obs=batch.utterance_emb,
        )


@dataclass
class ConventionFit:
    speaker_m_loc:   np.ndarray  # (n_games, n_images, D) posterior mean m_{g,i}
    speaker_m_scale: np.ndarray  # (n_games, n_images) posterior std (scalar per pair)
    listener_m_loc:  np.ndarray  # (n_games, n_images, D)
    listener_m_scale: np.ndarray  # (n_games, n_images)
    mu_i:            np.ndarray  # (n_images, D) fitted global prototypes
    sigma_game:      float
    sigma_min:       float
    sigma_max:       float


def _convention_guide(batch: TrialBatch, s_t: jnp.ndarray, mu_i_init: np.ndarray) -> None:
    """Mean-field guide for speaker_convention_model / listener_convention_model."""
    D = mu_i_init.shape[1]

    # Log-scale unconstrained params for positive scalars
    log_sigma_game_loc = numpyro.param("log_sigma_game_loc", jnp.log(jnp.array(0.5)))
    log_sigma_game_scale = numpyro.param(
        "log_sigma_game_scale", jnp.array(0.3), constraint=dist.constraints.positive
    )
    log_sigma_min_loc = numpyro.param("log_sigma_min_loc", jnp.log(jnp.array(0.3)))
    log_sigma_min_scale = numpyro.param(
        "log_sigma_min_scale", jnp.array(0.3), constraint=dist.constraints.positive
    )
    log_sigma_delta_loc = numpyro.param("log_sigma_delta_loc", jnp.log(jnp.array(0.7)))
    log_sigma_delta_scale = numpyro.param(
        "log_sigma_delta_scale", jnp.array(0.3), constraint=dist.constraints.positive
    )

    numpyro.sample(
        "sigma_game",
        dist.TransformedDistribution(
            dist.Normal(log_sigma_game_loc, log_sigma_game_scale),
            dist.transforms.ExpTransform(),
        ),
    )
    numpyro.sample(
        "sigma_min",
        dist.TransformedDistribution(
            dist.Normal(log_sigma_min_loc, log_sigma_min_scale),
            dist.transforms.ExpTransform(),
        ),
    )
    numpyro.sample(
        "sigma_delta",
        dist.TransformedDistribution(
            dist.Normal(log_sigma_delta_loc, log_sigma_delta_scale),
            dist.transforms.ExpTransform(),
        ),
    )

    mu_i_loc = numpyro.param("mu_i_loc", jnp.array(mu_i_init))
    mu_i_scale = numpyro.param(
        "mu_i_scale",
        jnp.ones((batch.n_images, D)) * 0.5,
        constraint=dist.constraints.positive,
    )
    numpyro.sample("mu_i", dist.Normal(mu_i_loc, mu_i_scale).to_event(1))

    delta_loc = numpyro.param(
        "delta_gi_loc", jnp.zeros((batch.n_images, batch.n_games, D))
    )
    delta_scale = numpyro.param(
        "delta_gi_scale",
        jnp.ones((batch.n_images, batch.n_games, D)) * 0.3,
        constraint=dist.constraints.positive,
    )
    with numpyro.plate("games", batch.n_games):
        with numpyro.plate("images_in_games", batch.n_images):
            numpyro.sample("delta_gi", dist.Normal(delta_loc, delta_scale).to_event(1))


def _extract_m_loc_scale(params: dict, batch: TrialBatch, D: int):
    """Extract per-(game, image) posterior mean and scale from SVI params."""
    mu_i_loc = np.array(params["mu_i_loc"])                        # (n_images, D)
    delta_loc = np.array(params["delta_gi_loc"])                   # (n_images, n_games, D)
    delta_scale = np.array(params["delta_gi_scale"])               # (n_images, n_games, D)

    m_loc = mu_i_loc[None, :, :] + delta_loc.transpose(1, 0, 2)   # (n_games, n_images, D)
    m_scale = delta_scale.transpose(1, 0, 2).mean(axis=-1)         # (n_games, n_images)
    return m_loc, m_scale


def save_convention_fit(fit: ConventionFit, path: str) -> None:
    np.savez(
        path,
        speaker_m_loc=fit.speaker_m_loc,
        speaker_m_scale=fit.speaker_m_scale,
        listener_m_loc=fit.listener_m_loc,
        listener_m_scale=fit.listener_m_scale,
        mu_i=fit.mu_i,
        sigma_game=np.array(fit.sigma_game),
        sigma_min=np.array(fit.sigma_min),
        sigma_max=np.array(fit.sigma_max),
    )


def load_convention_fit(path: str) -> ConventionFit:
    d = np.load(path)
    return ConventionFit(
        speaker_m_loc=d["speaker_m_loc"],
        speaker_m_scale=d["speaker_m_scale"],
        listener_m_loc=d["listener_m_loc"],
        listener_m_scale=d["listener_m_scale"],
        mu_i=d["mu_i"],
        sigma_game=float(d["sigma_game"]),
        sigma_min=float(d["sigma_min"]),
        sigma_max=float(d["sigma_max"]),
    )


def fit_convention(
    batch: TrialBatch,
    listener_fit: ListenerFit,
    mu_i_init: np.ndarray,
    n_steps: int = 8000,
    lr: float = 0.01,
    seed: int = 0,
    clip_norm: float | None = 10.0,
) -> ConventionFit:
    """
    Fits speaker_convention_model and listener_convention_model independently.
    Speaker s_t: L_0 success probs from listener_fit.
    Listener s_t: binary correctness (selected == target).
    """
    # Lazy import breaks the mutual dependency:
    # conventional_listener imports compute_sigma2_t from here;
    # this file needs listener_convention_model from there.
    from code.models.conventional_listener import listener_convention_model

    s_t_speaker = jnp.array(compute_success_probs(listener_fit, batch))
    s_t_listener = jnp.array(
        (np.array(batch.selected_idx) == np.array(batch.target_idx)).astype(np.float32)
    )

    D = mu_i_init.shape[1]

    speaker_result = run_svi(
        speaker_convention_model,
        _convention_guide,
        (batch, s_t_speaker, mu_i_init),
        n_steps=n_steps,
        lr=lr,
        seed=seed,
        clip_norm=clip_norm,
    )

    listener_result = run_svi(
        listener_convention_model,
        _convention_guide,
        (batch, s_t_listener, mu_i_init),
        n_steps=n_steps,
        lr=lr,
        seed=seed + 1,
        clip_norm=clip_norm,
    )

    sp = speaker_result.params
    lp = listener_result.params

    speaker_m_loc, speaker_m_scale = _extract_m_loc_scale(sp, batch, D)
    listener_m_loc, listener_m_scale = _extract_m_loc_scale(lp, batch, D)

    return ConventionFit(
        speaker_m_loc=speaker_m_loc,
        speaker_m_scale=speaker_m_scale,
        listener_m_loc=listener_m_loc,
        listener_m_scale=listener_m_scale,
        mu_i=np.array(sp["mu_i_loc"]),
        sigma_game=float(np.exp(sp["log_sigma_game_loc"])),
        sigma_min=float(np.exp(sp["log_sigma_min_loc"])),
        sigma_max=float(np.exp(sp["log_sigma_min_loc"]) + np.exp(sp["log_sigma_delta_loc"])),
    )
