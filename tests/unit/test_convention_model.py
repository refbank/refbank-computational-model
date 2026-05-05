import numpy as np
import jax.numpy as jnp
import numpyro
import pytest

from code.prep_data.pipeline import TrialBatch, build_trial_batch
from code.models.conventional_speaker import (
    compute_sigma2_t,
    compute_r1_average,
    speaker_convention_model,
    ConventionFit,
    fit_convention,
    save_convention_fit,
    load_convention_fit,
)
from code.models.conventional_listener import listener_convention_model
from code.models.literal_listener import ListenerFit


def _make_batch(n_games=2, n_images=2, n_reps=3, D=8, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    n_total = n_games * n_images * n_reps
    utt = rng.standard_normal((n_total, D)).astype(np.float32)
    utt /= np.linalg.norm(utt, axis=1, keepdims=True)
    imgs = rng.standard_normal((n_total, 12, D)).astype(np.float32)
    imgs /= np.linalg.norm(imgs, axis=-1, keepdims=True)

    game_ids = np.repeat(np.arange(n_games), n_images * n_reps)
    image_ids = np.tile(np.repeat(np.arange(n_images), n_reps), n_games)
    rep_nums = np.tile(np.arange(1, n_reps + 1), n_games * n_images)
    target_idx = (image_ids % 12).astype(np.int32)

    return build_trial_batch(
        utterance_emb=jnp.array(utt),
        option_embs=jnp.array(imgs),
        target_idx=jnp.array(target_idx),
        selected_idx=jnp.array(target_idx),
        listener_ids=jnp.zeros(n_total, dtype=jnp.int32),
        game_ids=jnp.array(game_ids, dtype=jnp.int32),
        image_ids=jnp.array(image_ids, dtype=jnp.int32),
        rep_num=jnp.array(rep_nums, dtype=jnp.int32),
        n_listeners=1,
        n_games=n_games,
        n_images=n_images,
    )


def test_compute_sigma2_t_at_success():
    s_t = jnp.array([1.0])
    result = compute_sigma2_t(s_t, sigma_min=0.1, sigma_max=2.0)
    np.testing.assert_allclose(result, [0.1], atol=1e-5)


def test_compute_sigma2_t_at_failure():
    s_t = jnp.array([0.0])
    result = compute_sigma2_t(s_t, sigma_min=0.1, sigma_max=2.0, eps=0.01)
    expected = 0.1 + 1.9 * (1 / 0.01 - 1)
    np.testing.assert_allclose(result, [expected], rtol=1e-5)


def test_compute_sigma2_t_monotone_decreasing_in_s():
    s_t = jnp.array([0.1, 0.5, 0.9])
    result = compute_sigma2_t(s_t, sigma_min=0.1, sigma_max=2.0)
    assert result[0] > result[1] > result[2], (
        f"Expected monotone decreasing: {result}"
    )


def test_compute_r1_average_shape():
    batch = _make_batch(n_games=2, n_images=3, n_reps=4, D=8)
    result = compute_r1_average(batch)
    assert result.shape == (3, 8)


def test_compute_r1_average_uses_only_rep1():
    rng = np.random.default_rng(0)
    D = 8
    n_total = 6  # 2 images × 3 reps
    utt = np.zeros((n_total, D), dtype=np.float32)
    # rep_num=1 rows → all zeros; rep_num>1 → all ones
    utt[0] = 0.0  # image 0, rep 1 — stays zero
    utt[1] = 1.0 / np.sqrt(D)  # image 0, rep 2
    utt[2] = 1.0 / np.sqrt(D)  # image 0, rep 3
    utt[3] = 0.0  # image 1, rep 1 — stays zero
    utt[4] = 1.0 / np.sqrt(D)  # image 1, rep 2
    utt[5] = 1.0 / np.sqrt(D)  # image 1, rep 3

    imgs = rng.standard_normal((n_total, 12, D)).astype(np.float32)
    imgs /= np.linalg.norm(imgs, axis=-1, keepdims=True)

    batch = build_trial_batch(
        utterance_emb=jnp.array(utt),
        option_embs=jnp.array(imgs),
        target_idx=jnp.zeros(n_total, dtype=jnp.int32),
        selected_idx=jnp.zeros(n_total, dtype=jnp.int32),
        listener_ids=jnp.zeros(n_total, dtype=jnp.int32),
        game_ids=jnp.zeros(n_total, dtype=jnp.int32),
        image_ids=jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.int32),
        rep_num=jnp.array([1, 2, 3, 1, 2, 3], dtype=jnp.int32),
        n_listeners=1,
        n_games=1,
        n_images=2,
    )
    result = compute_r1_average(batch)
    np.testing.assert_allclose(result, np.zeros((2, D)), atol=1e-6)


def test_compute_r1_average_raises_if_image_missing_from_rep1():
    rng = np.random.default_rng(0)
    D = 8
    # Only image 0 has rep_num=1; image 1 has only rep_num=2,3
    n_total = 4
    utt = rng.standard_normal((n_total, D)).astype(np.float32)
    utt /= np.linalg.norm(utt, axis=1, keepdims=True)
    imgs = rng.standard_normal((n_total, 12, D)).astype(np.float32)
    imgs /= np.linalg.norm(imgs, axis=-1, keepdims=True)
    batch = build_trial_batch(
        utterance_emb=jnp.array(utt),
        option_embs=jnp.array(imgs),
        target_idx=jnp.zeros(n_total, dtype=jnp.int32),
        selected_idx=jnp.zeros(n_total, dtype=jnp.int32),
        listener_ids=jnp.zeros(n_total, dtype=jnp.int32),
        game_ids=jnp.zeros(n_total, dtype=jnp.int32),
        image_ids=jnp.array([0, 0, 1, 1], dtype=jnp.int32),
        rep_num=jnp.array([1, 2, 2, 3], dtype=jnp.int32),  # image 1: no rep 1
        n_listeners=1,
        n_games=1,
        n_images=2,
    )
    with pytest.raises(ValueError, match="1"):
        compute_r1_average(batch)


def test_speaker_convention_model_traces():
    batch = _make_batch(n_games=2, n_images=2, n_reps=3, D=8)
    mu_i_init = np.zeros((2, 8), dtype=np.float32)
    s_t = jnp.ones(batch.utterance_emb.shape[0]) * 0.9
    with numpyro.handlers.seed(rng_seed=0):
        speaker_convention_model(batch, s_t, mu_i_init)


def test_listener_convention_model_traces():
    batch = _make_batch(n_games=2, n_images=2, n_reps=3, D=8)
    mu_i_init = np.zeros((2, 8), dtype=np.float32)
    s_t_binary = jnp.ones(batch.utterance_emb.shape[0])
    with numpyro.handlers.seed(rng_seed=0):
        listener_convention_model(batch, s_t_binary, mu_i_init)


def test_speaker_convention_model_sigma_max_exceeds_sigma_min():
    """Model must structurally guarantee sigma_max > sigma_min at every sample."""
    batch = _make_batch(n_games=2, n_images=2, n_reps=3, D=8)
    mu_i_init = np.zeros((2, 8), dtype=np.float32)
    s_t = jnp.ones(batch.utterance_emb.shape[0]) * 0.9

    for seed in range(5):
        with numpyro.handlers.seed(rng_seed=seed):
            trace = numpyro.handlers.trace(speaker_convention_model).get_trace(
                batch, s_t, mu_i_init
            )
        sigma_min = float(trace["sigma_min"]["value"])
        sigma_max = float(trace["sigma_max"]["value"])
        assert sigma_max > sigma_min, (
            f"seed={seed}: sigma_max={sigma_max:.4f} <= sigma_min={sigma_min:.4f}"
        )


def test_fitted_sigma_max_exceeds_sigma_min():
    """After fitting, the constraint sigma_max > sigma_min must hold."""
    batch = _make_batch(n_games=3, n_images=2, n_reps=4, D=8)
    listener_fit = ListenerFit(
        beta_loc=np.array([0.0]),
        beta_scale=np.array([1.0]),
        mu_beta=0.0,
        sigma_beta=1.0,
    )
    mu_i_init = compute_r1_average(batch)
    fit = fit_convention(batch, listener_fit, mu_i_init, n_steps=200, seed=0)
    assert fit.sigma_max > fit.sigma_min, (
        f"sigma_max={fit.sigma_max:.4f} <= sigma_min={fit.sigma_min:.4f}"
    )


def test_save_load_convention_fit_roundtrip(tmp_path):
    fit = ConventionFit(
        speaker_m_loc=np.zeros((2, 3, 4), dtype=np.float32),
        speaker_m_scale=np.ones((2, 3), dtype=np.float32),
        listener_m_loc=np.ones((2, 3, 4), dtype=np.float32) * 0.5,
        listener_m_scale=np.ones((2, 3), dtype=np.float32) * 0.2,
        mu_i=np.zeros((3, 4), dtype=np.float32),
        sigma_game=0.5,
        sigma_min=0.1,
        sigma_max=2.0,
    )
    path = str(tmp_path / "convention_fit.npz")
    save_convention_fit(fit, path)
    loaded = load_convention_fit(path)
    np.testing.assert_array_equal(loaded.speaker_m_loc, fit.speaker_m_loc)
    np.testing.assert_array_equal(loaded.listener_m_loc, fit.listener_m_loc)
    np.testing.assert_array_equal(loaded.mu_i, fit.mu_i)
    assert loaded.sigma_game == fit.sigma_game
    assert loaded.sigma_min == fit.sigma_min
    assert loaded.sigma_max == fit.sigma_max


def test_fit_convention_output_shapes():
    batch = _make_batch(n_games=3, n_images=2, n_reps=4, D=8)
    listener_fit = ListenerFit(
        beta_loc=np.array([0.0]),
        beta_scale=np.array([1.0]),
        mu_beta=0.0,
        sigma_beta=1.0,
    )
    mu_i_init = compute_r1_average(batch)
    fit = fit_convention(batch, listener_fit, mu_i_init, n_steps=200, seed=0)
    assert fit.speaker_m_loc.shape == (3, 2, 8), fit.speaker_m_loc.shape
    assert fit.listener_m_loc.shape == (3, 2, 8), fit.listener_m_loc.shape
