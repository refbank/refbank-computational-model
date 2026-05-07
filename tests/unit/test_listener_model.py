import numpy as np
import jax.numpy as jnp
import pytest
from unittest.mock import patch

from code.prep_data.pipeline import TrialBatch, build_trial_batch
from code.models.literal_listener import (
    literal_listener_model,
    literal_listener_guide,
    compute_success_probs,
    fit_listener,
    save_listener_fit,
    load_listener_fit,
    ListenerFit,
)


def _make_batch(N=10, D=8, n_listeners=2, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    utt = rng.standard_normal((N, D)).astype(np.float32)
    utt /= np.linalg.norm(utt, axis=1, keepdims=True)
    imgs = rng.standard_normal((N, 12, D)).astype(np.float32)
    imgs /= np.linalg.norm(imgs, axis=-1, keepdims=True)
    return build_trial_batch(
        utterance_emb=jnp.array(utt),
        option_embs=jnp.array(imgs),
        target_idx=jnp.zeros(N, dtype=jnp.int32),
        selected_idx=jnp.zeros(N, dtype=jnp.int32),
        listener_ids=jnp.array(np.arange(N) % n_listeners, dtype=jnp.int32),
        game_ids=jnp.zeros(N, dtype=jnp.int32),
        image_ids=jnp.zeros(N, dtype=jnp.int32),
        rep_num=jnp.zeros(N, dtype=jnp.int32),
        n_listeners=n_listeners,
        n_games=1,
        n_images=1,
    )


def test_listener_model_traces_without_error():
    import numpyro
    batch = _make_batch()
    # Tracing the model means running it once to check for errors
    with numpyro.handlers.seed(rng_seed=0):
        literal_listener_model(batch)


def test_compute_success_probs_shape():
    batch = _make_batch(N=20)
    fit = ListenerFit(
        beta_loc=np.zeros(2),
        beta_scale=np.ones(2),
        mu_beta=0.0,
        sigma_beta=1.0,
    )
    probs = compute_success_probs(fit, batch)
    assert probs.shape == (20,)


def test_compute_success_probs_values_in_range():
    batch = _make_batch(N=20)
    fit = ListenerFit(
        beta_loc=np.zeros(2),
        beta_scale=np.ones(2),
        mu_beta=0.0,
        sigma_beta=1.0,
    )
    probs = compute_success_probs(fit, batch)
    assert np.all(probs > 0) and np.all(probs < 1)


def test_compute_success_probs_high_for_clear_target():
    """Utterance identical to target; distractors orthogonal to target."""
    rng = np.random.default_rng(7)
    N, D = 10, 64
    target_vec = np.zeros(D, dtype=np.float32)
    target_vec[0] = 1.0  # unit vector along axis 0

    # Distractors: random vectors in the D-1 dimensional orthogonal complement
    distractor_vecs = rng.standard_normal((11, D)).astype(np.float32)
    distractor_vecs[:, 0] = 0.0  # zero out component along target axis
    distractor_vecs /= np.linalg.norm(distractor_vecs, axis=1, keepdims=True)

    imgs = np.vstack([target_vec[np.newaxis], distractor_vecs])  # (12, D)
    target_idx = 0
    utt = np.tile(imgs[target_idx], (N, 1))
    option_embs = np.tile(imgs[np.newaxis], (N, 1, 1))

    batch = build_trial_batch(
        utterance_emb=jnp.array(utt),
        option_embs=jnp.array(option_embs),
        target_idx=jnp.zeros(N, dtype=jnp.int32),
        selected_idx=jnp.zeros(N, dtype=jnp.int32),
        listener_ids=jnp.zeros(N, dtype=jnp.int32),
        game_ids=jnp.zeros(N, dtype=jnp.int32),
        image_ids=jnp.zeros(N, dtype=jnp.int32),
        rep_num=jnp.zeros(N, dtype=jnp.int32),
        n_listeners=1,
        n_games=1,
        n_images=1,
    )
    # High beta → sharp discrimination
    fit = ListenerFit(
        beta_loc=np.array([np.log(5.0)]),  # beta ≈ 5
        beta_scale=np.array([0.01]),
        mu_beta=np.log(5.0),
        sigma_beta=1.0,
    )
    probs = compute_success_probs(fit, batch)
    assert np.all(probs > 0.9), f"Expected s_t > 0.9 for clear target, got min {probs.min():.3f}"


def test_fit_listener_raises_on_nan_loss():
    batch = _make_batch(N=20)
    with patch("code.models.literal_listener.run_svi",
               side_effect=RuntimeError("NaN loss at step 150")):
        with pytest.raises(RuntimeError, match="NaN loss"):
            fit_listener(batch, n_steps=200)


def test_save_load_listener_fit_roundtrip(tmp_path):
    fit = ListenerFit(
        beta_loc=np.array([1.0, 2.0, 3.0]),
        beta_scale=np.array([0.1, 0.2, 0.3]),
        mu_beta=0.5,
        sigma_beta=1.2,
    )
    path = str(tmp_path / "listener_fit.npz")
    save_listener_fit(fit, path)
    loaded = load_listener_fit(path)
    np.testing.assert_array_equal(loaded.beta_loc, fit.beta_loc)
    np.testing.assert_array_equal(loaded.beta_scale, fit.beta_scale)
    assert loaded.mu_beta == fit.mu_beta
    assert loaded.sigma_beta == fit.sigma_beta


