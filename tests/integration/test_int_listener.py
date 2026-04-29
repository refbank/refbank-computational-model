import numpy as np
import pytest

from code.prep_data.pipeline import build_trial_batch
from code.models.literal_listener import fit_listener, compute_success_probs


@pytest.mark.slow
def test_listener_beta_recovery():
    """INT-1: fit listener model on synthetic data; recover two known beta values."""
    rng = np.random.default_rng(42)
    D, N, n_images = 16, 200, 12
    true_betas = np.array([2.0, 5.0])

    # Image embeddings: random unit vectors
    img_embs = rng.standard_normal((n_images, D)).astype(np.float32)
    img_embs /= np.linalg.norm(img_embs, axis=1, keepdims=True)

    # For each trial: target gets cos≈0.9, distractors get cos≈0.1-0.3
    utterance_embs = np.zeros((N, D), dtype=np.float32)
    target_idxs = np.zeros(N, dtype=np.int32)
    selected_idxs = np.zeros(N, dtype=np.int32)
    listener_ids = np.zeros(N, dtype=np.int32)
    option_embs = np.zeros((N, 12, D), dtype=np.float32)

    for n in range(N):
        listener = n % 2
        listener_ids[n] = listener
        target = rng.integers(n_images)
        target_idxs[n] = target

        # Construct utterance: blend of target embedding + noise, giving cos≈0.9
        noise = rng.standard_normal(D).astype(np.float32)
        noise -= noise.dot(img_embs[target]) * img_embs[target]
        noise /= np.linalg.norm(noise)
        utt = 0.9 * img_embs[target] + np.sqrt(1 - 0.9**2) * noise
        utt /= np.linalg.norm(utt)
        utterance_embs[n] = utt

        option_embs[n] = img_embs
        cos_sims = utt @ img_embs.T
        logits = true_betas[listener] * cos_sims
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        selected_idxs[n] = rng.choice(n_images, p=probs)

    import jax.numpy as jnp

    batch = build_trial_batch(
        utterance_emb=jnp.array(utterance_embs),
        option_embs=jnp.array(option_embs),
        target_idx=jnp.array(target_idxs),
        selected_idx=jnp.array(selected_idxs),
        listener_ids=jnp.array(listener_ids),
        game_ids=jnp.zeros(N, dtype=jnp.int32),
        image_ids=jnp.array(target_idxs),
        rep_num=jnp.zeros(N, dtype=jnp.int32),
        n_listeners=2,
        n_games=1,
        n_images=n_images,
    )

    fit = fit_listener(batch, n_steps=5000, seed=0)

    fitted_betas = np.exp(fit.beta_loc)  # posterior means in beta space
    assert fitted_betas[0] < fitted_betas[1], (
        f"Expected beta order preserved: {fitted_betas[0]:.2f} < {fitted_betas[1]:.2f}"
    )
    for i, (fitted, true) in enumerate(zip(fitted_betas, true_betas)):
        assert abs(fitted - true) < 1.5, (
            f"listener {i}: fitted beta {fitted:.2f}, true {true:.2f}, diff {abs(fitted-true):.2f}"
        )
