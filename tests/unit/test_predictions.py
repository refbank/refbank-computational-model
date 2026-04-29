import numpy as np
import jax.numpy as jnp
import pytest

from code.prep_data.pipeline import build_trial_batch
from code.models.conventional_speaker import ConventionFit
from code.analysis.predictions import step_sizes_over_reps, semantic_displacement_after_error


def _make_fit(n_games, n_images, D, rng, sigma_min=0.3, sigma_max=1.0, sigma_game=0.5):
    mu_i = rng.standard_normal((n_images, D)).astype(np.float32)
    mu_i /= np.linalg.norm(mu_i, axis=1, keepdims=True)
    delta = rng.standard_normal((n_games, n_images, D)).astype(np.float32) * sigma_game
    m_loc = mu_i[None, :, :] + delta
    m_scale = np.ones((n_games, n_images), dtype=np.float32) * 0.1
    return ConventionFit(
        speaker_m_loc=m_loc,
        speaker_m_scale=m_scale,
        listener_m_loc=m_loc.copy(),
        listener_m_scale=m_scale.copy(),
        mu_i=mu_i,
        sigma_game=sigma_game,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )


def _make_batch(n_games, n_images, n_reps, D, rng, all_correct=True):
    n_total = n_games * n_images * n_reps
    utt = rng.standard_normal((n_total, D)).astype(np.float32)
    utt /= np.linalg.norm(utt, axis=1, keepdims=True)
    imgs = rng.standard_normal((n_total, 12, D)).astype(np.float32)
    imgs /= np.linalg.norm(imgs, axis=-1, keepdims=True)
    game_ids = np.repeat(np.arange(n_games), n_images * n_reps)
    image_ids = np.tile(np.repeat(np.arange(n_images), n_reps), n_games)
    rep_nums = np.tile(np.arange(1, n_reps + 1), n_games * n_images)
    target_idx = np.zeros(n_total, dtype=np.int32)
    selected_idx = target_idx.copy() if all_correct else np.ones(n_total, dtype=np.int32)
    return build_trial_batch(
        utterance_emb=jnp.array(utt),
        option_embs=jnp.array(imgs),
        target_idx=jnp.array(target_idx),
        selected_idx=jnp.array(selected_idx),
        listener_ids=jnp.zeros(n_total, dtype=jnp.int32),
        game_ids=jnp.array(game_ids, dtype=jnp.int32),
        image_ids=jnp.array(image_ids, dtype=jnp.int32),
        rep_num=jnp.array(rep_nums, dtype=jnp.int32),
        n_listeners=1,
        n_games=n_games,
        n_images=n_images,
    )


def test_step_sizes_over_reps_columns():
    rng = np.random.default_rng(0)
    fit = _make_fit(2, 2, 8, rng)
    batch = _make_batch(2, 2, 4, 8, rng)
    df = step_sizes_over_reps(fit, batch)
    assert set(df.columns) == {"game_id", "image_id", "rep_num", "step_size"}


def test_step_sizes_over_reps_decreasing_on_tight_emission():
    rng = np.random.default_rng(42)
    D = 8
    n_reps = 5

    # Tight emission: sigma_min is very small relative to sigma_game,
    # so each observation strongly updates the posterior → step sizes shrink fast
    fit = _make_fit(1, 1, D, rng, sigma_min=0.001, sigma_max=0.01, sigma_game=1.0)
    true_m = fit.mu_i[0]

    # Utterances tightly clustered around true_m; all trials correct (s_t = 1.0)
    noise = rng.standard_normal((n_reps, D)).astype(np.float32) * 0.01
    utt = true_m[None, :] + noise
    utt /= np.linalg.norm(utt, axis=1, keepdims=True)
    imgs = rng.standard_normal((n_reps, 12, D)).astype(np.float32)
    imgs /= np.linalg.norm(imgs, axis=-1, keepdims=True)

    batch = build_trial_batch(
        utterance_emb=jnp.array(utt),
        option_embs=jnp.array(imgs),
        target_idx=jnp.zeros(n_reps, dtype=jnp.int32),
        selected_idx=jnp.zeros(n_reps, dtype=jnp.int32),
        listener_ids=jnp.zeros(n_reps, dtype=jnp.int32),
        game_ids=jnp.zeros(n_reps, dtype=jnp.int32),
        image_ids=jnp.zeros(n_reps, dtype=jnp.int32),
        rep_num=jnp.arange(1, n_reps + 1, dtype=jnp.int32),
        n_listeners=1,
        n_games=1,
        n_images=1,
    )

    df = step_sizes_over_reps(fit, batch).sort_values("rep_num").reset_index(drop=True)
    # df has 4 rows (step from rep 1→2, 2→3, 3→4, 4→5)
    early = df.iloc[:2]["step_size"].mean()
    late = df.iloc[-2:]["step_size"].mean()
    assert late < early, f"Expected step sizes to decrease: early={early:.5f}, late={late:.5f}"


def test_step_sizes_over_reps_omits_pairs_with_fewer_than_2_reps():
    rng = np.random.default_rng(0)
    D = 8
    # (game=0, image=0): 3 reps; (game=0, image=1): 1 rep only
    utt = rng.standard_normal((4, D)).astype(np.float32)
    utt /= np.linalg.norm(utt, axis=1, keepdims=True)
    imgs = rng.standard_normal((4, 12, D)).astype(np.float32)
    imgs /= np.linalg.norm(imgs, axis=-1, keepdims=True)

    batch = build_trial_batch(
        utterance_emb=jnp.array(utt),
        option_embs=jnp.array(imgs),
        target_idx=jnp.zeros(4, dtype=jnp.int32),
        selected_idx=jnp.zeros(4, dtype=jnp.int32),
        listener_ids=jnp.zeros(4, dtype=jnp.int32),
        game_ids=jnp.zeros(4, dtype=jnp.int32),
        image_ids=jnp.array([0, 0, 0, 1], dtype=jnp.int32),
        rep_num=jnp.array([1, 2, 3, 1], dtype=jnp.int32),
        n_listeners=1,
        n_games=1,
        n_images=2,
    )

    fit = _make_fit(1, 2, D, rng)
    df = step_sizes_over_reps(fit, batch)

    assert len(df[(df["game_id"] == 0) & (df["image_id"] == 1)]) == 0, (
        "Pair with only 1 rep should be omitted"
    )
    assert len(df[(df["game_id"] == 0) & (df["image_id"] == 0)]) == 2, (
        "3 reps should produce 2 step sizes"
    )


def test_semantic_displacement_columns():
    rng = np.random.default_rng(0)
    fit = _make_fit(2, 2, 8, rng)
    batch = _make_batch(2, 2, 4, 8, rng)
    df = semantic_displacement_after_error(fit, batch)
    assert set(df.columns) == {"game_id", "image_id", "rep_num", "displacement", "prev_success"}


def test_semantic_displacement_direction():
    D = 8
    n_reps = 5

    # Construct utterances: small jump after success, large jump after failure.
    # trial_success: [True, False, True, False, True]
    # Displacements at reps 2..5; prev_success = success of previous trial.
    # After success (reps 2, 4): tiny perturbation  → small displacement
    # After failure (reps 3, 5): orthogonal direction → large displacement
    u = np.zeros((n_reps, D), dtype=np.float32)
    u[0] = np.eye(D)[0]  # unit vector along axis 0
    u[1] = u[0] + np.eye(D)[1] * 0.01  # tiny perturbation (success after trial 0)
    u[1] /= np.linalg.norm(u[1])
    u[2] = np.eye(D)[1]  # orthogonal to u[0] (large jump after trial 1 failure)
    u[3] = u[2] + np.eye(D)[2] * 0.01  # tiny perturbation (success after trial 2)
    u[3] /= np.linalg.norm(u[3])
    u[4] = np.eye(D)[2]  # orthogonal (large jump after trial 3 failure)

    trial_success = np.array([True, False, True, False, True])
    target_idx = np.zeros(n_reps, dtype=np.int32)
    selected_idx = np.where(trial_success, target_idx, target_idx + 1)

    rng = np.random.default_rng(0)
    imgs = rng.standard_normal((n_reps, 12, D)).astype(np.float32)
    imgs /= np.linalg.norm(imgs, axis=-1, keepdims=True)

    batch = build_trial_batch(
        utterance_emb=jnp.array(u),
        option_embs=jnp.array(imgs),
        target_idx=jnp.array(target_idx),
        selected_idx=jnp.array(selected_idx),
        listener_ids=jnp.zeros(n_reps, dtype=jnp.int32),
        game_ids=jnp.zeros(n_reps, dtype=jnp.int32),
        image_ids=jnp.zeros(n_reps, dtype=jnp.int32),
        rep_num=jnp.arange(1, n_reps + 1, dtype=jnp.int32),
        n_listeners=1,
        n_games=1,
        n_images=1,
    )

    fit = _make_fit(1, 1, D, rng)
    df = semantic_displacement_after_error(fit, batch)

    after_failure = df[~df["prev_success"]]["displacement"].mean()
    after_success = df[df["prev_success"]]["displacement"].mean()
    assert after_failure > after_success, (
        f"Expected larger displacement after failure: {after_failure:.4f} vs {after_success:.4f}"
    )
