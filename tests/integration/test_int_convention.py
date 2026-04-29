import numpy as np
import jax.numpy as jnp
import pytest

from code.prep_data.pipeline import build_trial_batch
from code.models.literal_listener import ListenerFit
from code.models.conventional_speaker import compute_r1_average, fit_convention
from code.analysis.predictions import step_sizes_over_reps


@pytest.mark.slow
def test_step_sizes_decrease():
    """INT-2: convention posterior concentrates → step sizes shrink over reps."""
    rng = np.random.default_rng(0)
    n_games, n_images, n_reps, D = 6, 3, 6, 8
    n_listeners = 1
    n_total = n_games * n_images * n_reps

    # True convention points: random unit vectors per (game, image)
    true_m = rng.standard_normal((n_games, n_images, D)).astype(np.float32)
    true_m /= np.linalg.norm(true_m, axis=-1, keepdims=True)

    utterance_embs = np.zeros((n_total, D), dtype=np.float32)
    option_embs_arr = np.zeros((n_total, 12, D), dtype=np.float32)
    target_idxs = np.zeros(n_total, dtype=np.int32)
    selected_idxs = np.zeros(n_total, dtype=np.int32)
    game_ids_arr = np.zeros(n_total, dtype=np.int32)
    image_ids_arr = np.zeros(n_total, dtype=np.int32)
    rep_nums = np.zeros(n_total, dtype=np.int32)

    # 12 shared image embeddings (random, L2-normalized)
    shared_imgs = rng.standard_normal((12, D)).astype(np.float32)
    shared_imgs /= np.linalg.norm(shared_imgs, axis=1, keepdims=True)

    idx = 0
    for g in range(n_games):
        for i in range(n_images):
            for r in range(n_reps):
                noise = rng.standard_normal(D).astype(np.float32) * np.sqrt(0.01)
                utt = true_m[g, i] + noise
                utt /= np.linalg.norm(utt)
                utterance_embs[idx] = utt
                option_embs_arr[idx] = shared_imgs
                target_idxs[idx] = i % 12
                selected_idxs[idx] = i % 12  # all "correct"
                game_ids_arr[idx] = g
                image_ids_arr[idx] = i
                rep_nums[idx] = r + 1
                idx += 1

    batch = build_trial_batch(
        utterance_emb=jnp.array(utterance_embs),
        option_embs=jnp.array(option_embs_arr),
        target_idx=jnp.array(target_idxs),
        selected_idx=jnp.array(selected_idxs),
        listener_ids=jnp.zeros(n_total, dtype=jnp.int32),
        game_ids=jnp.array(game_ids_arr),
        image_ids=jnp.array(image_ids_arr),
        rep_num=jnp.array(rep_nums),
        n_listeners=n_listeners,
        n_games=n_games,
        n_images=n_images,
    )

    # Listener fit: fixed high beta (s_t ≈ 0.9 via high sharpness)
    listener_fit = ListenerFit(
        beta_loc=np.array([np.log(10.0)]),
        beta_scale=np.array([0.01]),
        mu_beta=np.log(10.0),
        sigma_beta=0.5,
    )

    mu_i_init = compute_r1_average(batch)
    fit = fit_convention(batch, listener_fit, mu_i_init, n_steps=8000, seed=0)

    sizes_df = step_sizes_over_reps(fit, batch, role="speaker")

    pairs = sizes_df.groupby(["game_id", "image_id"])
    n_pairs = 0
    n_decreasing = 0
    for (g, i), group in pairs:
        group = group.sort_values("rep_num")
        if len(group) < 4:
            continue
        early = group.iloc[:2]["step_size"].mean()
        late = group.iloc[-2:]["step_size"].mean()
        n_pairs += 1
        if late < early:
            n_decreasing += 1

    assert n_pairs > 0, "No (game, image) pairs with enough reps"
    frac = n_decreasing / n_pairs
    assert frac >= 0.8, (
        f"Expected ≥80% of pairs to show decreasing step sizes, got {frac:.1%} ({n_decreasing}/{n_pairs})"
    )
