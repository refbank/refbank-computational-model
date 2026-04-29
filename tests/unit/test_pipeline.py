import numpy as np
import jax.numpy as jnp
import pandas as pd
import pytest

from code.prep_data.pipeline import TrialBatch, build_trial_batch, build_trial_batch_from_df


def _make_embeddings(n_trials, n_images, D, rng):
    """Helper: create valid L2-normalized embeddings for tests."""
    utt = rng.standard_normal((n_trials, D)).astype(np.float32)
    utt /= np.linalg.norm(utt, axis=1, keepdims=True)
    imgs = rng.standard_normal((n_trials, n_images, D)).astype(np.float32)
    imgs /= np.linalg.norm(imgs, axis=-1, keepdims=True)
    return utt, imgs


def test_build_trial_batch_shapes():
    rng = np.random.default_rng(0)
    N, D = 50, 16
    utt, imgs = _make_embeddings(N, 12, D, rng)
    batch = build_trial_batch(
        utterance_emb=jnp.array(utt),
        option_embs=jnp.array(imgs),
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
    assert batch.utterance_emb.shape == (N, D)
    assert batch.option_embs.shape == (N, 12, D)
    assert batch.target_idx.shape == (N,)
    assert batch.selected_idx.shape == (N,)
    assert batch.listener_ids.shape == (N,)
    assert batch.game_ids.shape == (N,)
    assert batch.image_ids.shape == (N,)
    assert batch.rep_num.shape == (N,)


def test_build_trial_batch_raises_on_wrong_option_count():
    rng = np.random.default_rng(0)
    N, D = 5, 8
    utt, imgs = _make_embeddings(N, 11, D, rng)
    with pytest.raises(ValueError, match="12"):
        build_trial_batch(
            utterance_emb=jnp.array(utt),
            option_embs=jnp.array(imgs),
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


def test_build_trial_batch_stores_metadata():
    rng = np.random.default_rng(0)
    N, D = 10, 8
    utt, imgs = _make_embeddings(N, 12, D, rng)
    batch = build_trial_batch(
        utterance_emb=jnp.array(utt),
        option_embs=jnp.array(imgs),
        target_idx=jnp.zeros(N, dtype=jnp.int32),
        selected_idx=jnp.zeros(N, dtype=jnp.int32),
        listener_ids=jnp.zeros(N, dtype=jnp.int32),
        game_ids=jnp.zeros(N, dtype=jnp.int32),
        image_ids=jnp.zeros(N, dtype=jnp.int32),
        rep_num=jnp.zeros(N, dtype=jnp.int32),
        n_listeners=3,
        n_games=4,
        n_images=5,
    )
    assert batch.n_listeners == 3
    assert batch.n_games == 4
    assert batch.n_images == 5
