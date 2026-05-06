import re

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


# ---------------------------------------------------------------------------
# Helpers for build_trial_batch_from_df tests
# ---------------------------------------------------------------------------

def _make_df(n_trials=12, n_games=2, n_listeners=2, n_images=3, D=8, rng=None):
    """Build a minimal valid DataFrame and matching embedding dicts."""
    if rng is None:
        rng = np.random.default_rng(0)

    # Need at least 12 distinct images to fill option_sets
    n_pool = max(12, n_images + 9)
    pool_ids = [f"img_{i}" for i in range(n_pool)]
    target_ids = pool_ids[:n_images]

    rows = []
    for t in range(n_trials):
        target = target_ids[t % n_images]
        distractors = [img for img in pool_ids if img != target][:11]
        option_set = [target] + distractors
        rows.append({
            "game_id": f"game_{t % n_games}",
            "trial_num": t,
            "rep_num": 1 + t // (n_games * n_images),
            "image_id": target,
            "option_set": option_set,
            "utterance": f"utt_{t}",
            "selected_image_id": target,
            "listener_id": f"listener_{t % n_listeners}",
        })

    df = pd.DataFrame(rows)

    def _unit(shape):
        v = rng.standard_normal(shape).astype(np.float32)
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    image_embs = {img: _unit(D) for img in pool_ids}
    utt_embs = {f"utt_{t}": _unit(D) for t in range(n_trials)}

    return df, image_embs, utt_embs


# ---------------------------------------------------------------------------
# build_trial_batch_from_df tests
# ---------------------------------------------------------------------------

def test_build_trial_batch_from_df_shapes():
    N, D = 20, 8
    df, image_embs, utt_embs = _make_df(n_trials=N, D=D)
    batch = build_trial_batch_from_df(df, image_embs, utt_embs)
    assert batch.utterance_emb.shape == (N, D)
    assert batch.option_embs.shape == (N, 12, D)
    assert batch.target_idx.shape == (N,)
    assert batch.selected_idx.shape == (N,)
    assert batch.listener_ids.shape == (N,)
    assert batch.game_ids.shape == (N,)
    assert batch.image_ids.shape == (N,)
    assert batch.rep_num.shape == (N,)


def test_build_trial_batch_from_df_raises_on_wrong_option_count():
    df, image_embs, utt_embs = _make_df(n_trials=5)
    df = df.copy()
    df.at[0, "option_set"] = df.at[0, "option_set"][:11]
    with pytest.raises(ValueError, match="12"):
        build_trial_batch_from_df(df, image_embs, utt_embs)


def test_build_trial_batch_from_df_raises_on_missing_image_embedding():
    df, image_embs, utt_embs = _make_df(n_trials=5)
    missing_id = df.iloc[0]["option_set"][1]
    del image_embs[missing_id]
    with pytest.raises(KeyError, match=re.escape(missing_id)):
        build_trial_batch_from_df(df, image_embs, utt_embs)


def test_build_trial_batch_from_df_raises_on_missing_utterance_embedding():
    df, image_embs, utt_embs = _make_df(n_trials=5)
    missing_utt = df.iloc[0]["utterance"]
    del utt_embs[missing_utt]
    with pytest.raises(KeyError, match=re.escape(missing_utt)):
        build_trial_batch_from_df(df, image_embs, utt_embs)


def test_build_trial_batch_from_df_ids_are_contiguous():
    df, image_embs, utt_embs = _make_df(n_trials=24, n_games=3, n_listeners=2, n_images=4)
    batch = build_trial_batch_from_df(df, image_embs, utt_embs)
    assert set(np.array(batch.game_ids)) == set(range(batch.n_games))
    assert set(np.array(batch.listener_ids)) == set(range(batch.n_listeners))
    assert set(np.array(batch.image_ids)) == set(range(batch.n_images))


def test_build_trial_batch_from_df_option_set_as_numpy_array():
    """Parquet round-trips option_set as np.ndarray; build_trial_batch_from_df must handle it."""
    df, image_embs, utt_embs = _make_df(n_trials=12, D=8)
    df = df.copy()
    df["option_set"] = df["option_set"].apply(np.array)
    batch = build_trial_batch_from_df(df, image_embs, utt_embs)
    assert batch.utterance_emb.shape[0] == 12


def test_build_trial_batch_from_df_has_option_image_ids():
    """option_image_ids must have shape (N, 12) with int dtype."""
    N, D = 20, 8
    df, image_embs, utt_embs = _make_df(n_trials=N, D=D)
    batch = build_trial_batch_from_df(df, image_embs, utt_embs)
    assert hasattr(batch, "option_image_ids"), "TrialBatch missing option_image_ids field"
    assert batch.option_image_ids.shape == (N, 12)
    assert jnp.issubdtype(batch.option_image_ids.dtype, jnp.integer)


def test_build_trial_batch_from_df_has_all_image_embs():
    """all_image_embs must have shape (n_all_images, D) covering all option images."""
    N, D = 20, 8
    df, image_embs, utt_embs = _make_df(n_trials=N, D=D)
    batch = build_trial_batch_from_df(df, image_embs, utt_embs)
    assert hasattr(batch, "all_image_embs"), "TrialBatch missing all_image_embs field"
    all_option_ids = {img for opts in df["option_set"] for img in opts}
    assert batch.all_image_embs.shape == (len(all_option_ids), D)


def test_build_trial_batch_from_df_option_image_ids_consistent_with_all_image_embs():
    """all_image_embs[option_image_ids[n, j]] must equal option_embs[n, j] for all n, j."""
    N, D = 20, 8
    df, image_embs, utt_embs = _make_df(n_trials=N, D=D)
    batch = build_trial_batch_from_df(df, image_embs, utt_embs)
    reconstructed = batch.all_image_embs[batch.option_image_ids]  # (N, 12, D)
    assert reconstructed.shape == batch.option_embs.shape
    assert np.allclose(np.array(reconstructed), np.array(batch.option_embs), atol=1e-6), (
        "all_image_embs[option_image_ids] must reproduce option_embs exactly"
    )
