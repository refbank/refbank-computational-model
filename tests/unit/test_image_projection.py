import numpy as np
import jax.numpy as jnp
import pytest

from code.prep_data.pipeline import build_trial_batch
from code.models.image_projection import (
    apply_projection,
    fit_image_projection,
    ImageProjection,
    save_image_projection,
    load_image_projection,
    project_batch,
)


def _make_batch(N=30, D=8, n_all_images=12, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    utt = rng.standard_normal((N, D)).astype(np.float32)
    utt /= np.linalg.norm(utt, axis=1, keepdims=True)
    all_img_embs = rng.standard_normal((n_all_images, D)).astype(np.float32)
    all_img_embs /= np.linalg.norm(all_img_embs, axis=1, keepdims=True)
    option_image_ids = np.tile(np.arange(12, dtype=np.int32), (N, 1))
    imgs = all_img_embs[option_image_ids]
    return build_trial_batch(
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
        option_image_ids=jnp.array(option_image_ids),
        all_image_embs=jnp.array(all_img_embs),
    )


# ---------------------------------------------------------------------------
# apply_projection
# ---------------------------------------------------------------------------

def test_apply_projection_output_shape():
    """Output shape matches input."""
    rng = np.random.default_rng(0)
    D, r, n = 8, 4, 12
    A = rng.standard_normal((D, r)).astype(np.float32)
    B = rng.standard_normal((r, D)).astype(np.float32)
    proj = ImageProjection(A=A, B=B, rank=r)
    embs = rng.standard_normal((n, D)).astype(np.float32)
    out = apply_projection(proj, embs)
    assert out.shape == (n, D)


def test_apply_projection_output_l2_normalized():
    """Output rows must be L2-normalized."""
    rng = np.random.default_rng(0)
    D, r = 8, 4
    A = rng.standard_normal((D, r)).astype(np.float32)
    B = rng.standard_normal((r, D)).astype(np.float32)
    proj = ImageProjection(A=A, B=B, rank=r)
    embs = rng.standard_normal((12, D)).astype(np.float32)
    out = apply_projection(proj, embs)
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, np.ones(12), atol=1e-5,
        err_msg="apply_projection output must be L2-normalized")


def test_apply_projection_identity_at_zero():
    """When A=0 and B=0, output equals normalized input."""
    rng = np.random.default_rng(0)
    D, r = 8, 4
    proj = ImageProjection(A=np.zeros((D, r), dtype=np.float32),
                           B=np.zeros((r, D), dtype=np.float32), rank=r)
    embs = rng.standard_normal((12, D)).astype(np.float32)
    embs_norm = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    out = apply_projection(proj, embs)
    np.testing.assert_allclose(out, embs_norm, atol=1e-5,
        err_msg="Zero projection must leave normalized input unchanged")


# ---------------------------------------------------------------------------
# fit_image_projection
# ---------------------------------------------------------------------------

def test_fit_image_projection_returns_image_projection():
    """fit_image_projection must return an ImageProjection."""
    batch = _make_batch()
    proj = fit_image_projection(batch, rank=2, n_steps=5, seed=0)
    assert isinstance(proj, ImageProjection)


def test_fit_image_projection_shape():
    """A and B have the right shapes."""
    D, r = 8, 4
    batch = _make_batch(D=D)
    proj = fit_image_projection(batch, rank=r, n_steps=5, seed=0)
    assert proj.A.shape == (D, r), f"Expected A shape ({D}, {r}), got {proj.A.shape}"
    assert proj.B.shape == (r, D), f"Expected B shape ({r}, {D}), got {proj.B.shape}"
    assert proj.rank == r


def test_fit_image_projection_requires_option_image_ids():
    """Must raise if batch.option_image_ids is None."""
    rng = np.random.default_rng(0)
    N, D = 10, 8
    utt = rng.standard_normal((N, D)).astype(np.float32)
    utt /= np.linalg.norm(utt, axis=1, keepdims=True)
    imgs = rng.standard_normal((N, 12, D)).astype(np.float32)
    imgs /= np.linalg.norm(imgs, axis=-1, keepdims=True)
    batch = build_trial_batch(
        utterance_emb=jnp.array(utt),
        option_embs=jnp.array(imgs),
        target_idx=jnp.zeros(N, dtype=jnp.int32),
        selected_idx=jnp.zeros(N, dtype=jnp.int32),
        listener_ids=jnp.zeros(N, dtype=jnp.int32),
        game_ids=jnp.zeros(N, dtype=jnp.int32),
        image_ids=jnp.zeros(N, dtype=jnp.int32),
        rep_num=jnp.zeros(N, dtype=jnp.int32),
        n_listeners=1, n_games=1, n_images=1,
    )
    with pytest.raises(ValueError, match="option_image_ids"):
        fit_image_projection(batch, rank=2, n_steps=5, seed=0)


# ---------------------------------------------------------------------------
# save / load roundtrip
# ---------------------------------------------------------------------------

def test_save_load_image_projection_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    D, r = 16, 4
    proj = ImageProjection(
        A=rng.standard_normal((D, r)).astype(np.float32),
        B=rng.standard_normal((r, D)).astype(np.float32),
        rank=r,
    )
    path = str(tmp_path / "proj.npz")
    save_image_projection(proj, path)
    loaded = load_image_projection(path)
    assert loaded.rank == proj.rank
    np.testing.assert_array_equal(loaded.A, proj.A)
    np.testing.assert_array_equal(loaded.B, proj.B)


# ---------------------------------------------------------------------------
# project_batch
# ---------------------------------------------------------------------------

def test_project_batch_updates_option_embs():
    """project_batch must replace option_embs with projected values."""
    rng = np.random.default_rng(0)
    D, r = 8, 4
    batch = _make_batch(D=D, rng=rng)
    proj = ImageProjection(
        A=rng.standard_normal((D, r)).astype(np.float32),
        B=rng.standard_normal((r, D)).astype(np.float32),
        rank=r,
    )
    projected_batch = project_batch(proj, batch)
    assert not np.allclose(
        np.array(projected_batch.option_embs),
        np.array(batch.option_embs),
    ), "project_batch must change option_embs when A, B are nonzero"


def test_project_batch_option_embs_consistent_with_all_image_embs():
    """After projection, option_embs[n, j] == all_image_embs[option_image_ids[n, j]]."""
    rng = np.random.default_rng(1)
    D, r = 8, 4
    batch = _make_batch(D=D, rng=rng)
    proj = ImageProjection(
        A=rng.standard_normal((D, r)).astype(np.float32),
        B=rng.standard_normal((r, D)).astype(np.float32),
        rank=r,
    )
    pb = project_batch(proj, batch)
    reconstructed = np.array(pb.all_image_embs)[np.array(pb.option_image_ids)]
    np.testing.assert_allclose(
        np.array(pb.option_embs), reconstructed, atol=1e-5,
        err_msg="option_embs must equal all_image_embs[option_image_ids] after projection"
    )


def test_project_batch_preserves_other_fields():
    """project_batch must not alter utterance_emb, target_idx, game_ids, etc."""
    rng = np.random.default_rng(2)
    D, r = 8, 4
    batch = _make_batch(D=D, rng=rng)
    proj = ImageProjection(
        A=rng.standard_normal((D, r)).astype(np.float32),
        B=rng.standard_normal((r, D)).astype(np.float32),
        rank=r,
    )
    pb = project_batch(proj, batch)
    np.testing.assert_array_equal(np.array(pb.utterance_emb), np.array(batch.utterance_emb))
    np.testing.assert_array_equal(np.array(pb.target_idx), np.array(batch.target_idx))
    np.testing.assert_array_equal(np.array(pb.game_ids), np.array(batch.game_ids))
