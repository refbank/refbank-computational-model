import numpy as np
import jax.numpy as jnp
import pytest

from code.prep_data.pipeline import build_trial_batch
from code.models.literal_listener import ListenerFit
from code.eval.cross_val import game_level_splits, listener_log_likelihood


def _make_batch(N=60, D=8, n_games=10, n_listeners=10, n_all_images=12, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    utt = rng.standard_normal((N, D)).astype(np.float32)
    utt /= np.linalg.norm(utt, axis=1, keepdims=True)
    all_img_embs = rng.standard_normal((n_all_images, D)).astype(np.float32)
    all_img_embs /= np.linalg.norm(all_img_embs, axis=1, keepdims=True)
    option_image_ids = np.tile(np.arange(12, dtype=np.int32), (N, 1))
    imgs = all_img_embs[option_image_ids]
    game_ids = np.repeat(np.arange(n_games), N // n_games)
    listener_ids = np.repeat(np.arange(n_listeners), N // n_listeners)
    return build_trial_batch(
        utterance_emb=jnp.array(utt),
        option_embs=jnp.array(imgs),
        target_idx=jnp.zeros(N, dtype=jnp.int32),
        selected_idx=jnp.zeros(N, dtype=jnp.int32),
        listener_ids=jnp.array(listener_ids, dtype=jnp.int32),
        game_ids=jnp.array(game_ids, dtype=jnp.int32),
        image_ids=jnp.zeros(N, dtype=jnp.int32),
        rep_num=jnp.zeros(N, dtype=jnp.int32),
        n_listeners=n_listeners,
        n_games=n_games,
        n_images=1,
        option_image_ids=jnp.array(option_image_ids),
        all_image_embs=jnp.array(all_img_embs),
    )


def _make_fit(mu_beta=1.0, rng=None):
    return ListenerFit(
        beta_loc=np.full(5, mu_beta),
        beta_scale=np.ones(5) * 0.1,
        mu_beta=mu_beta,
        sigma_beta=0.5,
    )


# ---------------------------------------------------------------------------
# game_level_splits
# ---------------------------------------------------------------------------

def test_game_level_splits_returns_n_splits():
    """Returns exactly n_splits (train, test) tuples."""
    game_ids = np.repeat(np.arange(20), 6)
    splits = game_level_splits(game_ids, n_splits=10, seed=0)
    assert len(splits) == 10


def test_game_level_splits_80_20_ratio():
    """Each split has approximately 80% train and 20% test games."""
    game_ids = np.repeat(np.arange(20), 6)  # 20 games
    splits = game_level_splits(game_ids, n_splits=5, seed=0)
    for train, test in splits:
        assert len(test) == 4, f"Expected 4 test games (20% of 20), got {len(test)}"
        assert len(train) == 16, f"Expected 16 train games (80% of 20), got {len(train)}"


def test_game_level_splits_train_test_disjoint():
    """Train and test sets must be disjoint in every split."""
    game_ids = np.repeat(np.arange(20), 6)
    splits = game_level_splits(game_ids, n_splits=10, seed=0)
    for train, test in splits:
        assert len(set(train) & set(test)) == 0, "Train/test must not overlap"


def test_game_level_splits_reproducible():
    """Same seed produces the same sequence of splits."""
    game_ids = np.repeat(np.arange(20), 4)
    s1 = game_level_splits(game_ids, n_splits=5, seed=42)
    s2 = game_level_splits(game_ids, n_splits=5, seed=42)
    for (tr1, te1), (tr2, te2) in zip(s1, s2):
        np.testing.assert_array_equal(tr1, tr2)
        np.testing.assert_array_equal(te1, te2)


def test_game_level_splits_different_seeds_differ():
    """Different seeds produce different splits."""
    game_ids = np.repeat(np.arange(20), 4)
    s1 = game_level_splits(game_ids, n_splits=5, seed=0)
    s2 = game_level_splits(game_ids, n_splits=5, seed=1)
    test_sets_1 = [set(te) for _, te in s1]
    test_sets_2 = [set(te) for _, te in s2]
    assert test_sets_1 != test_sets_2


def test_game_level_splits_independent():
    """Splits are independent — 100 splits should not all have the same test set."""
    game_ids = np.repeat(np.arange(20), 4)
    splits = game_level_splits(game_ids, n_splits=100, seed=0)
    test_sets = [frozenset(te) for _, te in splits]
    assert len(set(test_sets)) > 1, "100 splits must not all have the same test set"


# ---------------------------------------------------------------------------
# listener_log_likelihood
# ---------------------------------------------------------------------------

def test_listener_log_likelihood_shape():
    """Returns (N,) array of per-trial log-likelihoods."""
    batch = _make_batch(N=60)
    fit = _make_fit()
    ll = listener_log_likelihood(fit, batch)
    assert ll.shape == (60,)


def test_listener_log_likelihood_values_non_positive():
    """Log-probabilities must be ≤ 0 (probabilities ≤ 1)."""
    batch = _make_batch(N=60)
    fit = _make_fit()
    ll = listener_log_likelihood(fit, batch)
    assert np.all(ll <= 0), f"Log-probs must be ≤ 0, got max {ll.max():.4f}"



def test_listener_log_likelihood_uses_population_mu_beta():
    """Must use fit.mu_beta (population mean) not per-listener beta_loc."""
    rng = np.random.default_rng(7)
    batch = _make_batch(N=60, D=8, rng=rng)

    # Same beta_loc but different mu_beta — LL must differ.
    fit_low = ListenerFit(
        beta_loc=np.full(10, 1.0), beta_scale=np.ones(10) * 0.1,
        mu_beta=0.0, sigma_beta=0.5,
    )
    fit_high = ListenerFit(
        beta_loc=np.full(10, 1.0), beta_scale=np.ones(10) * 0.1,
        mu_beta=3.0, sigma_beta=0.5,
    )
    ll_low = listener_log_likelihood(fit_low, batch)
    ll_high = listener_log_likelihood(fit_high, batch)
    assert not np.allclose(ll_low, ll_high), (
        "LL must depend on fit.mu_beta, not just beta_loc"
    )


def test_listener_log_likelihood_uses_batch_option_embs():
    """LL must differ when batch.option_embs differ (with same mu_beta)."""
    rng = np.random.default_rng(9)
    N, D = 60, 8
    fit = _make_fit(mu_beta=1.0)

    batch_a = _make_batch(N=N, D=D, rng=np.random.default_rng(1))
    batch_b = _make_batch(N=N, D=D, rng=np.random.default_rng(2))
    ll_a = listener_log_likelihood(fit, batch_a)
    ll_b = listener_log_likelihood(fit, batch_b)
    assert not np.allclose(ll_a, ll_b), "LL must depend on batch.option_embs"
