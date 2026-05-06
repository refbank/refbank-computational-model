import logging

import numpy as np
import pandas as pd
import scipy.special

from code.models.literal_listener import ListenerFit, fit_listener
from code.prep_data.pipeline import TrialBatch, build_trial_batch_from_df

log = logging.getLogger(__name__)


def game_level_splits(
    game_ids: np.ndarray,
    n_splits: int,
    seed: int,
    test_fraction: float = 0.2,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Repeated random 80/20 game-level splits.

    Each split independently draws test_fraction of unique games as the test
    set and the remainder as train.  Splits are independent — a game may appear
    as test in multiple splits or none.

    Returns a list of n_splits (train_game_ids, test_game_ids) tuples.
    """
    unique_games = np.unique(game_ids)
    n_test = max(1, round(len(unique_games) * test_fraction))
    rng = np.random.default_rng(seed)
    splits = []
    for _ in range(n_splits):
        shuffled = rng.permutation(unique_games)
        test_games = shuffled[:n_test]
        train_games = shuffled[n_test:]
        splits.append((train_games, test_games))
    return splits


def listener_log_likelihood(fit: ListenerFit, batch: TrialBatch) -> np.ndarray:
    """
    Per-trial log P(selected_idx | utterance, options) for held-out evaluation.

    Uses fit.mu_beta as the beta for all trials (population predictive mean for
    listeners not seen during training).  Uses fit.image_emb_loc when available,
    otherwise falls back to batch.option_embs.

    Returns (N,) array of per-trial log-probabilities (all ≤ 0).
    """
    beta = np.exp(fit.mu_beta)

    if fit.image_emb_loc is not None:
        if batch.option_image_ids is None:
            raise ValueError(
                "batch.option_image_ids must be set when fit.image_emb_loc is present"
            )
        option_embs = fit.image_emb_loc[np.array(batch.option_image_ids)]  # (N, 12, D)
        norms = np.linalg.norm(option_embs, axis=-1, keepdims=True)
        option_embs = option_embs / norms
    else:
        option_embs = np.array(batch.option_embs)

    cos_sims = np.einsum("nd,nkd->nk", np.array(batch.utterance_emb), option_embs)
    logits = beta * cos_sims  # (N, 12)

    log_probs = logits - scipy.special.logsumexp(logits, axis=1, keepdims=True)
    selected = np.array(batch.selected_idx)
    return log_probs[np.arange(len(selected)), selected]


def cross_val_listener(
    df: pd.DataFrame,
    image_embeddings: dict,
    utterance_embeddings: dict,
    n_splits: int = 20,
    n_steps: int = 5000,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Repeated random 80/20 game-level cross-validation of the listener model.

    Each split independently draws 20% of games as test and fits the listener
    model on the remaining 80%.  Held-out log-likelihood is evaluated using
    exp(mu_beta) as beta for unseen listeners and the learned image_emb_loc.

    Returns a DataFrame with columns:
        split, n_train_trials, n_test_trials, mean_ll, mean_ll_bits
    """
    all_game_ids = df["game_id"].unique()
    splits = game_level_splits(all_game_ids, n_splits=n_splits, seed=seed)

    rows = []
    for split_i, (train_games, test_games) in enumerate(splits):
        log.info(
            "Split %d/%d: %d train games, %d test games",
            split_i + 1, n_splits, len(train_games), len(test_games),
        )

        train_df = df[df["game_id"].isin(train_games)].reset_index(drop=True)
        test_df = df[df["game_id"].isin(test_games)].reset_index(drop=True)

        train_batch = build_trial_batch_from_df(train_df, image_embeddings, utterance_embeddings)
        test_batch = build_trial_batch_from_df(test_df, image_embeddings, utterance_embeddings)

        fit = fit_listener(train_batch, n_steps=n_steps, seed=seed + split_i)
        log.info(
            "  Split %d: mu_beta=%.3f  sigma_beta=%.3f",
            split_i + 1, fit.mu_beta, fit.sigma_beta,
        )

        ll = listener_log_likelihood(fit, test_batch)
        rows.append({
            "split": split_i,
            "n_train_trials": len(train_df),
            "n_test_trials": len(test_df),
            "mean_ll": float(ll.mean()),
            "mean_ll_bits": float(ll.mean() / np.log(2)),  # nats → bits
        })
        log.info("  Split %d: mean LL = %.4f nats", split_i + 1, rows[-1]["mean_ll"])

    return pd.DataFrame(rows)
