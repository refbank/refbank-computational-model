from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class TrialBatch:
    utterance_emb:  jnp.ndarray  # (N, D) CLIP text, L2-normalized
    option_embs:    jnp.ndarray  # (N, 12, D) CLIP image, L2-normalized
    target_idx:     jnp.ndarray  # (N,) int — index of target in option_embs[n]
    selected_idx:   jnp.ndarray  # (N,) int — index of listener's pick
    listener_ids:   jnp.ndarray  # (N,) int in [0, n_listeners)
    game_ids:       jnp.ndarray  # (N,) int in [0, n_games)
    image_ids:      jnp.ndarray  # (N,) int in [0, n_images)
    rep_num:        jnp.ndarray  # (N,) int
    n_listeners:    int
    n_games:        int
    n_images:       int


def build_trial_batch(
    utterance_emb:  jnp.ndarray,
    option_embs:    jnp.ndarray,
    target_idx:     jnp.ndarray,
    selected_idx:   jnp.ndarray,
    listener_ids:   jnp.ndarray,
    game_ids:       jnp.ndarray,
    image_ids:      jnp.ndarray,
    rep_num:        jnp.ndarray,
    n_listeners:    int,
    n_games:        int,
    n_images:       int,
) -> TrialBatch:
    """
    Validates inputs and returns a TrialBatch.
    Raises ValueError if option_embs second dimension != 12.
    """
    if option_embs.shape[1] != 12:
        raise ValueError(
            f"option_embs must have 12 options per trial, got {option_embs.shape[1]}"
        )
    return TrialBatch(
        utterance_emb=utterance_emb,
        option_embs=option_embs,
        target_idx=target_idx,
        selected_idx=selected_idx,
        listener_ids=listener_ids,
        game_ids=game_ids,
        image_ids=image_ids,
        rep_num=rep_num,
        n_listeners=n_listeners,
        n_games=n_games,
        n_images=n_images,
    )
