from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pandas as pd


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


def build_trial_batch_from_df(
    df: pd.DataFrame,
    image_embeddings: dict[str, np.ndarray],
    utterance_embeddings: dict[str, np.ndarray],
) -> TrialBatch:
    """
    Constructs a TrialBatch from a joined DataFrame (output of join_tables).

    Raises ValueError if any option_set has length != 12.
    Raises KeyError("<id> not found in embeddings") for any missing image or utterance.
    Encodes game_ids, listener_ids, image_ids as contiguous integers starting from 0.
    """
    for _, row in df.iterrows():
        if len(row["option_set"]) != 12:
            raise ValueError(
                f"option_set must have 12 options, got {len(row['option_set'])}"
            )

    for utt in df["utterance"]:
        if utt not in utterance_embeddings:
            raise KeyError(f"{utt} not found in embeddings")

    all_image_ids: set[str] = set(df["image_id"]) | set(df["selected_image_id"])
    for options in df["option_set"]:
        all_image_ids.update(options)
    for img_id in all_image_ids:
        if img_id not in image_embeddings:
            raise KeyError(f"{img_id} not found in embeddings")

    game_enc = {g: i for i, g in enumerate(sorted(df["game_id"].unique()))}
    listener_enc = {l: i for i, l in enumerate(sorted(df["listener_id"].unique()))}
    image_enc = {img: i for i, img in enumerate(sorted(df["image_id"].unique()))}

    utterance_emb = np.stack([utterance_embeddings[u] for u in df["utterance"]])
    option_embs = np.stack([
        np.stack([image_embeddings[img] for img in row["option_set"]])
        for _, row in df.iterrows()
    ])
    target_idx = np.array(
        [row["option_set"].index(row["image_id"]) for _, row in df.iterrows()],
        dtype=np.int32,
    )
    selected_idx = np.array(
        [row["option_set"].index(row["selected_image_id"]) for _, row in df.iterrows()],
        dtype=np.int32,
    )

    return build_trial_batch(
        utterance_emb=jnp.array(utterance_emb),
        option_embs=jnp.array(option_embs),
        target_idx=jnp.array(target_idx),
        selected_idx=jnp.array(selected_idx),
        listener_ids=jnp.array([listener_enc[l] for l in df["listener_id"]], dtype=jnp.int32),
        game_ids=jnp.array([game_enc[g] for g in df["game_id"]], dtype=jnp.int32),
        image_ids=jnp.array([image_enc[img] for img in df["image_id"]], dtype=jnp.int32),
        rep_num=jnp.array(df["rep_num"].to_numpy(), dtype=jnp.int32),
        n_listeners=len(listener_enc),
        n_games=len(game_enc),
        n_images=len(image_enc),
    )
