import logging
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)


def join_tables(
    trials_df: pd.DataFrame,
    messages_df: pd.DataFrame,
    choices_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns one row per (trial_id, listener_id) with columns:
      trial_id, game_id, trial_num, rep_num, image_id, option_set,
      utterance, selected_image_id, listener_id

    Multi-message trials: sorted ascending by order, joined with ' '.
    Trials with no listener choice are dropped; count is logged as WARNING.
    """
    utterances = (
        messages_df.sort_values("order")
        .groupby("trial_id")["text"]
        .apply(lambda texts: " ".join(texts))
        .reset_index()
        .rename(columns={"text": "utterance"})
    )

    with_utterances = trials_df.merge(utterances, on="trial_id", how="left")
    with_choices = with_utterances.merge(choices_df, on="trial_id", how="left")

    missing = with_choices["listener_id"].isna().sum()
    if missing > 0:
        logger.warning("Dropped %d trial-listener rows with no listener choice", missing)
        with_choices = with_choices.dropna(subset=["listener_id"])

    return with_choices.reset_index(drop=True)


def filter_images(
    df: pd.DataFrame,
    mode: Literal["top_n", "threshold"],
    n: int,
) -> pd.DataFrame:
    """
    Keeps only games where every image in option_set passes the filter.
    mode="top_n": every image must be among the n most-frequent globally.
    mode="threshold": every image must appear in >= n games.
    Raises ValueError if no games survive.
    """
    all_images = df["option_set"].explode()

    if mode == "top_n":
        top_images = set(all_images.value_counts().head(n).index)
        def game_passes(option_set):
            return all(img in top_images for img in option_set)
    elif mode == "threshold":
        images_per_game = (
            df.explode("option_set")
            .groupby("option_set")["game_id"]
            .nunique()
        )
        qualifying = set(images_per_game[images_per_game >= n].index)
        def game_passes(option_set):
            return all(img in qualifying for img in option_set)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    # A game is valid only if ALL its rows pass; filter at game level
    game_valid = df.groupby("game_id")["option_set"].transform(
        lambda option_sets: all(game_passes(os) for os in option_sets)
    )
    result = df[game_valid].reset_index(drop=True)

    if len(result) == 0:
        raise ValueError(
            f"filter_images(mode={mode!r}, n={n}) removed all games"
        )
    return result
