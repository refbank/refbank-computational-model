import redivis
import pandas as pd


def fetch_tables(dataset: str = "mcfrank/refbank") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fetches trials, messages, and choices tables from Redivis.

    Returns (trials_df, messages_df, choices_df) with the columns expected
    by loader.join_tables:
      trials_df:  trial_id, game_id, trial_num, rep_num, image_id, option_set
      messages_df: trial_id, order, text
      choices_df:  trial_id, listener_id, selected_image_id

    Requires a valid Redivis API token (redivis.login() or REDIVIS_API_TOKEN env var).
    """
    owner, name = dataset.split("/")
    ds = redivis.user(owner).dataset(name)

    # --- trials ---
    raw_trials = ds.table("trials").to_pandas_dataframe()
    # Drop excluded trials; keep only the columns we need.
    raw_trials = raw_trials[raw_trials["exclude"] == False].copy()  # noqa: E712
    trials_df = raw_trials[
        ["trial_id", "game_id", "trial_num", "rep_num", "target", "option_set"]
    ].rename(columns={"target": "image_id"})

    # --- messages ---
    # Only keep describer (speaker) messages; ignore system/matcher messages.
    raw_messages = ds.table("messages").to_pandas_dataframe()
    speaker_messages = raw_messages[raw_messages["role"] == "describer"].copy()
    messages_df = speaker_messages[
        ["trial_id", "message_number", "text"]
    ].rename(columns={"message_number": "order"})

    # --- choices ---
    # choice_id (singular) is the selected image ID; choices_id is the row PK.
    raw_choices = ds.table("choices").to_pandas_dataframe()
    choices_df = raw_choices[["trial_id", "player_id", "choice_id"]].rename(
        columns={"player_id": "listener_id", "choice_id": "selected_image_id"}
    )

    return trials_df, messages_df, choices_df
