import os

import redivis
import pandas as pd

_TABLE = {
    "trials":      "trials:zkj2",
    "messages":    "messages:2q18",
    "choices":     "choices:s1zj",
    "images":      "images:jw0t",
    "image_files": "image_files:zkvc",
}


def _ds(dataset: str = "mcfrank/refbank:2zy7", version: str = "current"):
    owner, name = dataset.split("/")
    return redivis.user(owner).dataset(name, version=version)


def fetch_tables(
    study_id: str,
    dataset: str = "mcfrank/refbank:2zy7",
    version: str = "current",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fetches trials, messages, and choices for study_id from Redivis.

    Returns (trials_df, messages_df, choices_df) with the columns expected
    by loader.join_tables:
      trials_df:  trial_id, game_id, trial_num, rep_num, image_id, option_set (list)
      messages_df: trial_id, order, text
      choices_df:  trial_id, listener_id, selected_image_id

    Filters server-side so only rows for study_id are downloaded.
    Requires Redivis auth: call redivis.authenticate() once to open the browser prompt,
    or set the REDIVIS_API_TOKEN environment variable.
    """
    ds = _ds(dataset, version)
    t = _TABLE["trials"]
    m = _TABLE["messages"]
    c = _TABLE["choices"]

    trials_df = ds.query(f"""
        SELECT trial_id, game_id, trial_num, rep_num,
               target AS image_id, option_set, `exclude`
        FROM `{t}`
        WHERE dataset_id = '{study_id}'
    """).to_pandas_dataframe()
    # Filter excluded trials in pandas — the exclude column may be BOOL or INT64
    # and BigQuery strict typing makes `= FALSE` unreliable for non-BOOL columns.
    trials_df = trials_df[~trials_df["exclude"].fillna(False).astype(bool)].drop(columns=["exclude"])
    trials_df["option_set"] = trials_df["option_set"].apply(_parse_semicolon_list)

    kept_trial_ids = set(trials_df["trial_id"])

    messages_df = ds.query(f"""
        SELECT trial_id, message_number AS `order`, text
        FROM `{m}`
        WHERE role = 'describer'
          AND dataset_id = '{study_id}'
    """).to_pandas_dataframe()
    messages_df = messages_df[messages_df["trial_id"].isin(kept_trial_ids)]

    choices_df = ds.query(f"""
        SELECT trial_id, player_id AS listener_id, choice_id AS selected_image_id
        FROM `{c}`
        WHERE dataset_id = '{study_id}'
    """).to_pandas_dataframe()
    choices_df = choices_df[choices_df["trial_id"].isin(kept_trial_ids)]

    return trials_df, messages_df, choices_df


def download_images(
    study_id: str,
    destination: str,
    dataset: str = "mcfrank/refbank:2zy7",
    version: str = "current",
    overwrite: bool = False,
) -> dict[str, str]:
    """
    Downloads image files for every image that appears in study_id's option sets.
    Saves files under destination/, preserving the image_path directory structure.

    Returns {image_id: absolute_local_path} for all images in the study.

    Requires Redivis auth (same as fetch_tables).
    """
    ds = _ds(dataset, version)
    t = _TABLE["trials"]
    i = _TABLE["images"]
    f = _TABLE["image_files"]

    # Get image_id -> image_path mapping for this study.
    images_df = ds.query(f"""
        WITH study_images AS (
            SELECT DISTINCT TRIM(image_id) AS image_id
            FROM `{t}`
            CROSS JOIN UNNEST(SPLIT(option_set, ';')) AS image_id
            WHERE dataset_id = '{study_id}'
        )
        SELECT img.image_id, img.image_path
        FROM `{i}` img
        JOIN study_images s ON img.image_id = s.image_id
    """).to_pandas_dataframe()

    # Download the actual files (image_files.file_name == images.image_path).
    os.makedirs(destination, exist_ok=True)
    ds.query(f"""
        WITH study_images AS (
            SELECT DISTINCT TRIM(image_id) AS image_id
            FROM `{t}`
            CROSS JOIN UNNEST(SPLIT(option_set, ';')) AS image_id
            WHERE dataset_id = '{study_id}'
        )
        SELECT files.*
        FROM `{f}` files
        JOIN `{i}` img ON files.file_name = img.image_path
        JOIN study_images s ON img.image_id = s.image_id
    """).download_files(path=destination, overwrite=overwrite)

    return {
        row["image_id"]: os.path.join(destination, row["image_path"])
        for _, row in images_df.iterrows()
    }


def _parse_semicolon_list(value) -> list[str]:
    """Parses option_set, which is stored as a semicolon-separated string."""
    if isinstance(value, list):
        return value
    return [item.strip() for item in str(value).split(";") if item.strip()]
