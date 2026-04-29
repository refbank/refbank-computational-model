import logging

import pandas as pd
import pytest

from code.prep_data.loader import join_tables, filter_images


def _trials(rows):
    return pd.DataFrame(rows, columns=["trial_id", "game_id", "trial_num", "rep_num", "image_id", "option_set"])


def _messages(rows):
    return pd.DataFrame(rows, columns=["trial_id", "order", "text"])


def _choices(rows):
    return pd.DataFrame(rows, columns=["trial_id", "listener_id", "selected_image_id"])


def test_join_tables_concats_messages_in_order():
    trials = _trials([("t1", "g1", 1, 1, "img1", ["img1", "img2"])])
    messages = _messages([("t1", 2, "second"), ("t1", 1, "first")])
    choices = _choices([("t1", "l1", "img1")])

    df = join_tables(trials, messages, choices)

    assert len(df) == 1
    assert df.iloc[0]["utterance"] == "first second"


def test_join_tables_multi_listener_produces_multiple_rows():
    trials = _trials([("t1", "g1", 1, 1, "img1", ["img1", "img2"])])
    messages = _messages([("t1", 1, "hello")])
    choices = _choices([("t1", "l1", "img1"), ("t1", "l2", "img2")])

    df = join_tables(trials, messages, choices)

    assert len(df) == 2
    assert set(df["listener_id"]) == {"l1", "l2"}
    assert list(df["utterance"].unique()) == ["hello"]


def test_join_tables_drops_missing_choices_with_warning(caplog):
    trials = _trials([
        ("t1", "g1", 1, 1, "img1", ["img1", "img2"]),
        ("t2", "g1", 2, 2, "img1", ["img1", "img2"]),
    ])
    messages = _messages([("t1", 1, "hello"), ("t2", 1, "world")])
    choices = _choices([("t1", "l1", "img1")])  # t2 has no choice

    with caplog.at_level(logging.WARNING):
        df = join_tables(trials, messages, choices)

    assert len(df) == 1
    assert df.iloc[0]["trial_id"] == "t1"
    assert any("1" in record.message for record in caplog.records), (
        f"Expected warning mentioning count of dropped rows; got: {[r.message for r in caplog.records]}"
    )


def test_filter_images_top_n():
    # Each game has option_set with 2 images (no repeats within a row).
    # img1: 5 rows, img2: 4 rows, img3: 3 rows, img4: 1 row, img5: 1 row
    # With n=3, top images are img1, img2, img3
    # game g_pass uses only img1+img2 → kept
    # game g_fail uses img1+img4 → dropped (img4 not in top 3)
    df = pd.DataFrame({
        "game_id": (
            ["g_pass"] * 5 +   # all rows use img1, img2
            ["g_fail"] * 4 +   # uses img1, img4 (img4 only 1 row here)
            ["g_extra"] * 3    # uses img1, img3
        ),
        "option_set": (
            [["img1", "img2"]] * 5 +
            [["img1", "img2"]] * 3 + [["img1", "img4"]] * 1 +  # g_fail has 1 row with img4
            [["img1", "img3"]] * 3
        ),
    })
    result = filter_images(df, mode="top_n", n=3)
    assert "g_pass" in result["game_id"].values
    assert "g_extra" in result["game_id"].values
    assert "g_fail" not in result["game_id"].values


def test_filter_images_threshold():
    # img1 appears in 3 games, img2 in 2 games, img3 in 1 game
    # threshold=2: img3 fails → game g3 (which uses img3) is dropped
    df = pd.DataFrame({
        "game_id": ["g1", "g2", "g3"],
        "option_set": [
            ["img1", "img2"],
            ["img1", "img2"],
            ["img1", "img3"],
        ],
    })
    result = filter_images(df, mode="threshold", n=2)
    assert "g1" in result["game_id"].values
    assert "g2" in result["game_id"].values
    assert "g3" not in result["game_id"].values


def test_filter_images_raises_on_empty_result():
    df = pd.DataFrame({
        "game_id": ["g1"],
        "option_set": [["img1", "img2"]],
    })
    with pytest.raises(ValueError):
        filter_images(df, mode="top_n", n=0)
