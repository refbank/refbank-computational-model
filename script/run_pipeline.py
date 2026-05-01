#!/usr/bin/env python3
"""
End-to-end pipeline: fetch → embed → fit listener → fit convention → analysis.

Run from the project root:
    .venv/bin/python script/run_pipeline.py --config quick_cpu
    .venv/bin/python script/run_pipeline.py --config gpu_test
    .venv/bin/python script/run_pipeline.py --config full_gpu
    .venv/bin/python script/run_pipeline.py --config path/to/custom.toml

Config files live in script/configs/. See plan/compute_plan.md for cluster setup.
Redivis will open a browser auth window on first use.
Images and embeddings are cached in data/ so subsequent runs are faster.

If --output-dir is given, saves listener_fit.npz, convention_fit.npz,
step_sizes.csv, and displacement.csv there for offline analysis.
"""
import argparse
import logging
import os
import sys
import time
import tomllib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIXTURES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "tests", "fixtures"
)
CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")

import redivis

from code.prep_data.redivis_client import fetch_tables, download_images
from code.prep_data.loader import join_tables
from code.prep_data.pipeline import build_trial_batch_from_df
from code.embeddings.clip_encoder import CLIPEncoder
from code.embeddings.cache import EmbeddingCache, embeddings_for_ids
from code.models.literal_listener import (
    fit_listener,
    save_listener_fit,
    load_listener_fit,
)
from code.models.conventional_speaker import (
    compute_r1_average,
    fit_convention,
    save_convention_fit,
    load_convention_fit,
)
from code.analysis.predictions import step_sizes_over_reps, semantic_displacement_after_error

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

STUDY_ID = "hawkins2020_characterizing_cued"
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images", STUDY_ID)
CACHE_DIR = os.path.join(DATA_DIR, "embeddings")


def load_config(name_or_path: str) -> dict:
    if os.path.exists(name_or_path):
        path = name_or_path
    else:
        path = os.path.join(CONFIGS_DIR, f"{name_or_path}.toml")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Config '{name_or_path}' not found as a path or in {CONFIGS_DIR}/. "
                f"Available configs: {', '.join(p.stem for p in __import__('pathlib').Path(CONFIGS_DIR).glob('*.toml'))}"
            )
    with open(path, "rb") as f:
        return tomllib.load(f)


def _elapsed(t0: float) -> str:
    secs = time.time() - t0
    if secs < 60:
        return f"{secs:.1f}s"
    return f"{secs / 60:.1f}min"


def main() -> None:
    parser = argparse.ArgumentParser(description="RefBank model pipeline")
    parser.add_argument(
        "--config",
        default="quick_cpu",
        help="Config name (looks in script/configs/) or path to a .toml file",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save fits and analysis CSVs (created if absent)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    log.info("Config: %s — %s", args.config, cfg.get("description", ""))

    n_games = cfg["n_games"]          # 0 means all games
    listener_steps = cfg["listener_steps"]
    convention_steps = cfg["convention_steps"]

    output_dir = args.output_dir
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        log.info("Results will be saved to %s/", output_dir)

    os.makedirs(CACHE_DIR, exist_ok=True)

    # --- auth ---
    log.info("Authenticating with Redivis (browser window will open if needed)...")
    redivis.authenticate()

    # --- fetch tabular data ---
    t0 = time.time()
    log.info("Fetching tables for %s...", STUDY_ID)
    trials_df, messages_df, choices_df = fetch_tables(STUDY_ID)
    log.info(
        "Fetched: %d trials  %d messages  %d choices  (%s)",
        len(trials_df), len(messages_df), len(choices_df), _elapsed(t0),
    )
    log.info("trials columns:  %s", list(trials_df.columns))
    log.info("messages columns: %s", list(messages_df.columns))
    log.info("choices columns:  %s", list(choices_df.columns))
    if len(trials_df):
        log.info("trials sample:\n%s", trials_df.head(2).to_string())
    if len(choices_df):
        log.info("choices sample:\n%s", choices_df.head(2).to_string())

    log.info("Joining tables...")
    df = join_tables(trials_df, messages_df, choices_df)
    log.info("After join: %d rows", len(df))

    if n_games > 0:
        first_games = df["game_id"].unique()[:n_games]
        df = df[df["game_id"].isin(first_games)].reset_index(drop=True)
        log.info("Using %d games, %d trial-listener rows", len(first_games), len(df))
    else:
        log.info("Using all %d games, %d trial-listener rows", df["game_id"].nunique(), len(df))

    # --- download images (cached after first run) ---
    log.info("Downloading images to %s/ ...", IMAGE_DIR)
    image_paths = download_images(STUDY_ID, IMAGE_DIR, overwrite=False)
    log.info("%d images available", len(image_paths))

    # --- compute embeddings (cached in .npz files) ---
    log.info("Loading CLIP encoder (downloads model on first run)...")
    encoder = CLIPEncoder()

    img_cache = EmbeddingCache(os.path.join(CACHE_DIR, f"{STUDY_ID}_images.npz"))
    all_image_ids = list({img for opts in df["option_set"] for img in opts})

    def image_encoder_fn(ids: list[str]) -> np.ndarray:
        paths = [image_paths[img_id] for img_id in ids]
        return encoder.encode_images(paths)

    t0 = time.time()
    log.info("Encoding %d distinct images...", len(all_image_ids))
    image_embs = embeddings_for_ids(all_image_ids, img_cache, image_encoder_fn)

    text_cache = EmbeddingCache(os.path.join(CACHE_DIR, f"{STUDY_ID}_texts.npz"))
    utterances = df["utterance"].unique().tolist()
    log.info("Encoding %d distinct utterances...", len(utterances))
    text_embs = embeddings_for_ids(utterances, text_cache, encoder.encode_texts)
    log.info("Embeddings ready (%s)", _elapsed(t0))

    # --- build batch ---
    log.info("Building TrialBatch...")
    batch = build_trial_batch_from_df(df, image_embs, text_embs)
    log.info(
        "Batch: %d trials  %d games  %d images  %d listeners",
        batch.utterance_emb.shape[0], batch.n_games, batch.n_images, batch.n_listeners,
    )

    # --- save fixtures for integration tests ---
    if n_games == 5:
        os.makedirs(FIXTURES_DIR, exist_ok=True)
        df_path = os.path.join(FIXTURES_DIR, "hawkins2020_5games.parquet")
        df.to_parquet(df_path, index=False)
        np.savez(os.path.join(FIXTURES_DIR, "hawkins2020_images.npz"), **image_embs)
        np.savez(
            os.path.join(FIXTURES_DIR, "hawkins2020_texts.npz"),
            **{k: v for k, v in text_embs.items() if k in set(df["utterance"])},
        )
        log.info("Fixtures saved to %s/", FIXTURES_DIR)

    # --- Stage 1: literal listener ---
    t0 = time.time()
    log.info("Fitting literal listener (%d SVI steps)...", listener_steps)
    listener_fit = fit_listener(batch, n_steps=listener_steps)
    log.info(
        "Listener fit done (%s) — mu_beta=%.3f  sigma_beta=%.3f",
        _elapsed(t0), listener_fit.mu_beta, listener_fit.sigma_beta,
    )
    log.info("Per-listener log-beta (posterior means): %s", listener_fit.beta_loc.round(3))

    if output_dir is not None:
        save_listener_fit(listener_fit, os.path.join(output_dir, "listener_fit.npz"))
        log.info("Saved listener_fit.npz")

    # --- Stage 2: convention model ---
    log.info("Computing R1 averages for mu_i init...")
    mu_i_init = compute_r1_average(batch)

    t0 = time.time()
    log.info("Fitting convention model (%d SVI steps)...", convention_steps)
    convention_fit = fit_convention(batch, listener_fit, mu_i_init, n_steps=convention_steps)
    log.info(
        "Convention fit done (%s) — sigma_game=%.4f  sigma_min=%.4f  sigma_max=%.4f",
        _elapsed(t0), convention_fit.sigma_game, convention_fit.sigma_min, convention_fit.sigma_max,
    )

    if output_dir is not None:
        save_convention_fit(convention_fit, os.path.join(output_dir, "convention_fit.npz"))
        log.info("Saved convention_fit.npz")

    # --- analysis ---
    log.info("Computing step sizes over reps (speaker role)...")
    step_df = step_sizes_over_reps(convention_fit, batch, role="speaker")
    print("\n--- Step sizes over repetitions (first 15 rows) ---")
    print(step_df.head(15).to_string(index=False))

    early = step_df[step_df["rep_num"] <= 2]["step_size"].mean()
    late  = step_df[step_df["rep_num"] >= 5]["step_size"].mean()
    print(f"\nMean step size reps 1-2: {early:.4f}")
    print(f"Mean step size reps 5+:  {late:.4f}")
    if not step_df[step_df["rep_num"] >= 5].empty:
        print(f"Step sizes decrease as expected: {late < early}")

    log.info("Computing semantic displacement after error...")
    disp_df = semantic_displacement_after_error(convention_fit, batch, role="speaker")
    after_failure = disp_df[~disp_df["prev_success"]]["displacement"].mean()
    after_success = disp_df[ disp_df["prev_success"]]["displacement"].mean()
    print(f"\n--- Semantic displacement ---")
    print(f"Mean displacement after failure:  {after_failure:.4f}")
    print(f"Mean displacement after success:  {after_success:.4f}")
    if not disp_df[~disp_df["prev_success"]].empty and not disp_df[disp_df["prev_success"]].empty:
        print(f"Larger after failure (expected):  {after_failure > after_success}")

    if output_dir is not None:
        step_df.to_csv(os.path.join(output_dir, "step_sizes.csv"), index=False)
        disp_df.to_csv(os.path.join(output_dir, "displacement.csv"), index=False)
        log.info("Saved step_sizes.csv and displacement.csv")


if __name__ == "__main__":
    main()
