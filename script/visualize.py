#!/usr/bin/env python3
"""
Generate interactive convention trajectory visualisations from a completed run.

Usage:
    # Overview (dropdown: image or game) — saves overview.html
    .venv/bin/python script/visualize.py --results results/run_23851534

    # Single game detail — saves game_5.html
    .venv/bin/python script/visualize.py --results results/run_23851534 --game 5

    # All games — saves game_0.html, game_1.html, ...
    .venv/bin/python script/visualize.py --results results/run_23851534 --all-games

t-SNE takes ~30s on first run and is cached to results/run_.../tsne_coords.parquet.
"""
import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.prep_data.pipeline import build_trial_batch_from_df
from code.embeddings.cache import EmbeddingCache
from code.models.conventional_speaker import load_convention_fit
from code.analysis.visualization import (
    compute_tsne_coords,
    compute_per_game_tsne_coords,
    plot_overview,
    plot_per_game_overview,
    plot_game,
    combined_html,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

STUDY_ID = "hawkins2020_characterizing_cued"
DATA_DIR = "data"
CACHE_DIR = os.path.join(DATA_DIR, "embeddings")


def load_batch(n_games: int = 0):
    parquet = os.path.join(DATA_DIR, f"{STUDY_ID}_joined.parquet")
    if not os.path.exists(parquet):
        raise FileNotFoundError(
            f"Joined data not found at {parquet}. "
            "Run the pipeline with --config full_gpu first."
        )
    df = pd.read_parquet(parquet)
    log.info("Loaded %d trial-listener rows, %d games", len(df), df["game_id"].nunique())

    if n_games > 0:
        first_games = df["game_id"].unique()[:n_games]
        df = df[df["game_id"].isin(first_games)].reset_index(drop=True)
        log.info("Filtered to %d games", n_games)

    all_image_ids = list({img for opts in df["option_set"] for img in opts})
    utterances = df["utterance"].unique().tolist()

    img_cache = EmbeddingCache(os.path.join(CACHE_DIR, f"{STUDY_ID}_images.npz"))
    text_cache = EmbeddingCache(os.path.join(CACHE_DIR, f"{STUDY_ID}_texts.npz"))

    image_embs, missing = img_cache.get(all_image_ids)
    if missing:
        raise RuntimeError(f"{len(missing)} image embeddings missing from cache.")
    text_embs, missing = text_cache.get(utterances)
    if missing:
        raise RuntimeError(f"{len(missing)} text embeddings missing from cache.")

    batch = build_trial_batch_from_df(df, image_embs, text_embs)
    log.info(
        "Batch: %d trials, %d games, %d images, %d listeners",
        batch.utterance_emb.shape[0], batch.n_games, batch.n_images, batch.n_listeners,
    )
    return batch


def get_coords(fit, batch, results_dir: str, role: str, perplexity: int, seed: int):
    cache_path = os.path.join(results_dir, "tsne_coords.parquet")
    if os.path.exists(cache_path):
        log.info("Loading cached t-SNE coords from %s", cache_path)
        return pd.read_parquet(cache_path)
    log.info("Computing t-SNE (~30s)...")
    coords = compute_tsne_coords(fit, batch, role=role, perplexity=perplexity, seed=seed)
    coords.to_parquet(cache_path, index=False)
    log.info("Cached to %s", cache_path)
    return coords


def get_per_game_coords(fit, batch, results_dir: str, role: str, perplexity: int, seed: int):
    cache_path = os.path.join(results_dir, "tsne_per_game_coords.parquet")
    if os.path.exists(cache_path):
        log.info("Loading cached per-game t-SNE coords from %s", cache_path)
        return pd.read_parquet(cache_path)
    log.info("Computing per-game t-SNE...")
    coords = compute_per_game_tsne_coords(fit, batch, role=role, perplexity=perplexity, seed=seed)
    coords.to_parquet(cache_path, index=False)
    log.info("Cached to %s", cache_path)
    return coords


def main():
    parser = argparse.ArgumentParser(description="Visualise convention trajectories")
    parser.add_argument("--results", required=True, help="Path to results directory (e.g. results/run_23851534)")
    parser.add_argument("--game", type=int, default=None, help="Plot a single game detail (saves game_N.html)")
    parser.add_argument("--all-games", action="store_true", help="Plot all game details (saves game_N.html for each)")
    parser.add_argument("--role", default="speaker", choices=["speaker", "listener"])
    parser.add_argument("--perplexity", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-games", type=int, default=0, help="Limit to first N games (0 = all)")
    parser.add_argument("--per-game", action="store_true", help="Generate per-game t-SNE grid (per_game_overview.html)")
    parser.add_argument("--combined", action="store_true", help="Generate single combined HTML with Overview and Per-game tabs (combined.html)")
    args = parser.parse_args()

    if not os.path.isdir(args.results):
        raise FileNotFoundError(f"Results directory not found: {args.results}")

    fit = load_convention_fit(os.path.join(args.results, "convention_fit.npz"))
    log.info(
        "Fit: sigma_min=%.5f  sigma_max=%.5f  sigma_game=%.5f",
        fit.sigma_min, fit.sigma_max, fit.sigma_game,
    )

    batch = load_batch(n_games=args.n_games)
    coords = get_coords(fit, batch, args.results, args.role, args.perplexity, args.seed)

    if args.combined:
        per_game_coords = get_per_game_coords(fit, batch, args.results, args.role, args.perplexity, args.seed)
        log.info("Building overview figure...")
        ov_fig = plot_overview(coords)
        log.info("Building per-game figure...")
        pg_fig = plot_per_game_overview(per_game_coords)
        log.info("Serializing combined HTML...")
        html = combined_html(ov_fig, pg_fig)
        out = os.path.join(args.results, "combined.html")
        log.info("Writing %.1f MB...", len(html) / 1e6)
        with open(out, "w") as f:
            f.write(html)
        log.info("Saved combined → %s", out)

    if args.per_game:
        per_game_coords = get_per_game_coords(fit, batch, args.results, args.role, args.perplexity, args.seed)
        out = os.path.join(args.results, "per_game_overview.html")
        plot_per_game_overview(per_game_coords).write_html(out)
        log.info("Saved per-game overview → %s", out)

    if args.game is None and not args.all_games and not args.per_game and not args.combined:
        # Default: overview only
        out = os.path.join(args.results, "overview.html")
        plot_overview(coords).write_html(out)
        log.info("Saved overview → %s", out)

    if args.game is not None:
        out = os.path.join(args.results, f"game_{args.game}.html")
        plot_game(coords, game_id=args.game).write_html(out)
        log.info("Saved game %d → %s", args.game, out)

    if args.all_games:
        n = int(coords[coords["kind"] == "utterance"]["game_id"].max()) + 1
        for g in range(n):
            out = os.path.join(args.results, f"game_{g}.html")
            plot_game(coords, game_id=g).write_html(out)
        log.info("Saved %d game plots to %s/", n, args.results)


if __name__ == "__main__":
    main()
