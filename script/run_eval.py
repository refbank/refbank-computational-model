#!/usr/bin/env python3
"""
Cross-validate the listener model and report held-out log-likelihood.

Usage:
    .venv/bin/python script/run_eval.py
    .venv/bin/python script/run_eval.py --n-folds 5 --n-steps 5000 --seed 0

The script runs K-fold game-level cross-validation:
  - Splits 83 games into K folds
  - Fits the listener model (with learnable image embeddings) on train games
  - Evaluates held-out log P(selected | utterance, options) on test games
    using exp(mu_beta) as beta for unseen listeners
  - Reports mean held-out log-likelihood per fold and overall

Results saved to results/eval_listener_cv.csv.
"""
import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.embeddings.cache import EmbeddingCache
from code.eval.cross_val import cross_val_listener

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

STUDY_ID = "hawkins2020_characterizing_cued"
DATA_DIR = "data"
CACHE_DIR = os.path.join(DATA_DIR, "embeddings")
RESULTS_DIR = "results"


def main():
    parser = argparse.ArgumentParser(description="Cross-validate listener model")
    parser.add_argument("--n-splits", type=int, default=20)
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True, help="Directory to write eval_listener_cv.csv")
    args = parser.parse_args()

    parquet = os.path.join(DATA_DIR, f"{STUDY_ID}_joined.parquet")
    if not os.path.exists(parquet):
        raise FileNotFoundError(
            f"Joined data not found at {parquet}. "
            "Run the pipeline with --config full_gpu first."
        )
    df = pd.read_parquet(parquet)
    log.info("Loaded %d trial-listener rows, %d games", len(df), df["game_id"].nunique())

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

    log.info(
        "Running %d random 80/20 splits (%d SVI steps per split)...",
        args.n_splits, args.n_steps,
    )
    results = cross_val_listener(
        df, image_embs, text_embs,
        n_splits=args.n_splits,
        n_steps=args.n_steps,
        seed=args.seed,
    )

    log.info("\n=== Cross-validation results ===")
    log.info(results.to_string(index=False))
    log.info(
        "\nOverall: mean LL = %.4f ± %.4f nats  (%.3f ± %.3f bits)",
        results["mean_ll"].mean(),
        results["mean_ll"].std(),
        results["mean_ll_bits"].mean(),
        results["mean_ll_bits"].std(),
    )
    log.info("Baseline (uniform): %.4f nats", -np.log(12))

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "eval_listener_cv.csv")
    results.to_csv(out, index=False)
    log.info("Saved → %s", out)


if __name__ == "__main__":
    main()
