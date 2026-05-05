# Visualisation how-to

Interactive t-SNE plots of convention trajectories, generated from a completed pipeline run.

## Quick start

```bash
# Overview plot (dropdown: all images / all games) → results/run_.../overview.html
.venv/bin/python script/visualize.py --results results/run_23851534

# Single game detail → results/run_.../game_5.html
.venv/bin/python script/visualize.py --results results/run_23851534 --game 5

# All game detail plots → results/run_.../game_0.html, game_1.html, ...
.venv/bin/python script/visualize.py --results results/run_23851534 --all-games
```

Open the resulting `.html` files in any browser — no server needed.

## t-SNE caching

t-SNE takes ~30s at full data scale. After the first run, coordinates are cached to
`results/run_.../tsne_coords.parquet` and reloaded instantly on subsequent calls.
Delete that file to recompute (e.g. after changing `--perplexity` or `--seed`).

## What the plots show

**Overview (`overview.html`)** — all utterance embeddings as grey dots, global prototypes
$\mu_i$ as coloured diamonds (always shown). Dropdown at top-left lets you select:

- **image N** — highlights all utterances for that image across all 83 games (coloured),
  with convention points $m_{g,i}$ shown as stars. Useful for seeing whether different
  games describe the same image similarly.
- **game N** — highlights all utterances for that game (coloured by image), with trajectory
  lines connecting consecutive reps for each image. Useful for seeing per-game convergence.

**Game detail (`game_N.html`)** — single game, all other utterances in grey.
Utterances coloured by image, with lines connecting reps in order.
- Marker **size** increases with rep number (later reps are bigger dots).
- **Circle** = listener correct; **×** = listener incorrect.
- Stars = convention points $m_{g,i}$; diamonds = prototypes $\mu_i$.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--results` | (required) | Path to run output directory |
| `--game N` | — | Generate detail plot for game N only |
| `--all-games` | — | Generate detail plots for all games |
| `--role` | `speaker` | Which convention fit to use (`speaker` or `listener`) |
| `--perplexity` | 30 | t-SNE perplexity |
| `--seed` | 0 | Random seed for t-SNE |
| `--n-games` | 0 (all) | Limit to first N games (useful for quick checks) |

## Notes

- The fit is loaded from `--results`; the data (parquet + embeddings) is loaded from
  `data/` in the project root. Make sure the data matches the fit (same dataset).
- `--n-games` filters the batch to the first N games; the fit is sliced to match.
  Convention points for games beyond N are not plotted.
- Plots use the speaker-side convention points by default (`--role speaker`).
  Use `--role listener` to visualise the listener-side conventions instead.
