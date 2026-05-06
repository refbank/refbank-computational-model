# Scratchpad

## Current position: sigma collapse fix pending cluster validation; viz improved

All 68 unit tests pass. Sigma collapse fix deployed (LogNormal priors); needs a cluster
run to verify. Visualisation improved with arrows, halo, white background, better legend.

Full dataset: 5971 trial-listener rows, 83 games, 12 images, 83 listeners. Listener fit
takes ~6.6s, convention fit ~14.6s on GPU.

`run_svi` uses `jax.lax.scan` — unit tests run in ~8s. `script/cluster_job.sh` submits
via SLURM. Data is preloaded into the repo so the cluster runs with `--no-fetch`.
Fits and analysis CSVs saved to `results/run_<JOBID>/`.

## How to run the pipeline

```bash
# Sanity check on laptop (20 SVI steps — confirms pipeline works, posteriors meaningless)
.venv/bin/python script/run_pipeline.py --config quick_cpu

# GPU smoke test (500 steps, 5 games — confirms GPU env works, loss should decrease)
.venv/bin/python script/run_pipeline.py --config gpu_test

# Full inference (5000/8000 steps, all games — requires GPU; see plan/compute_plan.md)
.venv/bin/python script/run_pipeline.py --config full_gpu
```

Config files are in `script/configs/*.toml`. Custom configs can be passed as a path.

## How to run on the cluster

Data and embeddings are preloaded into the repo (`data/` directory), so no Redivis fetch
needed on the cluster. Just:

```bash
sbatch script/cluster_job.sh
```

Results go to `results/run_<JOBID>/`: `listener_fit.npz`, `convention_fit.npz`,
`step_sizes.csv`, `displacement.csv`.

To refresh the preloaded data (on laptop, with Redivis access):
```bash
.venv/bin/python script/run_pipeline.py --config fetch_only
```

## Cluster setup (working as of 2026-05-01)

Cluster runs CentOS 7 (glibc 2.17), Python 3.12.1. jaxlib ≥ 0.8 requires glibc 2.28+
so cannot be pip-installed; use the pre-built cluster module instead.

```bash
module load math
module load py-jax/0.4.36_py312   # JAX 0.4.36, Python 3.12
```

Then create venv and install remaining deps:
```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install numpyro redivis transformers torch Pillow cairosvg "httpx[socks]" "numpy>=2.0" "pandas>=2.0"
```

**Redivis not needed for cluster run** — use cached embeddings and data already in the
repo. The pipeline can be run directly from cached fixtures without fetching from Redivis.

Submit with: `sbatch script/cluster_job.sh` (edit `cluster_job.sh` to add
`module load math` and `module load py-jax/0.4.36_py312` before activating the venv).

**pyproject.toml note**: `jax[cpu]==0.9.2` is the laptop version; cluster uses 0.4.36
from the module. Don't `pip install -e .` on the cluster.

## Open questions / next steps

### Immediate
- [ ] Run cluster job to verify sigma collapse fix (LogNormal priors). Check that sigma_min
  and sigma_max are in a plausible range (not 1e-5).

### Visualisation — overview.html improvements
- [ ] Separate image / rep_num / game dropdowns (currently one combined dropdown).
  When filtering by image, only that image's prototype (◆) should be shown; all others hidden.
- [ ] Per-game t-SNE tab: a second view where t-SNE is computed separately per game
  (so each game's utterances fill the space rather than being scattered in a global embedding).
  Tabs could be faked with Plotly button groups toggling div visibility in HTML.
- [ ] Refresh overview.html: `.venv/bin/python script/visualize.py --results results/run_<JOBID>`
  Delete `tsne_coords.parquet` first to force t-SNE recomputation.

### Dashboard / results presentation
- [ ] Build a results dashboard — either GitHub Pages (static HTML) or locally hosted.
  Should consolidate all plots below into one place.
- [ ] **Predicted vs observed accuracy**: model-implied L_0 success probability vs
  empirical listener accuracy, broken out by rep_num (calibration curve + learning curve).
- [ ] **Counterfactual utility plot**: emission log-likelihood of the observed utterance
  under the convention model vs counterfactual utterances (e.g. utterances from other games
  for the same image, or randomly sampled utterances). Shows whether within-game conventions
  are more distinctive / higher-utility than cross-game utterances.
- [ ] **Step size / convergence plot**: one of:
  - Embedding distance between consecutive rep utterances ($\|z(u_t) - z(u_{t-1})\|$)
    over rep_num, averaged across games/images — should decrease as conventions form.
  - $\sigma^2_t$ (emission variance) over rep_num — should narrow as s_t rises and
    conventions tighten. Shows the model's "confidence" in the convention growing.
  Both are already partially computable from existing `step_sizes_over_reps` and
  `semantic_displacement_after_error` outputs; need a plot function.

### Evaluation
- [ ] **Log-likelihood on held-out data**: compute held-out log-likelihood as the primary
  scalar metric for model quality. Concretely: for listener model, log P(selected_idx | utterance,
  options, β); for convention model, log P(utterance_emb | m_{g,i}, σ²_t). Higher = better.
- [ ] **Many-fold 80/20 splits**: split at the game level (keep all reps of a game together
  to avoid leakage). Fit on 80% of games, evaluate log-likelihood on held-out 20%. Repeat
  many times (e.g. 20 folds) to get a stable estimate with error bars. This is the main
  robustness check for both the listener and convention models.
  - Split unit: game (not trial), so convention structure is intact in each fold.
  - For listener model: β_ℓ are game-specific — held-out listeners need new β estimates
    or held-out must be defined differently (TBD).
  - Log-likelihood on held-out utterance embeddings (convention model) is probably
    the most interpretable metric.

### Analysis (other)
- [ ] Utterance scoring: use convention model emission to score actual vs counterfactual
  utterances (e.g. other games' utterances for the same image). Tests whether within-game
  conventions are more distinctive than across-game utterances.
- [ ] Should `fit_convention` save intermediate checkpoints in case a cluster job is killed?

---

## Model development roadmap (post-eval)

Do not start on these until 80/20 cross-validation log-likelihood is working and giving
stable numbers. The eval harness is the measuring stick for all model changes below.

### Role / player identity

- **Describer / matcher in one embedding space**: currently the model fits separate
  speaker-side and listener-side conventions independently. A cleaner alternative is a
  single model with a `role` or `player_id` variable, where the same convention space is
  shared but the update signal differs by role (speaker: utterance emission; listener:
  choice likelihood). `player_id` could also index per-player deviations (see M4 below).
- **Role switch asymmetry**: in RefBank games one player is always the describer and the
  other(s) are matchers. The current model treats them fully separately. Consider whether
  the convention for an image should be the same object for both roles (shared latent m_{g,i})
  or role-specific. Shared is more parsimonious; role-specific is more flexible.

### Initialization

- **Current**: μ_i initialized from the mean of rep_num=1 utterance embeddings across games
  (compute_r1_average). Reasonable but depends entirely on round-1 utterances.
- **Alternative**: initialize μ_i from the CLIP image embedding directly (z(image_i)),
  projecting into text space via the text encoder. Would give a truly prior-to-data starting
  point. Downside: image and text embeddings are not perfectly aligned in CLIP space.
- **Alternative**: random init, let SVI find μ_i from scratch. Slower convergence but no
  data-dependent initialization bias.
- Decision point: does initialization affect final fitted values (not just convergence speed)?
  Check by comparing held-out log-likelihood across inits.

### Ablations (compare by held-out log-likelihood)

- **Complete-pooling**: single shared μ_i per image, no game-specific δ_{g,i}. All games
  share one convention. Baseline: can it explain listener choices at all?
- **No-pooling**: independent m_{k,i} per (game, image), no shared μ_i prior. No
  cross-game generalization. Upper bound on per-game fit, but can't generalize.
- **Fixed sigma**: remove accuracy-dependent emission (σ²_t = constant). Does the
  accuracy-weighting actually matter for convention fit quality?
- **No hierarchical β**: single shared β across all listeners instead of per-listener
  LogNormal hierarchy. Tests whether listener heterogeneity matters.
- **CLIP variant**: `openai/clip-vit-base-patch32` is the current default. Compare with
  `openai/clip-vit-large-patch14` or a language-only text encoder (e.g. sentence-transformers)
  to check sensitivity to embedding space.

### Additions / extensions (from minimal_model_schema.md out-of-scope list)

- **M1 — Learned projection P**: low-rank matrix P ∈ R^{d×D} applied to CLIP embeddings
  before computing cosine similarities. Addresses CLIP's possible misalignment between
  image and text spaces. Add if raw CLIP listener fit is poor. d is a hyperparameter.
- **M2 — Memory decay**: down-weight older observations in the convention update
  (β_memory ∈ [0,1]; paper uses 0.8). Currently β=1 (uniform). Add if earlier rounds
  appear empirically less predictive. Applies inside sequential_kalman_means.
- **M4 — Player-specific effects**: per-player deviation δ_{p,i} in the convention.
  Would let individual players drift from the game-level convention. Complex but may be
  needed if there's high within-game heterogeneity.
- **M5 — Repulsive failure likelihood**: add an anti-evidence term so failed utterances
  actively push m_{g,i} away (not just fail to update it). More faithful to RSA; harder
  to implement in the Gaussian emission framework.
- **OQ11 — Pragmatic L_1 listener**: replace literal L_0 with a listener that reasons
  about the speaker. More principled but requires a speaker model (see below).
- **Speaker generation**: given a convention m_{g,i}, generate predicted utterances via
  proposal (language model) + rerank (speaker utility). Would enable direct utterance
  prediction, not just convention tracking. Deferred; requires LM integration.

### Open decision points (from plan.md / model_components.md)

- Multi-listener trials: each (trial, listener) is currently an independent data point.
  Should listener choices within the same trial be modeled jointly (e.g. correlated noise)?
- Image filter threshold: top_n=12 for early iteration. Once eval is stable, test whether
  expanding the image set changes results.
- Whether C3 (hierarchical generalization, cross-game Θ) is identifiable from RefBank given
  that each game has the same 12 images — effectively partial pooling is already doing this.

## Completed
- [x] Fix sigma collapse (code side): changed sigma_min and sigma_delta priors from
  HalfNormal(1.0) to LogNormal(log(0.3), 0.5) / LogNormal(log(0.7), 0.5) in both
  speaker and listener convention models. Regression test added. 68 unit tests pass.
  **Still needs cluster run to confirm empirically.**
- [x] Visualisation improvements (2026-05-05): bigger figures (1200×900 / 1100×850),
  white background, more legend entries (correct/incorrect/convention/prototype symbols),
  rep-to-rep arrows via Plotly annotations (game dropdown triggers annotation update),
  rep 1 halo ring (marker.line.width=3 for rep_num=1 only). 18 viz tests pass.
- [x] `script/visualize.py` — generates interactive HTML plots from a results dir;
  `code/analysis/visualization.py` — `compute_tsne_coords`, `plot_overview`, `plot_game`;
  9 tests pass; t-SNE cached to `tsne_coords.parquet` after first run
- [x] Fix `sigma_max < sigma_min` ordering bug (job 23807580) — reparametrised to
  `sigma_max = sigma_min + sigma_delta`; 2 new tests; 58 unit tests total
- [x] Second GPU run (job 23851534): ordering fix confirmed, but sigma collapse emerged
- [x] Full GPU pipeline run on cluster (job 23807580): step sizes decrease (0.048→0.009),
  displacement larger after failure (0.49 vs 0.30) — qualitative checks pass
- [x] Data preloaded into repo; cluster runs with `--no-fetch` (no Redivis on cluster)
- [x] `script/configs/fetch_only.toml` added for refreshing preloaded data on laptop
- [x] `run_svi` replaced Python for-loop with `jax.lax.scan` (NaN injection via jnp.where;
  post-hoc NaN check); 48 unit tests pass, suite runs in ~8s
- [x] `save_listener_fit` / `load_listener_fit` in `literal_listener.py`
- [x] `save_convention_fit` / `load_convention_fit` in `conventional_speaker.py`
- [x] `script/cluster_job.sh` — SLURM job script (sbatch from project root)
- [x] `script/run_pipeline.py` updated with `--output-dir`, per-stage timing, saves fits + CSVs
- [x] Steps 1–16 — all unit tests pass (cache, convention_model, loader, pipeline, svi)
- [x] Steps 17–18 — `predictions.py` with `sequential_kalman_means`, `step_sizes_over_reps`,
  `semantic_displacement_after_error`; `tests/unit/test_predictions.py` all 5 pass
- [x] Steps 20–21 — `semantic_displacement_after_error`, `filter_images` threshold mode
- [x] `redivis_client.py` and `clip_encoder.py` implemented (SVG rasterization supported)
- [x] `script/run_pipeline.py` — end-to-end pipeline script with `--config` argument
- [x] `script/configs/` — three config files: `quick_cpu`, `gpu_test`, `full_gpu`
- [x] `tests/integration/test_int_real_data.py` — regression tests on real data fixtures
  (200-step fits; skipped when fixtures absent)
- [x] `plan/compute_plan.md` — computational footprint analysis and cluster setup notes

## Known issues / bugs

- **Sigma collapse — fix deployed, not yet validated** (job 23851534): sigma_min≈1e-5,
  sigma_max≈3e-5. Fix: changed priors to LogNormal (see Completed above). A new cluster
  run is needed to confirm the fix holds on real data.

## Decisions

- `sequential_kalman_means` uses binary correctness (selected_idx == target_idx) for s_t
  in both speaker and listener roles. Schema said speaker should use `compute_success_probs`
  (L_0 score), but that requires `ListenerFit` which isn't in `ConventionFit`. Binary
  correctness is a reasonable proxy and keeps the function self-contained.
- `semantic_displacement_after_error` takes a `fit` param for API consistency but doesn't
  use it (displacement is data-level; prev_success is binary from batch).
- The Python for-loop in `run_svi` is slow at D=512 scale. This caused the laptop to hang
  at 1000 convention steps. `lax.scan` is the fix — see plan/compute_plan.md.
- Fixture saving in `run_pipeline.py` only triggers when `n_games == 5` (the default for
  quick_cpu and gpu_test). Full-data runs don't overwrite the 5-game fixtures.
