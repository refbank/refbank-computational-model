# Scratchpad

## Current position: visualisation done; sigma collapse to fix next

All 58 unit tests pass. The full pipeline runs end-to-end and has been validated on the
cluster GPU. Visualisation tooling is complete. Next priority: fix sigma collapse.

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

- [ ] Fix sigma collapse — see Known Issues below. **Do this next.**
- [ ] Build predicted vs observed accuracy analysis (calibration + learning curve by rep_num).
- [ ] Utterance scoring: use the convention model's emission to score actual vs counterfactual
  alternative utterances (e.g. utterances from other games for the same image). Useful for
  checking whether within-game conventions are more distinctive than across-game utterances.
- [ ] Should `fit_convention` save intermediate checkpoints in case a cluster job is killed?

## Completed
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

- **Sigma collapse** (job 23851534, after ordering fix): sigma_min≈1e-5,
  sigma_max≈3e-5. The emission variance collapses to a near-point-mass, meaning the
  model treats each utterance as essentially identical to the convention point. The
  qualitative results still look right (step sizes decrease 0.25→0.06, displacement
  larger after failure) but the sigma values are not meaningful.
  Likely cause: the optimizer finds a degenerate solution where m_{g,i} drifts to fit
  each utterance exactly, so minimal emission variance is needed. Possible fixes:
  - Stronger priors on sigma_min / sigma_delta (e.g. HalfNormal with larger scale,
    or a LogNormal prior with a positive mean)
  - A prior that penalises sigma near zero (e.g. InverseGamma)
  - Fixing sigma_min to a reasonable constant and only fitting sigma_max
  - L2-normalising m_{g,i} after each update to keep it on the unit sphere
  Note: the `sigma_max < sigma_min` ordering issue from job 23807580 is now fixed.

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
