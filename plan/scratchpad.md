# Scratchpad

## Current position: lax.scan in; cluster script ready

All 46 unit tests pass. The full pipeline (fetch → embed → fit listener → fit convention →
analysis) runs on real Redivis data via `script/run_pipeline.py`.

`run_svi` now uses `jax.lax.scan` instead of a Python for-loop — unit tests run in ~8s
instead of ~61s, GPU fits should be ~10-100× faster. `script/cluster_job.sh` is the SLURM
submission script. Fits and analysis CSVs can be saved with `--output-dir`.

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

**Step 1 — on the laptop** (once, to populate `data/`):
```bash
.venv/bin/python script/run_pipeline.py --config full_gpu
# saves data/hawkins2020_characterizing_cued_joined.parquet + data/embeddings/*.npz
```

**Step 2 — rsync to cluster:**
```bash
rsync -av data/ cluster:/path/to/project/data/
```

**Step 3 — on the cluster** (no Redivis, no CLIP encoder needed):
```bash
sbatch script/cluster_job.sh
# uses --no-fetch; raises immediately with a clear message if any embeddings are missing
```

Results go to `results/run_<JOBID>/`: `listener_fit.npz`, `convention_fit.npz`,
`step_sizes.csv`, `displacement.csv`.

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

- [ ] Update `script/cluster_job.sh` and pipeline to skip Redivis fetch and use
  cached data/embeddings directly (Redivis not needed on cluster).
- [ ] Run full pipeline on cluster GPU and check that loss converges and analysis
  outputs make sense (step sizes decrease, displacement larger after failure).
- [ ] Confirm: full `hawkins2020_characterizing_cued` dataset size (n_games, n_images,
  n_trials) — needed to estimate GPU RAM and time precisely.
- [ ] Should `fit_convention` save intermediate checkpoints in case a cluster job is killed?

## Completed
- [x] `run_svi` replaced Python for-loop with `jax.lax.scan` (NaN injection via jnp.where;
  post-hoc NaN check); 46 unit tests pass, suite runs in ~8s
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

## Known issues / decisions

- `sequential_kalman_means` uses binary correctness (selected_idx == target_idx) for s_t
  in both speaker and listener roles. Schema said speaker should use `compute_success_probs`
  (L_0 score), but that requires `ListenerFit` which isn't in `ConventionFit`. Binary
  correctness is a reasonable proxy and keeps the function self-contained.
  Revisit if this causes INT-2 to fail.
- `semantic_displacement_after_error` takes a `fit` param for API consistency but doesn't
  use it (displacement is data-level; prev_success is binary from batch).
- The Python for-loop in `run_svi` is slow at D=512 scale. This caused the laptop to hang
  at 1000 convention steps. `lax.scan` is the fix — see plan/compute_plan.md.
- Fixture saving in `run_pipeline.py` only triggers when `n_games == 5` (the default for
  quick_cpu and gpu_test). Full-data runs don't overwrite the 5-game fixtures.
