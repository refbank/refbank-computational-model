# Compute Plan: Cluster Running

This document covers the computational footprint of the model and how to run
it on a cluster efficiently.

---

## Why the laptop runs are slow / crash

The current `run_svi` implementation uses a **Python for-loop** over SVI steps.
Each loop iteration dispatches one gradient update to JAX, which:

1. Triggers a Python→XLA kernel call per step (overhead accumulates over 1000s of steps).
2. Forces the XLA JIT to re-enter Python between steps (no loop fusion).
3. At D=512 CLIP dimensionality, the gradient computation is large enough that
   1000 steps takes ~10–30 minutes on a laptop CPU, pinning all cores.

### Parameter count (per convention model fit)

| Parameter | Shape | Size (float32) |
|---|---|---|
| `mu_i_loc`, `mu_i_scale` | 2 × n\_images × D | 2 × n\_images × 512 |
| `delta_gi_loc`, `delta_gi_scale` | 2 × n\_images × n\_games × D | 2 × n\_images × n\_games × 512 |
| sigma scalars (×3) | 6 scalars | negligible |

For the 5-game fixture (est. ~60 images): `delta_gi` ≈ 2 × 60 × 5 × 512 × 4 bytes ≈ **1.2 MB**.
For a full dataset (est. 200 games, 200 images): `delta_gi` ≈ 2 × 200 × 200 × 512 × 4 bytes ≈ **160 MB**.
Memory is not the problem. CPU time is.

---

## Key code change: replace Python loop with `jax.lax.scan`

The fix is to compile the entire SVI loop into a single XLA program using
`jax.lax.scan`. This gives **10–100× speedup** on CPU and enables GPU
parallelism without any algorithm changes.

```python
# Current (slow)
for i in range(n_steps):
    state, loss = svi.update(state, *model_args)

# Fast alternative using lax.scan
def step(state, _):
    state, loss = svi.update(state, *model_args)
    return state, loss

state, losses = jax.lax.scan(step, state, None, length=n_steps)
```

Tradeoffs:
- NaN early-stopping must become post-hoc: check `jnp.any(jnp.isnan(losses[100:]))`
  after the scan completes rather than raising inside the loop.
- All model args must be JAX-compatible arrays (no Python objects in `model_args`).
  `TrialBatch` already satisfies this.
- XLA compilation happens once on the first call, then subsequent calls (e.g.,
  re-fitting with different seeds) are fast.

---

## Estimated runtimes

Benchmarks below are rough estimates based on model scale.

| Setting | Listener (5000 steps) | Convention × 2 (8000 steps each) | Total |
|---|---|---|---|
| Laptop CPU, Python loop | ~5–15 min | ~20–60 min | ~25–75 min |
| Laptop CPU, lax.scan | ~1–3 min | ~5–15 min | ~6–18 min |
| Cluster CPU node (32 cores) | ~1–3 min | ~5–15 min | ~6–18 min (CPU nodes don't help much — JAX doesn't parallelize across cores for this workload) |
| Cluster GPU (A100 or V100) | <30 s | ~1–3 min | ~2–4 min |

For real fits on the full dataset (all games), GPU is strongly recommended.

---

## Cluster setup (SLURM)

Assuming a SLURM cluster with GPU nodes:

### Environment

```bash
module load cuda/12.x
pip install jax[cuda12] numpyro transformers torch torchvision
```

Or build a Singularity/Apptainer container from the project `.venv` if the
cluster has restricted internet access.

### Job structure

The pipeline has two sequential stages that cannot be parallelized:

```
Stage 1 (listener fit)  →  Stage 2 (convention fit × 2, independent)
                                         └── speaker model
                                         └── listener model  (can overlap)
```

Simplest approach: one SLURM job, one GPU, ~5 minutes end-to-end (with `lax.scan`).

If fitting multiple datasets or running sensitivity analysis (multiple seeds):
use a SLURM array job — one task per (dataset, seed) pair.

### Example job script

```bash
#!/bin/bash
#SBATCH --job-name=refbank-fit
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=logs/%j.out

source .venv/bin/activate
python script/run_pipeline.py --listener-steps 5000 --convention-steps 8000
```

---

## Recommended implementation order

1. **Now**: Use the 200-step sanity check in `run_pipeline.py` to confirm the
   pipeline runs end-to-end on real data.
2. **Before real fits**: Update `run_svi` to use `jax.lax.scan` (dropping
   in-loop NaN early-stopping, keeping post-hoc check).
3. **Real fits**: Run on cluster GPU with full step counts. Save `ConventionFit`
   and `ListenerFit` to disk (e.g., `.npz`) for analysis without re-fitting.
4. **Analysis**: Load saved fits locally for `step_sizes_over_reps` and
   `semantic_displacement_after_error` — these are fast numpy operations.

---

## Open questions

- Full dataset size: how many games and unique images in
  `hawkins2020_characterizing_cued`? This determines whether `delta_gi` fits
  comfortably in 16–40 GB GPU RAM (it almost certainly will).
- Cluster access: which cluster? (Sherlock, Farmshare, etc.)
  Some require specific CUDA module versions for JAX.
- Should `fit_convention` save intermediate checkpoints (every N steps) in
  case the job is killed?
