# Scratchpad

## Current position: pipeline runs end-to-end; ready for cluster

All 44 unit tests pass. The full pipeline (fetch → embed → fit listener → fit convention →
analysis) runs on real Redivis data via `script/run_pipeline.py`.

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

## Open questions / next steps

- [ ] Implement `jax.lax.scan` in `run_svi` to replace the Python for-loop.
  This is the key change needed before running full fits efficiently on GPU.
  (Python loop: ~30–60 min on laptop for full step counts; `lax.scan` + GPU: ~2–4 min)
- [ ] Run full pipeline on cluster GPU and check that loss converges and analysis
  outputs make sense (step sizes decrease, displacement larger after failure).
- [ ] Confirm: full `hawkins2020_characterizing_cued` dataset size (n_games, n_images,
  n_trials) — needed to estimate GPU RAM and time precisely.
- [ ] Should `fit_convention` save intermediate checkpoints in case a cluster job is killed?

## Completed
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
