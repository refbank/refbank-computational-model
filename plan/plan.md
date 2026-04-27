# Computational Model of Iterated Reference Games — Plan

## What we're building

A Bayesian model that fits to RefBank data to study how communication conventions emerge over repeated reference game interactions. The key extension over the CHAI model (Hawkins et al.) is replacing the discrete lexicon with continuous CLIP semantic embeddings, so the model works with open-vocabulary, free-text utterances.

The model is designed so that the phenomena of interest — step-size decrease over rounds, direction change after errors, within-game convergence, between-game divergence — emerge from Bayesian updating rather than being hard-coded.

**Phase 1 scope:** Fit listener choices and track convention evolution from observed utterances. No text generation.  
**Stretch goal:** Speaker model that generates/predicts utterances via proposal + rerank.

---

## Data

RefBank via Redivis Python API. Key tables (joined per trial):

| Field | Source | Notes |
|---|---|---|
| `game_id`, `trial_num`, `rep_num` | trials | rep_num = repetition of this image in this game |
| `option_set` | trials | list of 12 image IDs |
| `target_image_id` | trials | correct referent |
| `utterance` | messages | describer's text (multi-message trials concatenated in order) |
| `selected_image_id` | choices | listener's pick |
| `listener_id` | choices | multiple listeners → one row per (trial, listener) |

**Image filter (runtime parameter, two modes):**
- `top_n` — keep only games where all 12 images are among the N most-used images globally
- `threshold` — keep games where all images appear in at least K games

Start with `top_n, n=12` for iteration; expand later.

---

## Model

### Stage 1 — Listener choice model

For each trial: utterance embedding `z(u)` and 12 candidate image embeddings `x(j)`, optionally projected via a learnable low-rank matrix `P ∈ R^{d×D}`.

```
P(c | u, O, ℓ) = softmax_j( β_ℓ · cos(Pz(u), Px(j)) )
β_ℓ ~ LogNormal(μ_β, σ_β)    [per-listener calibration, hierarchical]
```

Fit via NumPyro SVI. Outputs: `P`, `{β_ℓ}`.

After fitting, compute per-trial continuous success probability:
```
s_t = P(target | u_t, O_t, ℓ_t)    [from the fitted listener model]
```

### Stage 2 — Convention model

Hierarchical latent convention for each (game, image) pair:

```
μ_i ~ N(0, I)                          [global prototype per image]
δ_{g,i} ~ N(0, σ_game² · I)           [game-specific deviation]
m_{g,i} = μ_i + δ_{g,i}               [convention in projected space]
```

Accuracy-weighted emission — the key mechanism for emergent learning dynamics:

```
σ²_t = σ²_min + (σ²_max - σ²_min) / clip(s_t, ε, 1)
Pz(u_t) ~ N(m_{g,i_t}, σ²_t · I)
```

When `s_t` is high (success), emission is tight → posterior concentrates fast → step sizes shrink.  
When `s_t` is low (failure), emission is wide → posterior barely moves toward the failed utterance → next utterance drifts elsewhere.

Fit via NumPyro SVI. Outputs: `{μ_i}`, `{δ_{g,i}}` posteriors, `σ_min`, `σ_max`.

### Emergent predictions (checked post-fit, not imposed)

- **Step-size decrease**: `||E[m_{g,i}]^(t) - E[m_{g,i}]^(t-1)||` decreases over `rep_num`
- **Direction change after error**: larger semantic displacement when `s_t < 0.5`
- **Within-game convergence**: consecutive utterance embeddings get closer
- **Between-game divergence**: `Var[δ_{g,i}]` across games is nonzero

---

## Implementation

### Project structure

```
refbank/
  data/
    redivis_client.py    # only file that imports redivis (seam for testing)
    loader.py            # join tables, filter images, concat multi-msg utterances
    pipeline.py          # flat DataFrame → TrialBatch (JAX arrays)
  embeddings/
    clip_encoder.py      # CLIPEncoder.encode_images / encode_texts
    cache.py             # .npz disk cache; handles partial misses
  models/
    listener.py          # NumPyro Stage 1
    convention.py        # NumPyro Stage 2
  inference/
    svi.py               # shared SVI training loop (Adam, ELBO, NaN detection)
  analysis/
    predictions.py       # step_sizes_over_reps, semantic_displacement_after_error
tests/
  conftest.py            # synthetic fixtures (no real API calls in unit tests)
  integration/           # @pytest.mark.slow
  unit/
pyproject.toml
```

### Central data structure

```python
@dataclass(frozen=True)
class TrialBatch:
    utterance_emb:  jnp.ndarray  # (N, D) — CLIP text, L2-normalized
    option_embs:    jnp.ndarray  # (N, 12, D) — CLIP image, L2-normalized
    target_idx:     jnp.ndarray  # (N,) int — index of target in option_embs
    selected_idx:   jnp.ndarray  # (N,) int — index of listener's pick
    listener_ids:   jnp.ndarray  # (N,) int
    game_ids:       jnp.ndarray  # (N,) int
    image_ids:      jnp.ndarray  # (N,) int — global target image index
    rep_num:        jnp.ndarray  # (N,) int
    n_listeners, n_games, n_images: int
```

`build_trial_batch` raises `ValueError` if any option set ≠ 12 images, and `KeyError` if any embedding is missing. No silent fallbacks.

### Data pipeline

```
Redivis API
  → trials_df, messages_df, choices_df
  → loader.join_tables()          one row per (trial, listener)
  → loader.filter_images()        top_n or threshold
  → CLIPEncoder + cache           embed all unique images and utterances
  → pipeline.build_trial_batch()  → TrialBatch
```

### Edge cases (all fail loudly)

| Case | Handling |
|---|---|
| Multi-message trials | Sort by `order`, join with space |
| Multiple listeners | One row per (trial_id, listener_id) |
| Missing listener choice | Drop row, `logging.warning` with count |
| Option set ≠ 12 | `ValueError` |
| Missing embedding | `KeyError` with the specific ID |
| NaN SVI loss after step 100 | `RuntimeError` |

### Implementation order (TDD: test first, then implement)

| Step | Test written | Code written |
|---|---|---|
| 1 | INT-1 stub (fails: no TrialBatch) | — |
| 2 | unit: join_tables | loader.join_tables |
| 3 | unit: filter_images | loader.filter_images |
| 4 | unit: build_trial_batch | pipeline.py + TrialBatch |
| 5 | unit: cache round-trip | embeddings/cache.py |
| 6 | unit: embeddings_for_ids | cache.py (partial miss, mock encoder) |
| 7 | unit: listener model trace | models/listener.py |
| 8 | unit: compute_listener_success_probs | listener.py inference fn |
| 9 | INT-1 green: β recovery on synthetic data | inference/svi.py |
| 10 | unit: compute_sigma2_t | standalone fn in convention.py |
| 11 | unit: convention model trace | models/convention.py |
| 12 | INT-2 green: step sizes decrease | convention model complete |
| 13 | unit: step_sizes_over_reps | analysis/predictions.py |
| 14 | unit: semantic_displacement_after_error | analysis/predictions.py |

### Integration test specs

**INT-1** (`@pytest.mark.slow`): 2 listeners (true β = [2.0, 5.0]), 200 synthetic trials, 12 images, 16-dim embeddings. Target cosine ≈ 0.9, distractors ≈ 0.1–0.3. Choices sampled from true softmax. 5000 SVI steps. Assert: β order preserved, each within 1.5 of true value.

**INT-2** (`@pytest.mark.slow`): 6 games × 3 images × 6 reps, 8-dim space. Utterances from N(μ_i + δ_{g,i}, 0.01I). s_t = 0.9 fixed. 8000 SVI steps. Assert: step_sizes[-1] < step_sizes[0] for ≥80% of (game, image) pairs.

---

## Open questions / decisions to revisit

- **Projection P**: start with `d=None` (no projection, use CLIP space directly) and add only if listener model fit is poor
- **Image filter threshold**: `top_n=12` for early iteration; revisit once we know image coverage across datasets
- **Multi-listener trials**: currently each (trial, listener) is an independent data point — is this right, or should listener choices within the same trial be modeled jointly?
- **Convention model plates**: `delta_gi` over all (n_games × n_images) pairs may be large; confirm NumPyro handles sparse indexing efficiently
- **CLIP model variant**: `openai/clip-vit-base-patch32` as default — may want to compare with larger variants or specialized models
