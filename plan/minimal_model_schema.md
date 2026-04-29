# Minimal Model Schema

Concrete specification for the first implementation target: modules, interfaces, tests, and
implementation order. This document drives coding; `model.md` is the mathematical reference.

---

## Scope

**In scope:**
- Stage 1: L_0 listener model with hierarchical per-listener sharpness $\beta_\ell$
- Stage 2: Convention model with separate speaker/listener posteriors, accuracy-weighted
  emission, hierarchical game-image convention (partial-pooling)
- Analysis: sequential step-size and displacement checks against emergent predictions

**Explicitly out of scope for minimal model:**
- Projection P (M1) — use raw CLIP space throughout
- Memory decay (M2) — $\gamma = 1$ (uniform weighting)
- Player-specific effects (M4) — no $\delta_{p,i}$
- Repulsive failure likelihood (M5) — wide emission only, no anti-evidence term
- Pragmatic $L_1$ listener (OQ11) — literal $L_0$ only
- Speaker generation — no proposal+rerank, no speaker utility fitting
- Model variants (complete-pooling, no-pooling) — partial-pooling only in first pass

**$\mu_i$ initialization:** R1 average — mean of round-1 utterance embeddings across games
for image $i$. No cross-modal projector needed. This is computed from the data before
fitting Stage 2, not a model parameter.

**Posterior representation:** Two separate NumPyro models (speaker-side, listener-side)
fitted independently with SVI. No shared posterior, no combination. At prediction time,
speaker role uses $p_S$, listener role uses $p_L$.

---

## Project Layout

```
code/
  prep_data/
    redivis_client.py    # sole file that imports redivis — seam for testing
    loader.py            # join_tables, filter_images
    pipeline.py          # build_trial_batch → TrialBatch
  embeddings/
    clip_encoder.py      # CLIPEncoder: encode_images, encode_texts
    cache.py             # EmbeddingCache: disk-backed .npz; embeddings_for_ids
  models/
    literal_listener.py          # L_0 NumPyro model, fit, score
    conventional_listener.py	 # speaker/listener convention models, fit, sequential analysis
    conventional_speaker.py 
    pragmatic_listener.py     
  inference/
    svi.py               # run_svi: shared training loop
  analysis/
    predictions.py       # step_sizes_over_reps, semantic_displacement_after_error
tests/
  conftest.py            # shared synthetic fixtures — no real API calls anywhere
  unit/
    test_loader.py
    test_pipeline.py
    test_cache.py
    test_listener_model.py
    test_convention_model.py
    test_svi.py
    test_predictions.py
  integration/           # @pytest.mark.slow
    test_int_listener.py
    test_int_convention.py
pyproject.toml
```

---

## Central Data Structure

```python
@dataclass(frozen=True)
class TrialBatch:
    utterance_emb:  jnp.ndarray  # (N, D) CLIP text, L2-normalized
    option_embs:    jnp.ndarray  # (N, 12, D) CLIP image, L2-normalized
    target_idx:     jnp.ndarray  # (N,) int — index of target in option_embs[n]
    selected_idx:   jnp.ndarray  # (N,) int — index of listener's pick in option_embs[n]
    listener_ids:   jnp.ndarray  # (N,) int in [0, n_listeners)
    game_ids:       jnp.ndarray  # (N,) int in [0, n_games)
    image_ids:      jnp.ndarray  # (N,) int in [0, n_images) — global target image index
    rep_num:        jnp.ndarray  # (N,) int — repetition count for this image in this game
    n_listeners:    int
    n_games:        int
    n_images:       int
```

One row per (trial, listener). Multi-listener trials share `utterance_emb`, `target_idx`,
`game_ids`, `image_ids`, `rep_num` but have different `listener_ids` and `selected_idx`.

---

## Module Specs

### prep_data/redivis_client.py

```python
def fetch_tables(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (trials_df, messages_df, choices_df)."""
```

Never called from tests. Only file that imports `redivis`.

---

### prep_data/loader.py

```python
def join_tables(
    trials_df: pd.DataFrame,
    messages_df: pd.DataFrame,
    choices_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns one row per (trial_id, listener_id) with columns:
      game_id, trial_num, rep_num, image_id, option_set (list[str]),
      utterance (str), selected_image_id (str), listener_id (str)

    Multi-message trials: sort by message order ascending, join with ' '.
    Missing listener choices: drop row, emit logging.warning with total count.
    """

def filter_images(
    df: pd.DataFrame,
    mode: Literal["top_n", "threshold"],
    n: int,
) -> pd.DataFrame:
    """
    mode="top_n": keep only games where every image in option_set is among
        the n most-frequent images globally (by count across all games).
    mode="threshold": keep only games where every image appears in >= n games.
    Raises ValueError if no games survive the filter.
    """
```

---

### prep_data/pipeline.py

```python
def build_trial_batch(
    df: pd.DataFrame,
    image_embeddings: dict[str, np.ndarray],    # image_id → L2-normalized (D,)
    utterance_embeddings: dict[str, np.ndarray], # utterance text → L2-normalized (D,)
) -> TrialBatch:
    """
    Raises ValueError if any option_set has length != 12.
    Raises KeyError("<id> not found in embeddings") for any missing image or utterance.
    Encodes game_ids, listener_ids, image_ids as contiguous integers starting from 0.
    """
```

---

### embeddings/clip_encoder.py

```python
class CLIPEncoder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None: ...

    def encode_images(self, image_paths: list[str]) -> np.ndarray:
        """Returns (N, D) L2-normalized. Raises FileNotFoundError for missing paths."""

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Returns (N, D) L2-normalized."""
```

---

### embeddings/cache.py

```python
class EmbeddingCache:
    def __init__(self, path: str) -> None:
        """Loads .npz from path if it exists; starts empty otherwise."""

    def get(self, keys: list[str]) -> tuple[dict[str, np.ndarray], list[str]]:
        """Returns (found: {key → array}, missing: [key])."""

    def put(self, embeddings: dict[str, np.ndarray]) -> None:
        """Merges new entries into cache and saves to disk atomically."""

def embeddings_for_ids(
    ids: list[str],
    cache: EmbeddingCache,
    encoder_fn: Callable[[list[str]], np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Returns embeddings for all ids. Calls encoder_fn only for ids missing from cache,
    then puts the new embeddings into the cache.
    """
```

---

### models/literal_listener.py

```python
def literal_listener_model(batch: TrialBatch) -> None:
    """
    NumPyro model. Plates over listeners for beta_l ~ LogNormal(mu_beta, sigma_beta)
    and over trials for the 12-way choice likelihood:
      P(selected_idx | utterance_emb, option_embs, beta_l) = softmax(beta_l * cos_sims)
    """

def literal_listener_guide(batch: TrialBatch) -> None:
    """AutoNormal guide for literal_listener_model."""

@dataclass
class ListenerFit:
    beta_loc:   np.ndarray  # (n_listeners,) posterior mean of log beta_l
    beta_scale: np.ndarray  # (n_listeners,) posterior std of log beta_l
    mu_beta:    float       # fitted hyperprior mean
    sigma_beta: float       # fitted hyperprior std

def fit_listener(
    batch: TrialBatch,
    n_steps: int = 5000,
    lr: float = 0.01,
    seed: int = 0,
) -> ListenerFit:
    """Raises RuntimeError("NaN loss at step {i}") if loss is NaN after step 100."""

def compute_success_probs(fit: ListenerFit, batch: TrialBatch) -> np.ndarray:
    """
    Returns (N,) array of s_t = L_0(target_idx | utterance_emb, option_embs, beta_l)
    using posterior mean beta_l for each listener. Values in (0, 1).
    """
```

---

### models/conventional_speaker.py

```python
def compute_sigma2_t(
    s_t: jnp.ndarray,  # (N,) success probabilities
    sigma_min: float,
    sigma_max: float,
    eps: float = 0.01,
) -> jnp.ndarray:
    """
    Returns (N,) accuracy-dependent emission variance:
      sigma2_t = sigma_min + (sigma_max - sigma_min) / clip(s_t, eps, 1)
    """

def compute_r1_average(batch: TrialBatch) -> np.ndarray:
    """
    Returns (n_images, D) mean of utterance_emb for rep_num == 1 trials,
    grouped by image_ids. Used to initialize mu_i.
    Raises ValueError if any image has no rep_num == 1 trials.
    """

def speaker_convention_model(
    batch: TrialBatch,
    s_t: jnp.ndarray,      # (N,) model-implied L_0 success probabilities
    mu_i_init: np.ndarray,  # (n_images, D) initial global prototypes
) -> None:
    """
    NumPyro model. Latents:
      sigma_game, sigma_min, sigma_max  (positive scalars)
      mu_i  (n_images, D)  — community prototype, initialized near mu_i_init
      delta_gi  (n_games, n_images, D) ~ N(0, sigma_game^2 * I)
    Convention: m_{g,i} = mu_i[image_ids] + delta_gi[game_ids, image_ids]
    Likelihood: utterance_emb[t] ~ N(m_{g,i_t}, sigma2_t * I)
      where sigma2_t = compute_sigma2_t(s_t, sigma_min, sigma_max)
    """

@dataclass
class ConventionFit:
    speaker_m_loc:   np.ndarray  # (n_games, n_images, D) posterior mean m_{g,i}
    speaker_m_scale: np.ndarray  # (n_games, n_images) posterior std (scalar per pair)
    listener_m_loc:  np.ndarray  # (n_games, n_images, D)
    listener_m_scale: np.ndarray # (n_games, n_images)
    mu_i:            np.ndarray  # (n_images, D) fitted global prototypes
    sigma_game:      float
    sigma_min:       float
    sigma_max:       float

def fit_convention(
    batch: TrialBatch,
    listener_fit: ListenerFit,
    mu_i_init: np.ndarray,   # (n_images, D) from compute_r1_average
    n_steps: int = 8000,
    lr: float = 0.01,
    seed: int = 0,
) -> ConventionFit:
    """
    Fits speaker_convention_model and listener_convention_model independently via SVI.
    s_t for speaker: compute_success_probs(listener_fit, batch)
    s_t for listener: (batch.selected_idx == batch.target_idx).astype(float)
    Raises RuntimeError on NaN loss in either fit.
    """
```

---

### models/conventional_listener.py

```python
def listener_convention_model(
    batch: TrialBatch,
    s_t_binary: jnp.ndarray,  # (N,) binary 1[selected_idx == target_idx]
    mu_i_init: np.ndarray,    # (n_images, D)
) -> None:
    """
    NumPyro model. Same structure as speaker_convention_model in conventional_speaker.py.
    s_t is binary correctness rather than model-implied L_0 score.
    Imports compute_sigma2_t and ConventionFit from conventional_speaker.
    """
```

---

### models/pragmatic_listener.py

Out of scope for the minimal model. Placeholder for the $L_1$ pragmatic listener (OQ11).

---

### inference/svi.py

```python
@dataclass
class SVIResult:
    params: dict          # final SVI parameter dict (from svi.get_params())
    losses: np.ndarray    # (n_steps,) ELBO per step

def run_svi(
    model: Callable,
    guide: Callable,
    model_args: tuple,
    n_steps: int,
    lr: float = 0.01,
    seed: int = 0,
) -> SVIResult:
    """
    Adam optimizer + ELBO loss via NumPyro SVI.
    Raises RuntimeError("NaN loss at step {i}") if loss is NaN at any step after step 100.
    """
```

---

### analysis/predictions.py

**Design note:** The batch SVI fit produces a single $m_{g,i}$ per (game, image), not a
per-round trajectory. Step sizes are computed by replaying sequential Kalman updates
using the fitted $\sigma_\text{min}$, $\sigma_\text{max}$ from `ConventionFit` and the
observed $s_t$ values. This gives the posterior mean position after each rep and lets us
measure how much it moves.

```python
def sequential_kalman_means(
    fit: ConventionFit,
    batch: TrialBatch,
    role: Literal["speaker", "listener"],
) -> dict[tuple[int, int], np.ndarray]:
    """
    For each (game_id, image_id) pair, returns array of shape (n_reps, D):
    the posterior mean m after processing each rep in order.

    Uses fit.sigma_min, fit.sigma_max to compute sigma2_t per rep.
    For role="speaker": s_t from compute_success_probs (L_0 score).
    For role="listener": s_t from binary correctness.
    Prior: N(fit.mu_i[image_id], fit.sigma_game^2 * I).
    Closed-form Gaussian update at each rep (no resampling).
    """

def step_sizes_over_reps(
    fit: ConventionFit,
    batch: TrialBatch,
    role: Literal["speaker", "listener"] = "speaker",
) -> pd.DataFrame:
    """
    Returns DataFrame with columns: game_id, image_id, rep_num, step_size.
    step_size[t] = ||E[m]^(t) - E[m]^(t-1)||_2 using sequential Kalman means.
    Requires >= 2 reps per (game, image); pairs with fewer are omitted.
    """

def semantic_displacement_after_error(
    fit: ConventionFit,
    batch: TrialBatch,
    success_threshold: float = 0.5,
    role: Literal["speaker", "listener"] = "speaker",
) -> pd.DataFrame:
    """
    Returns DataFrame with columns: game_id, image_id, rep_num, displacement, prev_success.
    displacement[t] = ||z(u_t) - z(u_{t-1})||_2 (data-level utterance embedding distance).
    prev_success = s_{t-1} >= success_threshold.
    Requires >= 2 reps per (game, image).
    """
```

---

## Parameters

| Parameter | Module | Role | How estimated |
|---|---|---|---|
| $\mu_\beta$, $\sigma_\beta$ | literal_listener.py | Hyperprior on $\log \beta_\ell$ | Fitted Stage 1 |
| $\beta_\ell$ | literal_listener.py | Per-listener inverse temperature | Fitted Stage 1, hierarchical |
| $\mu_i$ | conventional_speaker.py | Global prototype per image | Fitted Stage 2 (init from R1 average) |
| $\sigma_\text{game}$ | conventional_speaker.py | Spread of game deviations around $\mu_i$ | Fitted Stage 2 |
| $\sigma_\text{min}$ | conventional_speaker.py | Emission variance at $s_t = 1$ | Fitted Stage 2 |
| $\sigma_\text{max}$ | conventional_speaker.py | Emission variance at $s_t = 0$ | Fitted Stage 2 |
| $\varepsilon$ | conventional_speaker.py | Floor on $s_t$ in $\sigma^2_t$ formula | Fixed at 0.01 |

---

## Tests

### Unit tests

**test_loader.py**

- `test_join_tables_concats_messages_in_order`: two messages for one trial (orders 2, 1)
  → utterance = "msg1 msg2" (sorted ascending by order, joined with space)
- `test_join_tables_drops_missing_choices_with_warning`: trial with no listener choice
  → row absent from output, `logging.WARNING` emitted with count
- `test_join_tables_multi_listener_produces_multiple_rows`: one trial, two listeners
  → two rows with same utterance, different listener_id and selected_image_id
- `test_filter_images_top_n`: 5 images across games; n=3 → only games where all
  images are in the top-3 most-frequent survive
- `test_filter_images_threshold`: game where one image appears in only 1 game; threshold=2
  → that game dropped
- `test_filter_images_raises_on_empty_result`: n so high no games survive → `ValueError`

**test_pipeline.py**

- `test_build_trial_batch_shapes`: 50 trials, 12 options, D=16 → `utterance_emb` is
  (50,16), `option_embs` is (50,12,16), all index arrays are length-50
- `test_build_trial_batch_raises_on_wrong_option_count`: option_set of length 11 →
  `ValueError` mentioning "12"
- `test_build_trial_batch_raises_on_missing_image_embedding`: image_id not in
  `image_embeddings` → `KeyError` with that id in the message
- `test_build_trial_batch_raises_on_missing_utterance_embedding`: utterance not in
  `utterance_embeddings` → `KeyError` with the utterance text in the message
- `test_build_trial_batch_ids_are_contiguous`: game_ids ∈ [0, n_games),
  listener_ids ∈ [0, n_listeners), image_ids ∈ [0, n_images)

**test_cache.py**

- `test_cache_round_trip`: `put` three embeddings, `get` them → exact numpy equality
- `test_cache_partial_miss`: cache has keys {a, b}; `get`([a, b, c]) → found={a,b},
  missing=[c]
- `test_cache_persists_to_disk`: `put` embeddings, construct new `EmbeddingCache` from
  same path → same values returned
- `test_embeddings_for_ids_calls_encoder_only_for_missing`: ids=[a,b,c], cache has a;
  encoder mock called with exactly [b,c], not [a]
- `test_embeddings_for_ids_puts_new_entries_in_cache`: after call, cache.get([b,c])
  returns both without missing entries

**test_listener_model.py**

- `test_listener_model_traces_without_error`: call `listener_model` with a minimal batch
  (N=10, D=8, 2 listeners, 12 options) → no exception raised
- `test_compute_success_probs_shape`: call with N=20 batch → returns array of shape (20,)
- `test_compute_success_probs_values_in_range`: all values in (0, 1)
- `test_compute_success_probs_high_for_clear_target`: construct batch where utterance
  embedding is identical to target image embedding, cos ≈ 0.0 to all distractors,
  β=5.0 → s_t > 0.9 for all trials
- `test_fit_listener_raises_on_nan_loss`: construct degenerate model (patch `run_svi` to
  inject NaN at step 101) → `RuntimeError` with "NaN loss" in message

**test_convention_model.py**

- `test_compute_sigma2_t_at_success`: s_t=1.0, sigma_min=0.1, sigma_max=2.0 →
  result ≈ 0.1 (within 1e-5)
- `test_compute_sigma2_t_at_failure`: s_t=0.0 (clipped to eps=0.01), sigma_min=0.1,
  sigma_max=2.0 → result ≈ 0.1 + 1.9*(1/0.01 - 1) = 188.2
- `test_compute_sigma2_t_monotone_decreasing_in_s`: s_t values [0.1, 0.5, 0.9] →
  sigma2_t values are strictly decreasing
- `test_compute_r1_average_shape`: batch with n_images=3, D=8 → returns (3, 8)
- `test_compute_r1_average_uses_only_rep1`: batch where rep_num=1 utterances are all
  zeros, rep_num>1 are all ones → r1_average ≈ zeros
- `test_compute_r1_average_raises_if_image_missing_from_rep1`: image with no rep_num==1
  trials → `ValueError` naming that image
- `test_speaker_convention_model_traces`: call with minimal batch (N=20, D=8,
  2 games, 2 images) → no exception
- `test_listener_convention_model_traces`: same setup → no exception
- `test_fit_convention_output_shapes`: n_games=3, n_images=2, D=8 → `speaker_m_loc`
  shape is (3, 2, 8), `listener_m_loc` shape is (3, 2, 8)

**test_svi.py**

- `test_run_svi_loss_decreases`: Gaussian mean estimation model (2 params, 20 obs) →
  `losses[-1] < losses[0]`
- `test_run_svi_returns_correct_losses_length`: n_steps=100 → `len(losses) == 100`
- `test_run_svi_nan_raises_after_step_100`: patch model to return NaN loss at step 150 →
  `RuntimeError` with "NaN loss at step 150"
- `test_run_svi_nan_before_step_100_does_not_raise`: NaN at step 50 → no exception
  (grace period for early instability)

**test_predictions.py**

- `test_step_sizes_over_reps_columns`: returns DataFrame with exactly
  {game_id, image_id, rep_num, step_size}
- `test_step_sizes_over_reps_decreasing_on_tight_emission`: construct `ConventionFit`
  with small sigma_min (tight emission), batch with s_t=0.9 and 5 reps →
  mean step_size in reps 4–5 < mean step_size in reps 1–2
- `test_step_sizes_over_reps_omits_pairs_with_fewer_than_2_reps`: (game,image) with
  only 1 rep → not present in output
- `test_semantic_displacement_columns`: returns DataFrame with exactly
  {game_id, image_id, rep_num, displacement, prev_success}
- `test_semantic_displacement_direction`: construct batch where failed trials are
  followed by large jumps, successes by small jumps →
  `displacement[prev_success==False].mean() > displacement[prev_success==True].mean()`

---

### Integration tests (`@pytest.mark.slow`)

**INT-1: Listener β recovery** (`tests/integration/test_int_listener.py`)

Setup:
- 2 listeners with true β = [2.0, 5.0]
- 200 trials, 12 images, D=16
- Target embedding: cos ≈ 0.9 to target image embedding, cos ≈ 0.1–0.3 to distractors
  (randomly generated, fixed seed)
- Listener choices sampled from softmax(β · cos_sims) under true β values
- 5000 SVI steps

Assertions:
1. β order preserved: fitted $\hat\beta_0 < \hat\beta_1$ (posterior means)
2. Each $|\hat\beta_\ell - \beta_\ell^\text{true}| < 1.5$

**INT-2: Step sizes decrease** (`tests/integration/test_int_convention.py`)

Setup:
- 6 games × 3 images × 6 reps = 108 trials (before multi-listener expansion), D=8
- True $m_{g,i}$ = random unit vectors, one per (game, image), fixed seed
- Utterance embeddings: $z(u_t) \sim \mathcal{N}(m_{g,i}, 0.01 \cdot I)$, then L2-normalize
- $s_t = 0.9$ for all trials (fixed, high success)
- $\mu_i$ init from `compute_r1_average`
- 8000 SVI steps (speaker model only for this test)

Assertions:
1. `step_sizes_over_reps(role="speaker")` computed for all (game, image) pairs
2. For ≥ 80% of pairs: `mean(step_size[rep_num >= 5]) < mean(step_size[rep_num <= 2])`

---

## Implementation Order

BDD dual-loop TDD. Outer loops correspond to integration tests; inner cycles are unit
red-green-refactor. Never write production code without a failing test first.

**Outer loop 1: target INT-1 (listener β recovery)**

1. Write INT-1 stub — confirm it fails with ImportError or NameError
2. `TrialBatch` dataclass + `build_trial_batch` shape/error checks
   → `test_build_trial_batch_shapes`, `test_build_trial_batch_raises_*`
3. `join_tables` (message concat, multi-listener, drop missing)
   → all `test_join_tables_*`
4. `filter_images` top_n mode + empty result guard
   → `test_filter_images_top_n`, `test_filter_images_raises_*`
5. `EmbeddingCache` round-trip + partial miss + persistence
   → `test_cache_*`
6. `embeddings_for_ids` (calls encoder only for missing, updates cache)
   → `test_embeddings_for_ids_*`
7. `literal_listener_model` + `literal_listener_guide` traces without error
   → `test_listener_model_traces_without_error`
8. `compute_success_probs` shape, range, direction
   → `test_compute_success_probs_*`
9. `run_svi` loss trajectory, NaN detection, length
   → all `test_run_svi_*`
10. `fit_listener` wires model + guide + run_svi → `ListenerFit`
    → `test_fit_listener_raises_on_nan_loss`
11. **Green INT-1** — wire full listener pipeline; β recovery passes

**Outer loop 2: target INT-2 (step sizes decrease)**

12. Write INT-2 stub — confirm it fails
13. `compute_sigma2_t` formula and edge cases (`conventional_speaker.py`)
    → all `test_compute_sigma2_t_*`
14. `compute_r1_average` shape, values, error on missing image (`conventional_speaker.py`)
    → all `test_compute_r1_average_*`
15. `speaker_convention_model` (`conventional_speaker.py`) + `listener_convention_model` (`conventional_listener.py`) trace
    → `test_*_convention_model_traces`
16. `fit_convention` wires both models; output shape check (`conventional_speaker.py`)
    → `test_fit_convention_output_shapes`
17. `sequential_kalman_means` (no separate test — tested through step_sizes)
18. `step_sizes_over_reps` columns, decreasing, omit-single-rep
    → all `test_step_sizes_over_reps_*`
19. **Green INT-2** — wire full convention pipeline; step sizes decrease passes

**Cleanup and remaining analysis**

20. `semantic_displacement_after_error` columns and direction
    → all `test_semantic_displacement_*`
21. `filter_images` threshold mode
    → `test_filter_images_threshold`
22. Full suite refactor — no test should break; no new behavior without a test
