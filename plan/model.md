# Unified Model: Continuous CHAI for Iterated Reference Games

This document specifies the computational model as a unified mathematical and conceptual
description. It synthesizes the CHAI framework (Hawkins et al. 2021) adapted to continuous
semantics, and the embedding-based Bayesian learning approach developed in planning
discussions. Components are separated into the **core model** (the minimal structure needed
to capture within-game convention dynamics) and **additional modules** (extensions that
address further phenomena or add modeling power).

**Key design goal:** The phenomena of interest — step-size decrease over rounds, direction
change after communicative errors, within-game convergence, between-game divergence —
must *emerge* from the Bayesian structure rather than being imposed by hand (e.g., via
scheduled learning rates or explicit error-triggered exploration rules).

---

## Notation

| Symbol | Meaning |
|---|---|
| g | game index |
| i | image / referent index |
| u | utterance (free text) |
| c | listener's chosen image |
| O | option set (always 12 images in RefBank) |
| ℓ | listener identifier |
| t | trial index within a game (rep_num for a specific image) |
| z(u) ∈ R^D | CLIP text embedding of u, L2-normalized |
| x(i) ∈ R^D | CLIP image embedding of image i, L2-normalized |
| cos(a,b) | cosine similarity |

---

## Core Model

### 1. Semantic Representations

Both images and utterances are embedded in a shared semantic space using CLIP:

- **Image embedding**: x(i) = CLIP_image(i) ∈ R^D, L2-normalized
- **Utterance embedding**: z(u) = CLIP_text(u) ∈ R^D, L2-normalized

Using the same encoder for both modalities means that cosine similarity between z(u) and
x(i) is a direct, alignment-free measure of how well the utterance describes the image.
This handles the full range of utterance lengths in RefBank (single words through
multi-sentence descriptions) without special-casing.

*Relationship to CHAI:* CHAI uses a binary utterance–object meaning matrix L_φ(u, o).
Here, cos(z(u), x(i)) is the continuous analogue of L_φ(u, o): graded rather than Boolean,
but serving the same role of measuring the degree to which utterance u applies to image i.
The φ parameter (agent-specific lexicon) becomes the latent convention point m_{g,i}
described in Section 3 below.

---

### 2. Listener Model (L_0)

The **literal listener** selects an image from the option set based on the semantic
similarity of the utterance to each image, tempered by a per-listener sharpness parameter:

$$
L_0(c \mid u, O, \ell) = \text{softmax}_{j \in O}\big(\beta_\ell \cdot \cos(z(u), x(j))\big)
$$

- **β_ℓ** is the inverse temperature for listener ℓ. High β means a sharp decision that
  concentrates probability on the most similar image; low β means a flat, noisy distribution.
- β_ℓ is given a hierarchical prior:
  $$\beta_\ell \sim \text{LogNormal}(\mu_\beta, \sigma_\beta)$$
  This partial-pools calibration estimates across listeners while allowing individual variation.

The listener's **chosen image c_t** (not just success/failure) provides a 12-way
multinomial likelihood — a sharp, unambiguous signal for fitting.

*Relationship to CHAI:* This is the continuous analogue of CHAI's L_0(o | u) ∝ L_φ(u, o).
The softmax replaces the Boolean lookup, β_ℓ replaces the single precision parameter α_L,
and the partial-pooled prior on β_ℓ corresponds to CHAI's community-level hyperprior Θ
applied to listener sharpness.

> **Discussion point:** Whether L_0 should condition on the convention m_{g,i} (as in the
> CHAI listener-with-lexical-uncertainty version) or remain static (conditioning only on
> CLIP similarities) is a modeling choice. Using a static L_0 is simpler and allows the
> listener model to be fitted independently in Stage 1. A convention-conditioned L_0 would
> require joint fitting with the convention model but more faithfully captures how listeners
> update their interpretation based on in-game experience.

---

### 3. Convention Structure (C1 and C3 analogue)

The core representational commitment: each game g and image i has a **latent convention
point** m_{g,i} ∈ R^D that summarizes where in semantic space the pair's descriptions of
image i have converged.

This is given a two-level hierarchical prior:

$$
\mu_i \sim \mathcal{N}(0, I) \quad \text{[global prototype for image } i\text{, fitted across all games]}
$$
$$
\delta_{g,i} \sim \mathcal{N}(0, \sigma^2_\text{game} \cdot I)
$$
$$
m_{g,i} = \mu_i + \delta_{g,i}
$$

- **μ_i** is the community-level prototype for image i: the typical region of semantic
  space where descriptions of that image tend to land across all games. This corresponds to
  CHAI's Θ (community-level over-hypothesis).
- **δ_{g,i}** is the game-specific deviation: how far this particular game's convention
  has drifted from the community prototype. δ_{g,i} = 0 means the game uses the population
  consensus; larger δ_{g,i} means an idiosyncratic convention. This corresponds to CHAI's
  φ_k (partner-specific lexicon).
- **σ_game** controls how much games typically diverge from the prototype.

With the 12 focal images reused across many games, μ_i is well-identified from data.
The between-game divergence — a key phenomenon of interest — is directly quantifiable
as the empirical variance of δ_{g,i} across games for each image.

> **Discussion point:** How to initialize μ_i is not settled. Candidates:
> (a) fit μ_i freely from utterance data, treating it as a latent to be estimated;
> (b) initialize μ_i at the CLIP image embedding x(i) projected to text space, then
> allow it to drift. Option (a) requires no cross-modal assumptions; option (b) anchors
> the prior in perceptual content.

---

### 4. Observation Model

On each trial t, two observed signals update the posterior over m_{g,i}:

**Signal 1 — Listener choice c_t** (primary, no approximation needed):

$$
P(c_t \mid u_t, O_t, \ell_t) = L_0(c_t \mid u_t, O_t, \ell_t)
$$

This is the multinomial log-likelihood from the fitted listener model. It provides direct
evidence about whether the utterance was communicatively effective given the current option
geometry.

**Signal 2 — Utterance embedding z(u_t)** (approximate; replaces the full S_1 speaker
likelihood):

Rather than computing P(u_t | m_{g,i}) via a full RSA speaker distribution over all
possible strings (intractable for free text), the observed utterance embedding is treated
as a noisy sample from the latent convention:

$$
z(u_t) \sim \mathcal{N}(m_{g,i_t},\ \sigma^2_t \cdot I)
$$

The **accuracy-dependent variance** σ²_t is the mechanism by which communicative success
modulates the update:

$$
\sigma^2_t = \sigma^2_\text{min} + \frac{\sigma^2_\text{max} - \sigma^2_\text{min}}{\text{clip}(s_t, \varepsilon, 1)}
$$

where s_t = L_0(i_t | u_t, O_t, ℓ_t) is the model-implied probability that the listener
correctly identifies the target (computed from the fitted L_0).

- When s_t ≈ 1 (high success): σ²_t ≈ σ²_min — the emission is tight, and the observed
  utterance strongly informs the convention. The posterior over m_{g,i} tightens.
- When s_t ≈ 0 (failure): σ²_t → σ²_max — the emission is wide, meaning the failed
  utterance is treated as unreliable evidence. The posterior barely moves toward the
  failed embedding, and combined with the listener-choice signal that penalizes the current
  utterance direction, future utterances will tend to move elsewhere.

*Relationship to CHAI:* In CHAI, each observation tuple (o*, u', o') contributes
asymmetrically to speaker and listener beliefs (speaker updates on L_0(o' | u'); listener
updates on S_1(u | o*)). Here we simplify to a shared convention update. This asymmetry
is noted as a discussion point below.

---

### 5. Posterior Update (C2 analogue)

The full posterior over the convention for image i in game g is:

$$
p(m_{g,i} \mid D_{g,i}) \;\propto\; p(m_{g,i}) \cdot \prod_t p(c_t \mid u_t, O_t, \ell_t) \cdot p(z(u_t) \mid m_{g,i}, \sigma^2_t)
$$

where D_{g,i} = {(u_t, c_t, O_t, ℓ_t)} is all observations for image i in game g.

This is not conjugate in closed form (because L_0 is a softmax that depends on m_{g,i}
only through the accuracy-weighting of σ²_t). Inference options:

- **Variational (SVI)**: fit q(m_{g,i}) = N(m̂, Σ̂) with an ELBO objective.
  Posterior covariance Σ̂ shrinks naturally as reliable evidence accumulates.
- **Laplace/MAP**: optimize m̂ = argmax log p(m_{g,i} | D_{g,i}),
  then approximate Σ̂ by the inverse Hessian at m̂.
- **Particle filter**: maintain a set of weighted samples per (game, image),
  reweighted by the joint likelihood at each round. Supports online updating.

> **Discussion point:** SVI (via NumPyro) fits naturally into the existing plan.md
> infrastructure and scales to many (game, image) pairs. Laplace is cheaper per game
> but requires second-order computation. Particle filter is most faithful to online
> Bayesian inference but adds complexity. The minimal implementation uses SVI.

---

### 6. Emergent Predictions

The following phenomena are *not* imposed — they should emerge from the model structure
and be checked post-fitting on held-out games:

| Phenomenon | Mechanism |
|---|---|
| **Step-size decrease** | Posterior covariance shrinks with evidence → E[m_{g,i}] moves less per round |
| **Direction change after error** | Low s_t → wide σ²_t → failed embedding weakly informs posterior; listener-choice signal pushes away from failed direction |
| **Within-game convergence** | Utterances cluster around inferred m_{g,i} as posterior concentrates |
| **Between-game divergence** | δ_{g,i} differs across games due to stochastic initialization and different interaction trajectories |
| **Within-game differentiation** | Discriminability pressure from L_0 (12-way softmax) pushes m_{g,i} for different images within a game apart |

---

### 7. Core Free Parameters

| Parameter | Role | How estimated |
|---|---|---|
| μ_β, σ_β | Hyperprior on listener sharpness | Fitted in Stage 1 (listener model) |
| β_ℓ | Per-listener inverse temperature | Fitted in Stage 1, hierarchical |
| σ_game | Spread of game conventions around μ_i | Fitted in Stage 2 |
| σ_min | Emission variance for highly successful trials | Fitted in Stage 2 |
| σ_max | Emission variance for failed trials | Fitted in Stage 2 |
| ε | Clipping floor on s_t (numerical stability) | Fixed small constant (e.g. 0.01) |

---

## Additional Modules

### M1. Low-Rank Projection

A learnable linear projection P ∈ R^{d×D} (d ≪ D) maps CLIP embeddings to a
task-relevant subspace:

$$
L_0(c \mid u, O, \ell) = \text{softmax}_{j \in O}\big(\beta_\ell \cdot \cos(Pz(u), Px(j))\big)
$$

The motivation is that the full CLIP space (D ≈ 512 or 768) contains many dimensions
irrelevant to the visual reference game task. Projecting to d ≈ 10–50 dimensions isolates
discriminative directions and may improve listener model fit.

Fits jointly with β_ℓ in Stage 1. Choose d by held-out predictive likelihood.

**When to add:** Start without (d = None, use CLIP space directly). Add only if the raw
CLIP listener model has poor held-out accuracy.

---

### M2. Speaker Model (S_1)

The RSA pragmatic speaker selects utterances to maximize informativity minus cost:

$$
U(u; i, O, m_{g,i}) = \alpha \log L_0(i \mid u, O) - \lambda \cdot C(u) - \frac{1}{2\sigma^2}\|z(u) - m_{g,i}\|^2
$$
$$
S_1(u \mid i, O, m_{g,i}) \propto \exp\big(\alpha_S \cdot U(u; i, O, m_{g,i})\big)
$$

The three terms are:
- **Informativity**: log probability that the listener identifies target i (RSA term)
- **Cost**: C(u) = word count, character count, or similar surface measure
- **Convention pull**: Euclidean distance from the current convention point m_{g,i};
  this creates a drive to reuse established semantic regions (coordination pressure)

Because the utterance space is open (free text), S_1 cannot be used directly as a proper
generative distribution — the normalizing constant sums over all possible strings. Two uses:

1. **As a scoring function** (tractable): given observed u_t, evaluate U(u_t) without
   normalizing. Useful for quantifying how speaker-optimal observed utterances are.
2. **As a generative model via proposal + rerank** (deferred):
   - Propose K candidates from an LLM conditioned on the target image
   - Score each with U(u_k)
   - Select via softmax over scores
   This allows the model to predict/generate utterances but requires an LLM call per trial.

**When to add:** Phase 1 defers the speaker model. Add scoring (use 1) once the convention
model is fitted, to check whether observed utterances are consistent with RSA optimality.
Add proposal + rerank (use 2) only if utterance generation is a target.

Additional parameters: α_S (speaker rationality), λ (cost weight), σ (convention pull
strength). These can also be estimated from data but are not needed for listener-choice
fitting.

---

### M3. Memory / Recency Discounting

Older observations can be down-weighted in the posterior, on the grounds that earlier
conventions may be less predictive of current behavior:

$$
p(m_{g,i} \mid D_{g,i}) \propto p(m_{g,i}) \cdot \prod_{\tau=0}^{T} \gamma^\tau \cdot p(\text{obs}_{T-\tau} \mid m_{g,i})
$$

where τ = 0 is the most recent trial and γ ∈ [0, 1] is the decay parameter. In CHAI,
γ = 0.8.

**When to add:** Start with γ = 1 (no decay, all observations equally weighted). This is
appropriate when the convention is assumed to be stationary within a game. Add decay
only if evidence suggests that earlier rounds are less predictive, or if modeling games
with partner-switching where the convention resets.

---

### M4. Player-Specific Effects

In games with rotating speakers/listeners, player identity may introduce systematic
variation beyond the game-level convention. A player-specific deviation can be added:

$$
\delta_{p,i} \sim \mathcal{N}(0, \sigma^2_\text{player} \cdot I)
$$
$$
m_{g,p,i} = \mu_i + \delta_{g,i} + \delta_{p,i}
$$

This separates:
- δ_{g,i}: what this game's pair has converged on for image i
- δ_{p,i}: what this player brings regardless of partner (idiolect / personal semantic
  tendencies)

**When to add:** Only if there is enough data per player across multiple games to estimate
δ_{p,i} reliably. With the 12 focal images, this requires players who appear in many games.
Start without; add if player identity is found to predict residual variance in Stage 2.

Similarly, a per-speaker cost weight λ_s can be estimated hierarchically if cost
sensitivity varies across speakers.

---

### M5. Repulsive Failure Likelihood

The core model treats failed utterances as *weak* evidence (wide σ²_t), but not as
*anti-evidence*. A stronger version actively pushes the posterior away from the failed
utterance embedding:

$$
p(z(u_t) \mid m, y=0) \propto 1 - \exp\!\left(-\frac{\|z(u_t) - m\|^2}{2\rho^2}\right)
$$

implemented stably as a contrastive term in the log-likelihood. This makes "change
direction after error" more forceful: the posterior is explicitly pushed away from the
failed embedding, rather than just not being pulled toward it.

**When to add:** Check first whether the accuracy-weighted emission model (core) already
produces the predicted direction-change effect on held-out data. Add the repulsive term
only if the soft version is insufficient.

---

### M6. Asymmetric Speaker / Listener Inference

In CHAI, speaker and listener maintain *separate* beliefs about the partner's lexicon:

- **Speaker infers** from the listener's choice c_t: P_S(φ | data) updated via L_0(c_t | u_t)
- **Listener infers** from the speaker's utterance u_t: P_L(φ | data) updated via S_1(u_t | o*_t, φ)

In the core model we simplify by maintaining a single shared convention m_{g,i} updated
by both signals. This is appropriate when RefBank games have a fixed speaker role (one
person describes, others choose), because the update is primarily from the speaker's
perspective.

**When to add:** Implement separate speaker/listener beliefs if the data contain many
role-switching games, or if the symmetric version shows systematic residuals for the
two roles.

---

## Model Variants for Comparison

Three pooling structures correspond to different assumptions about cross-game sharing:

| Variant | Structure | What it captures |
|---|---|---|
| **Complete-pooling** | No δ_{g,i}; all games share μ_i | No game-specificity; convention is universal |
| **No-pooling** | Independent m_{g,i} per game with uninformative prior; no shared μ_i | Games develop independently; no generalization |
| **Partial-pooling (core model)** | Hierarchical m_{g,i} = μ_i + δ_{g,i} | Game-specificity + community structure |

These can be compared by held-out listener choice log-likelihood. Complete-pooling predicts
no between-game divergence; no-pooling predicts no community-level prototype; only
partial-pooling predicts both (as in CHAI).

---

## Open Questions

1. **Symmetric vs. asymmetric inference.** Should speaker and listener maintain separate
   beliefs (as in CHAI), or update a shared game-level convention? The symmetric version
   is simpler and may be sufficient for games with a fixed speaker role; the asymmetric
   version is more faithful to CHAI but requires the full S_1 speaker model to compute
   listener inference, which is expensive.

2. **Initializing μ_i.** Should the global prototype for image i be initialized at the
   CLIP image embedding (perceptually anchored) or estimated freely from utterance data?
   The CLIP initialization provides a principled starting point but introduces cross-modal
   assumptions; free estimation is more flexible but requires more data.

3. **σ²_t as a function of s_t.** The choice of functional form (the current formula
   with clip(s_t, ε, 1) in the denominator) is one of several possibilities. Alternative:
   a linear interpolation σ²_t = (1 - s_t) · σ²_max + s_t · σ²_min. The functional form
   affects how sharply the model distinguishes near-misses from clear successes.

4. **Inference method.** SVI (variational Gaussians) is the current plan. A particle
   filter is more faithful to online per-round Bayesian updating but more complex. Laplace
   is lighter but loses uncertainty in posterior covariance. The choice matters for how
   cleanly "step size decrease" emerges.

5. **Role of the listener choice in convention updating.** The core model has two signals:
   the listener choice c_t (via L_0 likelihood) and the utterance embedding z(u_t) (via
   emission model). It's not obvious how to weight these. The listener choice is clean and
   direct but does not directly constrain m_{g,i} (only indirectly through s_t). Does the
   listener-choice term need to enter the m_{g,i} posterior directly, or is accuracy-weighting
   of the emission sufficient?

6. **Whether failures are weak evidence or anti-evidence.** The choice between M5 (repulsive
   likelihood) and the core accuracy-weighted emission determines whether errors produce
   gentle drift or active repulsion. This is an empirical question: check whether the
   direction-change effect emerges clearly from the core model before adding M5.

7. **Multi-listener trials.** In trials with multiple simultaneous listeners, each listener
   provides an independent choice c_t^ℓ. Currently treated as independent data points.
   A jointly modeled version would acknowledge that listener choices on the same trial are
   not i.i.d. (they receive the same utterance). This is likely a minor correction.

8. **Low-rank projection and dimensionality.** Whether the raw CLIP space is adequate or
   a learned projection P is necessary is an empirical question to be answered in Stage 1.
   The optimal d (if projection is added) should be chosen by cross-validated predictive
   likelihood, not by prior assumption.
