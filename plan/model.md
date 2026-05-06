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

> **Note:** The current model uses a single σ_game shared across all images. A natural
> extension is a per-image σ_{game,i}, which would allow the model to represent differences
> in *nameability*: highly nameable images (with a dominant conventional label) would have
> small σ_{game,i} because games converge to similar conventions, while ambiguous or
> hard-to-name images would have large σ_{game,i}. This is deferred — fit the scalar
> version first and check whether residual between-game variance is image-specific.

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
\sigma^2_t = \sigma^2_\text{min} + (\sigma^2_\text{max} - \sigma^2_\text{min}) \cdot \frac{1 - \max(\varepsilon,\, \min(s_t, 1))}{\max(\varepsilon,\, \min(s_t, 1))}
$$

where s_t = L_0(i_t | u_t, O_t, ℓ_t) is the model-implied probability that the listener
correctly identifies the target (computed from the fitted L_0).

- When s_t = 1 (perfect success): σ²_t = σ²_min — the emission is tight, and the observed
  utterance strongly informs the convention. The posterior over m_{g,i} tightens.
- When s_t ≈ 0 (failure): σ²_t grows large — the emission is wide, meaning the failed
  utterance is treated as unreliable evidence. The posterior barely moves toward the
  failed embedding, and combined with the listener-choice signal that penalizes the current
  utterance direction, future utterances will tend to move elsewhere.

*Relationship to CHAI:* In CHAI, each observation tuple (o*, u', o') contributes
asymmetrically to speaker and listener beliefs (speaker updates on L_0(o' | u'); listener
updates on S_1(u | o*)). The model below maintains this asymmetry with separate
speaker-side and listener-side posteriors (Section 5).

---

### 5. Posterior Updates (C2 analogue)

Speaker and listener maintain separate beliefs about the convention, because they have
access to different information on each trial.

#### 5a. Speaker-side update

The speaker produced u_t knowing the target i_t, then observed the listener's choice c_t.
Their posterior over m_{g,i} is updated by the utterance embedding, with reliability
modulated by the listener's success:

$$
p_S(m_{g,i} \mid D^S_{g,i}) \;\propto\; p(m_{g,i}) \cdot \prod_t p(z(u_t) \mid m_{g,i},\, \sigma^2_t)
$$

where $D^S_{g,i} = \{(u_t, c_t, O_t, \ell_t)\}$ and $\sigma^2_t$ is the accuracy-dependent
variance from Section 4, computed from $s_t = L_0(i_t \mid u_t, O_t, \ell_t)$ — which the
speaker can evaluate because they know the target $i_t$.

#### 5b. Listener-side update

The listener heard u_t, chose c_t, and received correctness feedback. Their posterior over
m_{g,i} is updated by the utterance they heard, with reliability modulated by whether
their choice was correct:

$$
p_L(m_{g,i} \mid D^L_{g,i}) \;\propto\; p(m_{g,i}) \cdot \prod_t p(z(u_t) \mid m_{g,i},\, \sigma^2_t)
$$

where $D^L_{g,i} = \{(u_t, c_t, O_t)\}$ and $\sigma^2_t$ is computed from $s_t^L =
\mathbf{1}[c_t = i_t]$ (or a soft version) — the listener's own success signal, available
after feedback.

The two posteriors have the same emission structure but differ in how $s_t$ is computed:
the speaker uses the model-implied $L_0$ score (they know the target), the listener uses
their observed outcome (they know their choice was right or wrong).

This is not conjugate in closed form. Inference options:

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

### 6. Speaker Model (S_1)

The RSA pragmatic speaker selects utterances to maximize informativity minus cost:

$$
U(u; i, O) = \alpha_S \log L_0(i \mid u, O) - \lambda \cdot C(u)
$$
$$
S_1(u \mid i, O) \propto \exp\big(\alpha_S \cdot U(u; i, O)\big)
$$

The two terms are:
- **Informativity**: log probability that the listener identifies target i (RSA term)
- **Cost**: C(u) = word count, character count, or similar surface measure

Convention does not enter as an explicit pull term here; how game-specific conventions
reshape utterance behavior is an open question (see Open Question 12).

Because the utterance space is open (free text), S_1 cannot be used as a proper generative
distribution — the normalizing constant sums over all possible strings. Two uses:

1. **As a scoring function** (tractable): given observed u_t, evaluate U(u_t) without
   normalizing. Useful for quantifying how speaker-optimal observed utterances are.
2. **As a generative model via proposal + rerank**:
   - Propose K candidates from an LLM conditioned on the target image
   - Score each with U(u_k)
   - Select via softmax over scores

---

### 7. Emergent Predictions

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

### 8. Core Free Parameters

| Parameter | Role | How estimated |
|---|---|---|
| μ_β, σ_β | Hyperprior on listener sharpness | Fitted in Stage 1 (listener model) |
| β_ℓ | Per-listener inverse temperature | Fitted in Stage 1, hierarchical |
| σ_game | Spread of game conventions around μ_i | Fitted in Stage 2 |
| σ_min | Emission variance for highly successful trials | Fitted in Stage 2 |
| σ_max | Emission variance for failed trials | Fitted in Stage 2 |
| ε | Clipping floor on s_t (numerical stability) | Fixed small constant (e.g. 0.01) |
| α_S | Speaker rationality | Fitted in Stage 2 |
| λ | Cost weight | Fitted in Stage 2 |

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

### M2. Memory / Recency Discounting

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

*This module has been promoted to the core model (Section 5).* Separate speaker-side
and listener-side posteriors are now part of the base specification rather than an optional
extension, because the two roles have structurally different information access and the
asymmetry is necessary to correctly model convention updating.

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

## Evaluation

Games are split 80/20 into train and test sets. All model fitting (convention posteriors,
listener sharpness parameters, speaker parameters) is done on training games only. Evaluation
is on held-out test games, which the model has never seen.

The primary evaluation metric is **held-out listener choice log-likelihood**: for each
trial, $\log p(c_t \mid u_t, O_t, \ell_t)$ — the probability the model assigns to the
image the listener actually chose, given the utterance, option set, and listener identity.
This is computable for any game without access to ground-truth convention labels.

Model variants (complete-pooling, no-pooling, partial-pooling) are compared on this metric.

The **speaker model** is evaluated by ranking: because $S_1$'s normalizing constant is
intractable over free text, we evaluate the utility $U(u; i, O)$ of the speaker's actual
utterance relative to a set of sampled alternative utterances for the same target. A
well-fitted speaker model should assign higher utility to the observed utterance than to
alternatives — this can be summarized as rank or top-$k$ accuracy across trials.

> **TODO:** Work out how to assess whether the model captures the phenomena of interest
> (step-size decrease, direction change after error, within-game convergence, between-game
> divergence, within-game differentiation). These are listed as emergent predictions in
> Section 7 but the evaluation procedure for checking them against data is not yet specified.

---

## Open Questions

1. **How to combine speaker and listener posteriors.** The model now maintains separate
   $p_S(m_{g,i})$ and $p_L(m_{g,i})$ (Section 5). It is not yet specified how these
   are combined — whether predictions at round $t$ use one or the other (depending on
   role), or a mixture, or whether they are kept fully separate throughout. This requires
   a decision before implementation.

   > **Answer:** The two posteriors are kept fully separate and parallel throughout — no
   > combination. At prediction time, the speaker role uses $p_S$ and the listener role
   > uses $p_L$. This separation is also analytically useful: tracking the distance between
   > $p_S(m_{g,i})$ and $p_L(m_{g,i})$ over rounds lets us measure whether the pair's
   > meaning representations converge during a game.

2. **Initializing μ_i.** Should the global prototype for image i be initialized at the
   CLIP image embedding (perceptually anchored) or estimated freely from utterance data?
   The CLIP initialization provides a principled starting point but introduces cross-modal
   assumptions; free estimation is more flexible but requires more data.

   > **Answer:** Three options, each modularized so they can be swapped:
   > 1. **CLIP image embedding** — x(i) projected to text space. May require a learned
   >    linear projector to bridge the modality gap.
   > 2. **R1 average** — mean of utterance embeddings z(u) from round 1 across all games
   >    for image i. Purely data-driven; no cross-modal assumption.
   > 3. **Kilogram name embeddings** — text embeddings of the canonical names for each
   >    image from the Kilogram dataset. Provides a linguistically grounded prior.

3. **σ²_t as a function of s_t.** The choice of functional form (the current formula
   with max(ε, min(s_t, 1)) in the denominator) is one of several possibilities. Alternative:
   a linear interpolation σ²_t = (1 - s_t) · σ²_max + s_t · σ²_min. The functional form
   affects how sharply the model distinguishes near-misses from clear successes.

   > **Answer:** Current formulation is fine.

4. **Inference method.** SVI (variational Gaussians) is the current plan. A particle
   filter is more faithful to online per-round Bayesian updating but more complex. Laplace
   is lighter but loses uncertainty in posterior covariance. The choice matters for how
   cleanly "step size decrease" emerges.

   > **Answer:** SVI is okay.

5. **Role of the listener choice in the speaker-side posterior update.** With asymmetric
   updating, the listener's choice c_t enters the speaker-side update (5a) only through
   σ²_t (via s_t = L_0(i_t | u_t, O_t, ℓ_t)). The question is whether this
   accuracy-weighting of the emission is sufficient to constrain $p_S(m_{g,i})$, or
   whether the listener's multinomial choice also needs to enter the speaker posterior
   directly as a likelihood term.

   > **Answer:** Use binary correct/incorrect ($s_t = \mathbf{1}[c_t = i_t]$) for now;
   > adding the direct multinomial likelihood term is deferred. Note: the right treatment
   > here also depends on the feedback structure (what correctness information is actually
   > given to players) and on polyadic games (multiple simultaneous listeners), so revisit
   > alongside those features.

6. **Whether failures are weak evidence or anti-evidence.** The choice between M2 (repulsive
   likelihood) and the core accuracy-weighted emission determines whether errors produce
   gentle drift or active repulsion. This is an empirical question: check whether the
   direction-change effect emerges clearly from the core model before adding M2.

   > **Answer:** Keep M2 as optional for now; try it once the core model is fitted.

7. **Multi-listener trials.** In trials with multiple simultaneous listeners, each listener
   provides an independent choice c_t^ℓ. Currently treated as independent data points.
   A jointly modeled version would acknowledge that listener choices on the same trial are
   not i.i.d. (they receive the same utterance). This is likely a minor correction.

   > **Answer:** Treat as i.i.d. for now.

8. **How to bridge the CLIP modality gap for the listener model.** Empirically (run
   24087645), raw CLIP text→image cosine similarity is a weak signal for abstract tangrams:
   target image ranks 5.3/12 on average, L_0 argmax accuracy 12% vs 90% empirical. The
   text embeddings carry the convention signal and should remain fixed; the image embeddings
   need to be shifted into alignment. Three options:

   - **(a) Full D×D map.** Learn $P \in \mathbb{R}^{D \times D}$ applied only to image
     embeddings: $\text{sim}(u, i) = z_\text{text} \cdot (P\, z_\text{image})$. Initialize
     $P = I$, penalize $\|P - I\|_F$. Conceptually clean but heavily overparameterized:
     only 12 image vectors constrain 512² = 262k parameters; the matrix structure mainly
     serves as a regularizer.

   - **(b) Low-rank residual.** $P = I + AB$ where $A, B \in \mathbb{R}^{D \times r}$,
     $r \ll D$ (e.g. $r = 16$). Explicit low-rank constraint; 2Dr ≈ 16k parameters. More
     principled than (a) given only 12 images; equivalent in practice since the image
     embedding matrix has rank ≤ 12.

   - **(c) Free image embeddings.** Freely learn 12 new $D$-dimensional image embeddings,
     initialized from CLIP. Equivalent to (a)/(b) in terms of what is actually constrained
     by 12 images, but simpler to implement — just add 12 × D = 6,144 free parameters to
     the listener model. Regularize by initializing from CLIP and using an L2 penalty on
     deviation from initialization.

   > **Answer:** Implement (c) first — initialize image embeddings from CLIP, then shift
   > them freely within the listener model. Simplest implementation, and equivalent to (a)
   > and (b) for 12 images. Options (a) and (b) deferred.

9. **Whether the task-relevant subspace is low-rank.** Separately from the projection
   question, the full CLIP space (D ≈ 512 or 768) may contain many dimensions irrelevant
   to this task. Projecting to a lower-dimensional subspace (d ≈ 10–50) could isolate
   discriminative directions and improve fit. The optimal d should be chosen by
   cross-validated predictive likelihood, not by prior assumption.

10. **Shared semantic space and posterior update (under investigation).** The assumption
   that speakers and listeners operate in the same CLIP semantic space may not hold in
   practice. Additionally, the likelihood term in Section 5 (C2 posterior update) may
   have a bug — the interaction between the listener-choice signal and the emission model
   is under review.

   > **Answer:** Addressed by having non-shared spaces for speaker and listener (separate
   > posteriors per role). The likelihood bug remains under investigation.

11. **Absence of pragmatic listener (L1).** The model uses only L_0 — a literal listener
   that responds to raw CLIP similarity. A pragmatic L_1 would reason about what else the
   speaker could have said for each image option: $L_1(i \mid u) \propto S_1(u \mid i) \cdot P(i)$,
   giving contrastive inference without needing a full L_2 stack. The current model has no
   such contrastive mechanism.

   > **Answer:** We should include a pragmatic L_1 listener that considers what else the
   > speaker could have said for each option. L_1 should be implemented via sampling:
   > for each image $j \in O$, draw candidate utterance embeddings from the space around
   > $p_L(m_{g,j})$ (the listener's current convention posterior for image $j$), score
   > them under $L_0$, and use those samples to approximate $S_1(u \mid j)$. This makes
   > the pragmatic inference tractable without enumerating all possible strings.

12. **Per-image convention spread.** σ_game is currently a scalar shared across all images.
   Replacing it with a per-image σ_{game,i} would capture nameability: low σ_{game,i}
   for images with dominant conventional labels, high σ_{game,i} for ambiguous images.
   Deferred until after the scalar version is fitted and residuals examined.

   > **Answer:** Skip for now.

13. **Where convention reshapes behavior: pull term vs. space transformation.**
   The M2 speaker utility includes a convention pull term $-\frac{1}{2\sigma^2}\|z(u) - m_{g,i}\|^2$
   that steers utterance selection toward the game convention. This may be the wrong place
   for it: rather than pulling the speaker toward m_{g,i}, the convention should reshape
   how utterance-image similarity is computed — changing what utterances *mean* for a
   given game pair, so the effect propagates through both speaker and listener naturally.
   Possible options:

   Two options to pursue:

   - **(a) Shifted speaker prior.** Treat m_{g,i} as shifting the distribution from which
     the speaker samples utterances — recent conventions are more accessible in memory, so
     the speaker generates more readily from regions of semantic space near the established
     convention. Convention pull emerges from the generative prior rather than from an
     explicit utility term.
   - **(b) Updated semantics in the speaker's listener model.** The speaker maintains an
     internal model of how the listener interprets utterances, and updates that model's
     semantics based on in-game experience. Concretely: the speaker's L_0 uses
     game-specific representations (e.g., $m_{g,i}$ in place of $x(i)$) rather than raw
     CLIP, so the speaker reasons about what will work *for this listener* given what has
     been established. Convention enters through the speaker's belief about listener
     semantics, not through a pull term in utility.

   > **Answer:** Plan to implement both (a) and (b).

14. **Other social goals.** Speakers may pursue goals beyond successful reference —
   e.g., politeness, brevity norms, rapport, or avoiding face-threatening descriptions.
   Modeling these would require extending the speaker utility function beyond informativity
   and convention pull. Probably beyond scope.

   > **Answer:** Skip for now.
   
15. ** other things we're skippinng for now ** May want to consider a) a comparison/switch to RD-RSA and b) the use of criticAL style model critique loops. 
