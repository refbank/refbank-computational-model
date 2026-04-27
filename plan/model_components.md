# CHAI Model Components — Hawkins et al. (2021)

This document outlines the computational components of the CHAI (Continual Hierarchical
Adaptation through Inference) model as described in the paper. Components are organized
by the three core capacities (C1–C3) the model formalizes, plus supporting machinery.

Adaptation notes (marked **→**) discuss what changes when moving from the paper's toy
setting (small finite utterance sets, Boolean semantics) to a real natural-language
setting like RefBank (free-text utterances, real images, open vocabulary).

---

## Lexical meaning representation

The fundamental representational choice is to replace a static, shared lexicon with a
**parameterized family** of lexical meaning functions L_φ(u, o).

- **Static baseline**: L(u, o) ∈ {0, 1} — Boolean truth-conditional function (whether
  utterance u applies to object o)
- **Parameterized**: L_φ(u, o) — same structure but indexed by φ, which encodes a
  particular system of meaning (one agent's "idiolect")
- **Parameter space**: φ is a binary utterance-object matrix; each row is one utterance's
  extension, each column is one object. For |U| utterances and |O| objects, there are
  |O|^|U| possible lexicons.
- **Simplicity prior**: P(φ) ∝ exp(-|φ|), where |φ| is the number of lexical items
  (utterances that map to at least one object). Penalizes larger lexicons.
- **Multi-word utterances**: conjunctive Boolean semantics —
  L_φ(u_i u_j, o) = L_φ(u_i, o) × L_φ(u_j, o). A conjunction is true only when both
  components apply; a contradiction (components applying to different objects) is treated
  as uninterpretable and disregarded by L_0.

> **→ Adaptation:** The binary utterance-object matrix is not feasible for natural
> language: the utterance space is open (free text, unbounded vocabulary) and meaning is
> graded rather than Boolean. The representational substrate must shift from a discrete
> lookup table to **semantic embeddings**:
>
> - Images → CLIP image encoder: z(o) ∈ R^D, L2-normalized
> - Utterances → CLIP text encoder: z(u) ∈ R^D, L2-normalized (handles single words
>   through multi-sentence utterances; stays in the same space as image embeddings)
>
> "Meaning applies to object" becomes cosine similarity: sim(z(u), z(o)). This is
> continuous and graded rather than Boolean. The question of what φ *is* in this space
> — what partner-specific parameter governs meaning — is answered differently per
> component (see C1 and C2 below).
>
> The simplicity prior has no direct equivalent. A Gaussian prior over continuous
> parameters plays an analogous role (penalizing large deviations from the community
> prototype). The conjunctive Boolean semantics for multi-word utterances is also not
> needed: embedding models already represent multi-word meaning holistically.

---

## C1: Structured uncertainty over meaning (representing variability)

Rather than a single fixed φ, agents maintain a **probability distribution over possible
lexicons** P(φ). This represents the agent's uncertainty about which system of meaning
their partner is using.

**Two-level hierarchy**:

```
Θ              — community-level "over-hypothesis": distribution over possible partners
φ_k | Θ        — partner k's specific lexicon, drawn from Θ
D_k | φ_k      — observations from interactions with partner k
```

- Θ captures shared community conventions (long-term, abstracted from many partners)
- φ_k captures partner-specific conventions ("idiolect" for the current partner)
- The structure allows beliefs about both levels to co-exist and interact

> **→ Adaptation:** The hierarchical structure transfers directly; only the parametric
> form of φ needs to change.
>
> The most natural continuous analogue:
>
> - **φ_k per game**: a set of *convention points* in embedding space — one latent mean
>   vector m_{k,i} ∈ R^D per (game k, image i) pair. This represents where in semantic
>   space the utterances for image i have converged in game k.
> - **Θ community level**: a set of *global prototypes* μ_i ∈ R^D per image i,
>   representing the typical embedding of utterances used across all games for that image.
>
> The prior P(φ_k | Θ) becomes a Gaussian: m_{k,i} ~ N(μ_i, σ²_game · I). Games start
> near the community prototype and deviate as they develop their own conventions.
>
> This structure maps directly onto the plan (δ_{g,i} = m_{k,i} − μ_i). Whether to
> also include a per-listener β (inverse temperature / calibration) as part of φ is a
> separate question addressed under the listener model.

---

## Listener model (L_0)

The **literal listener** selects an object based on the literal meaning of the utterance:

```
L_0(o | u) ∝ L(u, o)
```

Under lexical uncertainty, the listener marginalizes over possible partner lexicons:

```
L_0(o | u) ∝ L_φ(u, o)     [for a fixed φ]
```

When the listener is uncertain about the speaker's φ, they use their posterior P_L(φ|D)
to compute expected literal meaning.

> **→ Adaptation:** This is the most straightforward component to adapt and should be
> fitted first. The discrete Boolean lookup is replaced by a softmax over cosine
> similarities across the 12-image option set:
>
> ```
> L_0(c | u, O, ℓ) = softmax_{j ∈ O}( β_ℓ · cos(z(u), z(j)) )
> ```
>
> - β_ℓ is a per-listener inverse temperature capturing how sharply different listeners
>   discriminate (some listeners are noisier than others). Can be given a hierarchical
>   prior: β_ℓ ~ LogNormal(μ_β, σ_β).
> - The 12-way listener choice (which image was selected) gives a clean **multinomial
>   likelihood** that is directly observable in RefBank data. This is the primary
>   empirical signal for fitting.
>
> Whether to include a learned low-rank projection P on top of CLIP is an open question.
> Start without it and add only if the raw CLIP similarities poorly predict listener
> choices. The lexical-uncertainty version of L_0 (marginalizing over P_L(φ|D)) is
> relevant for modeling the speaker's beliefs about the listener, which only matters
> when we build the convention model (C2).

---

## Speaker model (S_1)

The **pragmatic speaker** chooses utterances via RSA utility that balances informativity
and cost.

**Utility function** (Eq. 2):

```
U(u; o, φ) = (1 - w_C) · log L_0(o | u, φ)   — informativity
           - w_C · c(u)                          — cost
```

- Informativity: log probability that the imagined L_0 selects target o given u
- Cost: c(u) = number of words in utterance u
- w_C ∈ [0, 1] controls informativity–parsimony tradeoff

**Speaker distribution** (Eq. 1):

```
S_1(u | o, φ) ∝ exp{α_S · U(u; o, φ)}
```

**With lexical uncertainty** (Eq. 4) — agents marginalize over their posterior about
the partner's φ:

```
L(o | u)  ∝ exp{ α_L · ∫ P_L(φ|D) log S_1(u | o, φ) dφ }
S(u | o)  ∝ exp{ α_S · ∫ P_S(φ|D) U(u; o, φ) dφ }
```

Parameters α_S, α_L ∈ [0, ∞) control soft-max optimality for speaker and listener.

> **→ Adaptation:** This component has two separable roles:
>
> **As a scoring function** (needed): given an observed utterance u, we can evaluate
> its utility U(u; o, φ) using embedding cosine similarity for informativity and word
> count for cost. This does not require normalizing over all possible utterances.
>
> **As a generative model** (hard, not needed initially): S_1(u | o, φ) as a proper
> distribution over utterances requires a normalizing constant that sums over all possible
> strings — intractable for open vocabulary. Generating predicted utterances would require
> a proposal-and-rerank approach (LM generates candidates, speaker utility scores them).
> This should be deferred.
>
> **For fitting to RefBank data**, the speaker model serves primarily as a linking function
> inside the *listener's* inference about the speaker's lexicon
> (P_L({o*, u', o'}_t | φ_k) = S_1(u_t | o*_t, φ_k), Eq. below). Computing this
> requires evaluating how probable the observed utterance u_t is under a given convention,
> which is feasible as a score but requires an approximation for the normalizing constant.
> The simplest approximation: treat the utterance embedding as a Gaussian emission from
> the latent convention (see C2), bypassing the full RSA speaker distribution.
>
> Cost c(u) can be computed directly from utterance text (word count, character count).
> Whether cost matters much for listener-choice fitting vs. convention fitting is an
> empirical question.

---

## C2: Online learning via partner-specific inference

Agents update their beliefs about partner k's lexicon φ_k as they accumulate observations
D_k from interaction. This is the mechanism for *ad hoc* convention formation.

**Joint posterior** (Eq. 5):

```
P(φ_k, Θ | D_k)  ∝  P(D_k | φ_k, Θ) · P(φ_k, Θ)
                  =  P(D_k | φ_k) · P(φ_k | Θ) · P(Θ)
```

- Prior term P(φ_k | Θ): partner's lexicon should be consistent with community conventions
- Likelihood term P(D_k | φ_k): how partner would actually use language under φ_k

**Marginal partner-specific posterior** (Eq. 6):

```
P(φ_k | D_k) = ∫_Θ P(φ_k, Θ | D_k) dΘ
```

This is used to make predictions about the current partner (Eq. 4) and to update Θ
(C3 below).

### Social observation structure

Each trial t provides a tuple: (o*, u', o') — intended target, utterance produced,
listener's response. The likelihood contributions are asymmetric by role:

```
Speaker infers:   P_S({o*, u', o'}_t | φ_k) = L_0(o' | u'_t)
Listener infers:  P_L({o*, u', o'}_t | φ_k) = S_1(u_t | o*_t, φ_k)
```

Both agents condition on their partner's observable behavior (not their own internal
state), using RSA as the linking function.

> **→ Adaptation:** The Bayesian update structure is preserved. The challenge is
> specifying the observation likelihood P(D_k | φ_k) in continuous space.
>
> **Two data signals in RefBank**, with different tractability:
>
> 1. **Listener choice c_t** (clean): the selected image index gives a multinomial
>    likelihood directly from the L_0 softmax. This is the primary fitting signal and
>    requires no approximation. P(c_t | u_t, O_t) = softmax_j(β · cos(z(u_t), z(j))).
>
> 2. **Speaker utterance z(u_t)** (approximate): the produced utterance provides evidence
>    about the latent convention m_{k,i}. Rather than evaluating S_1 over all possible
>    utterances, treat the observed embedding as a noisy emission:
>    ```
>    z(u_t) ~ N(m_{k,i_t}, σ²_t · I)
>    ```
>    where σ²_t can be made accuracy-dependent: larger (less informative) when the
>    listener chose the wrong image. This encodes that failed utterances are a weaker
>    signal about what the convention should be.
>
> The asymmetry of speaker vs. listener inference (the last two equations above) applies
> in principle, but for a Phase 1 model we may simplify by treating both agents as
> updating a shared convention rather than separately tracking speaker and listener beliefs.
> RefBank games often have a single speaker and one or more listeners, so the update is
> primarily from the speaker's perspective.

---

## Memory / recency discounting

Not all past observations are equally informative about the partner's *current* lexicon.
A decay term down-weights older observations:

```
P(D_k | φ_k) = ∏_{τ=0}^{T} β^τ · P({o*, u', o'}_{T-τ} | φ_k)
```

- τ = 0 is the most recent trial; decay increases for older trials
- β ∈ [0, 1] is the memory discounting parameter (β = 0.8 in simulations)
- Motivated empirically by the power function of forgetting (Wixted & Ebbesen, 1991)
- Interpretable as weighted importance sampling of recent observations

> **→ Adaptation:** This mechanism transfers unchanged. Start with β = 1 (no decay,
> all observations weighted equally) as a simplification — this is appropriate when the
> convention is assumed to be stationary within a game. Add decay only if there is evidence
> that earlier rounds are less predictive (or if modeling partner-switching, where old
> data from the same partner becomes stale).

---

## C3: Hierarchical generalization to new partners

After interacting with N partners, agents update their community-level beliefs Θ by
marginalizing over partner-specific lexicons:

**Population posterior** (Eq. 7):

```
P(Θ | D) = ∫_φ P(φ, Θ | D) dφ
```

where D = ∪_{k=1}^N D_k and φ = φ_1 × ··· × φ_N.

- "Sharing strength" / partial pooling: partner-specific data informs community beliefs
- When many partners use similar conventions, Θ concentrates on those conventions
- Novel partners are approached with priors shaped by Θ (not a blank slate)

### Hyperprior for network simulations (Phenomenon 2)

For communities of interacting agents, the partner-specific prior is made explicit:

```
φ_k(u)  ~  Categorical(Θ)          [partner lexicon drawn from community distribution]
Θ        ~  Dirichlet(λ · α)        [community distribution itself has a prior]
```

- α encodes inductive biases about the central tendency of lexicons in the population
- λ ≈ 2 controls beliefs about spread/variability (larger λ = more concentrated Θ)
- In simulations, α differs by utterance type to encode weak initial preferences

> **→ Adaptation:** The Dirichlet-Categorical hyperprior is specific to the discrete
> lexicon. In the continuous case:
>
> ```
> μ_i ~ N(0, I)                    [global prototype for image i; fitted from data]
> m_{k,i} ~ N(μ_i, σ²_game · I)   [game-specific convention; deviation from prototype]
> ```
>
> μ_i is the community-level Θ for image i — the "expected" embedding of utterances for
> that image across all games. With many games reusing the same 12 images, μ_i is
> well-identified from data. The variance σ²_game governs how much individual games
> diverge from the prototype (between-game divergence).
>
> **C3 is a later-stage concern for RefBank.** The core phenomena of interest
> (within-game convergence, step-size decrease, direction change after error) are
> explained by the within-game convention dynamics (C2). The generalization question —
> how much do games diverge, and does Θ structure that divergence — is a second-order
> question best addressed once C2 is validated.

---

## Model variants (used for comparison)

| Model | Structure | Behavior |
|---|---|---|
| **Complete-pooling** | Single shared φ for all partners | No partner-specificity; ignores partner identity |
| **No-pooling** | Independent Θ_k per partner | Starts from scratch with each partner; no generalization |
| **Partial-pooling (CHAI)** | Hierarchical φ_k ∣ Θ | Partner-specificity + gradual generalization |

Complete-pooling cannot explain partner-specificity; no-pooling cannot explain network
convergence. Only partial-pooling (CHAI) predicts both.

> **→ Adaptation:** These three variants translate directly into the continuous setting
> by varying what is shared across games:
>
> - Complete-pooling: single shared μ_i per image, no game-specific deviation (all games
>   use the same convention)
> - No-pooling: independent m_{k,i} per (game, image) with no shared prior; initialized
>   from CLIP similarities with no cross-game information
> - Partial-pooling: hierarchical m_{k,i} = μ_i + δ_{k,i} with a shared Gaussian prior
>   on δ_{k,i}
>
> These variants can be fit to RefBank and compared by held-out listener choice likelihood.

---

## Simulation parameters (recurring across paper)

| Parameter | Symbol | Role | Typical value |
|---|---|---|---|
| Speaker optimality | α_S | Sharpness of S_1 distribution | 4 or 8 |
| Listener optimality | α_L | Sharpness of L marginal | 4 or 8 |
| Cost weight | w_C | Informativity–parsimony tradeoff | 0.24 |
| Memory decay | β | Down-weighting of older observations | 0.8 |
| Dirichlet concentration | λ | Breadth of prior over community conventions | 2 |

> **→ Adaptation:** The analogous free parameters for the continuous model:
>
> | Parameter | Role |
> |---|---|
> | β_ℓ (per-listener) | Inverse temperature for L_0 softmax — how sharply listeners discriminate |
> | σ_game | Prior spread of game conventions around community prototype (between-game divergence) |
> | σ_min, σ_max | Range of accuracy-dependent emission variance in C2 |
> | d (optional) | Dimensionality of learned projection on top of CLIP |
>
> The paper's α_S, α_L, and w_C remain relevant if the full speaker model (S_1 over
> utterances) is built, but are not needed for Phase 1 (listener choice fitting +
> convention tracking from observed utterances).
