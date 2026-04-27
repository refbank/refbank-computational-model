# Conversation
- conversationId: b82563fd-00cd-4eec-b9a5-fd7eeb1e7656
- endpoint: azureOpenAI
- title: ** Computational Modeling of Reference Games
- exportAt: 10:49:28 GMT-0700 (Pacific Daylight Time)

## Options
- presetId: null
- model: gpt-5.2
- resendFiles: true
- endpoint: azureOpenAI
- endpointType: undefined
- title: ** Computational Modeling of Reference Games

## History
**User**
I have a verbal theory about modelling iterated reference games. I also have a large database of such games. Can you sketch how I could implement this theory as a computational model? (Ideally using something like semantic embeddings to obtain continuous semantics)


**GPT-5.2**
Here’s a concrete way to turn your verbal theory into a computational, embedding-based model of iterated reference games (IRGs). The key move is to treat “meaning” as a low-dimensional subspace of a general semantic embedding space, and treat within-game learning as Bayesian (or approximate Bayesian) updates over a distribution of utterances conditioned on the image and distractor set.

---

## 1) Representations: images, utterances, and a *game-specific semantic subspace*

### Base semantic space
- **Utterance embedding**: \(e(u)\in\mathbb{R}^D\) from a sentence encoder (e.g., SBERT, instructor, etc.).
- **Image embedding**: \(v(i)\in\mathbb{R}^{D_v}\) from a vision model (e.g., CLIP).  
- Optionally map both into a shared space (CLIP already does).

### “General language system” → “task subspace”
Your theory predicts only a small number of interpretable dimensions explain most variance in meanings used in a given option space.

Implement this by learning a **projection** into a low-dimensional space:
- Global projection: \(P \in \mathbb{R}^{d\times D}\) (with \(d \ll D\)).
- Or *contextual/game-conditioned* projection \(P_g\) that depends on the current option set \(O_g\).

Then define:
\[
z(u) = P\,e(u),\quad z(i)=P_v\,v(i)
\]
and do all pragmatic inference in the \(d\)-dim space.

How to get \(P\):
- **Data-driven**: collect utterances from early rounds (your “weak prior”; e.g., R1 descriptions) and fit PCA / factor analysis / supervised low-rank regression to predict targets.
- **Option-conditioned**: learn a small network \(P_g = f(\{v(i): i\in O_g\})\) that outputs a projection emphasizing discriminative directions for that option set.

This operationalizes “only a relevant subspace is considered, conditioned on the option space”.

---

## 2) Semantics: a continuous listener model \(L_0\)

Define a literal listener that maps an utterance embedding to a distribution over images in the current option set \(O\).

A common choice:
\[
L_0(i\mid u,O) \propto \exp\Big(\beta \cdot \mathrm{sim}(z(u), z(i))\Big)
\]
where \(\mathrm{sim}\) could be cosine similarity or negative squared distance, and \(\beta\) is an inverse temperature.

This gives you:
- graded meaning,
- confusability structure (“more distinct directions are favoured”),
- and a way to predict accuracy.

---

## 3) Pragmatics: RSA-style speaker \(S_1\) with costs

Speaker chooses utterance balancing informativeness and cost:
\[
S_1(u\mid i,O) \propto \exp\Big(\alpha\cdot \log L_0(i\mid u,O) \;-\; \lambda \cdot C(u)\Big)
\]

Where cost can include:
- **production cost**: length, character count, word count (modality-dependent as your theory suggests),
- **memory/complexity cost**: surprisal under a base LM, or embedding norm / “novelty” penalty,
- **reuse bonus** (optional): discount cost for previously used utterances in the same game (captures conventionalization).

This gives you explicit handles for:
- accuracy–complexity tradeoff,
- modality differences (different \(C(u)\)).

---

## 4) Iteration / learning: updating a *within-game* utterance distribution

Your theory posits: for each image there is a weak prior over utterances (approx by R1), then after each observed utterance, probability mass shifts toward it; repeated resampling yields convergence and decreasing step size.

A simple implementation is to maintain, per game \(g\) and target image \(i\), a *distribution over utterance meanings* in embedding space.

### Option A: Prototype + shrinking variance (fast, interpretable)
Maintain a Gaussian over utterance embeddings for each target:
- Mean \(\mu_{g,i}\), covariance \(\Sigma_{g,i}\) in the subspace.

Initialize from R1:
\[
\mu_{g,i}^{(0)}=\text{mean}(z(u)\text{ from round 1 for }i),\quad \Sigma_{g,i}^{(0)}=\text{cov}(\cdot)
\]

Update after observing an utterance \(u_t\) used for \(i\):
\[
\mu^{(t+1)} = (1-\eta_t)\mu^{(t)} + \eta_t z(u_t)
\]
with a decreasing learning rate \(\eta_t\) (e.g., \(\eta_t = \frac{1}{t+k}\)) → predicts **step size decrease**.

Optionally shrink covariance:
\[
\Sigma^{(t+1)} = (1-\gamma)\Sigma^{(t)} + \gamma \,(z(u_t)-\mu^{(t)})(z(u_t)-\mu^{(t)})^\top
\]
and/or anneal \(\Sigma\) downward over time → predicts **within-game convergence**.

Then generate candidate utterances by sampling a “meaning point” \(m\sim \mathcal{N}(\mu_{g,i},\Sigma_{g,i})\), and choosing a discrete utterance \(u\) whose embedding is close to \(m\) and scores well under \(S_1\).

### Option B: Dirichlet-multinomial over a candidate set (more discrete)
Construct a candidate set \(\mathcal{U}(i,O)\) (e.g., top-k captions from an LM prompted on the image; or retrieved utterances from similar images).
Initialize Dirichlet prior from R1 counts, update with observations:
\[
\theta_{g,i} \sim \text{Dir}(\alpha_{0} + \text{counts})
\]
Resample utterances from \(\theta_{g,i}\) but filtered/weighted by RSA utility. This directly implements “probability mass moves onto observed utterance”.

---

## 5) Miscalibration and “change direction after error”

When there’s a communicative failure, your theory predicts the speaker updates their model of the listener and changes direction.

Implement this with an explicit “listener calibration” variable:
- Maintain \(\beta_g\) (how sharp/confusable listener is), or a noise parameter \(\sigma_g\) in the similarity function.
- After an error, update toward “listener is noisier than I thought”, and encourage exploration.

Concretely:
- After failure, temporarily increase an exploration temperature or add a novelty bonus:
  \[
  U(u) = \alpha \log L_0(i\mid u,O) - \lambda C(u) + \kappa \cdot \text{Novelty}(u)
  \]
where Novelty could be distance from current prototype \(\mu_{g,i}\) (i.e., “change direction” = move in embedding space).

This yields a measurable prediction: after an error, the next successful utterance should have a larger semantic displacement from the prior centroid/prototype than after a success.

---

## 6) Between-game divergence and idiosyncrasy

Between-game divergence can fall out of:
- stochastic initialization (different R1 seeds),
- different option-conditioned subspaces \(P_g\),
- different early random walk trajectories.

Operationalize “idiosyncrasy” as something like:
- distance between game-specific prototype \(\mu_{g,i}\) and the global prototype \(\bar{\mu}_i\),
- or entropy / concentration of the within-game utterance distribution.

You can also formalize:
- **Idiosyncrasy increase**: within-game distribution becomes more peaked on a specific utterance region (lower entropy / lower covariance).
- **Opacity maintenance**: naïve listener accuracy stays similar (you can approximate naïve listener with a generic embedding-only \(L_0\) without game adaptation, or with held-out human guess data if you have it).

---

## 7) Training and evaluation with your database

### What you can fit
Parameters like: \(d,\alpha,\beta,\lambda\), learning-rate schedule \(\eta_t\), covariance annealing, error-exploration strength \(\kappa\), modality-specific cost weights.

Fit by maximizing the likelihood of observed utterances (or their embeddings) under the model:
\[
\max_\theta \sum_{g,t} \log S_1^\theta(u_{g,t}\mid i_{g,t}, O_{g,t}, \text{state}_{g,t})
\]
where state includes \(\mu_{g,i}^{(t)}\), etc.

If you don’t want to generate text, you can model **utterance embeddings directly** (treat observed \(z(u)\) as samples from the evolving Gaussian/prototype), then separately use RSA only for predicting choices/accuracy.

### Key diagnostics tied to your predictions
- **Within-game convergence**: decreasing pairwise distance between successive utterance embeddings for same referent.
- **Step size decrease**: \(||z(u_{t})-z(u_{t-1})||\) decreases with \(t\).
- **Within-game differentiation**: distances between prototypes for different images within the same game increase.
- **Between-game divergence**: for same image across games, prototypes drift apart across games.
- **Cost effects**: utterance length/PoS shifts by modality; redundancy decreases (type-token changes, synonym clustering changes).
- **Error response**: larger semantic displacement after incorrect trials.

---

## 8) Practical implementation sketch (minimal pipeline)

1. **Precompute embeddings**
   - \(e(u)\) for every utterance.
   - \(v(i)\) for every image (CLIP).
2. **Learn / choose a projection**
   - Start with PCA on all utterance embeddings within each option set type, or a global PCA to \(d=10..50\).
3. **Define \(L_0\)**
   - similarity-based softmax over option set.
4. **Define \(S_1\)**
   - RSA utility with cost (length, modality).
5. **Within-game state**
   - prototype/covariance per referent (or Dirichlet over retrieved candidates).
6. **Update rules**
   - after each round, update state; on error trigger exploration shift.
7. **Fit parameters**
   - likelihood of utterance choices, plus auxiliary loss predicting success.
8. **Test predictions**
   - convergence, step size decrease, differentiation/divergence, modality effects.

---

## Two clarifying questions (so the model matches your dataset)
1) Do you have the *full candidate set* of possible utterances at each turn (e.g., speakers typed freely), or only the chosen utterance? This affects whether we model choice among candidates vs. generate/retrieve candidates.
2) Do you have “naïve listener” judgments (or can you simulate them), to quantify opacity \(P(\text{meaning}\mid \text{name})\)?


**User**
Participants typed freely. I would like to not "bake in" the predictions into the model (e.g., step size decrease / shrinking covariance should "emerge out of" Bayesian updating, as should the "change in direction after error"—so the updating should depend on the accuracy of the listener)


**GPT-5.2**
You can get both effects to *emerge* by (i) treating the speaker’s internal state as a Bayesian posterior over a latent “convention” for each referent, and (ii) conditioning posterior updates on whether the listener succeeded. Then diminishing step size comes from posterior concentration, and “change in direction after error” comes from Bayesian *rejection/anti-evidence* (or from updating uncertainty/noise upward and re-optimizing utility).

Below is a workable model skeleton that stays close to RSA and continuous semantics, without hard-coding shrinkage schedules.

---

## 1) Latent convention as a continuous parameter per referent (in embedding space)

Let each image/referent \(i\) in game \(g\) have a latent “conventional meaning point”
\[
m_{g,i}\in\mathbb{R}^d
\]
interpretable as where in semantic space the pair’s convention will land (a game-specific MAP away from the global centroid).

Prior:
\[
m_{g,i}\sim \mathcal{N}(m^{(0)}_i,\Sigma^{(0)}_i)
\]
where \(m^{(0)}_i,\Sigma^{(0)}_i\) come from round-1 behavior across games (your weak prior) or from a global model (e.g., image embedding mapped into the same space).

---

## 2) Free text generation handled via *proposal + rerank* (so you don’t need a closed candidate set)

Since speakers type freely, treat utterance choice as approximate optimization:

1. Propose \(K\) candidates \(u^{(1..K)}\) from an LLM or a captioner, conditioned on the target image + optional memory of previous successful utterances.
2. Embed each candidate \(z(u)\).
3. Choose by a probabilistic softmax over utility (RSA-like), so the model is still generative over discrete strings via the proposal mechanism.

You will evaluate likelihood **with importance sampling** (common in models with intractable action spaces): approximate the speaker distribution over the proposed set.

---

## 3) Listener model grounded in embeddings

Literal listener:
\[
L_0(j\mid u,O) \propto \exp(\beta\,\mathrm{sim}(z(u),z(j)))
\]

Pragmatic speaker uses that but also tries to match the latent convention \(m_{g,i}\) (this is what creates within-game convergence):

\[
S(u\mid i,O,m_{g,i}) \propto \exp\Big(
\alpha\log L_0(i\mid u,O)\;-\;\lambda C(u)\;-\;\tfrac{1}{2\sigma^2}\|z(u)-m_{g,i}\|^2
\Big)
\]

Interpretation:
- The RSA term prefers discriminative directions in the current option set.
- The \(\|z(u)-m\|^2\) term is a *convention/coordination* drive: reuse a stable semantic region once established.
- \(\sigma\) governs how tightly utterances cluster around the convention.

Nothing here forces shrinking; it will emerge because the posterior over \(m_{g,i}\) concentrates as you get evidence.

---

## 4) Bayesian update driven by success/failure (accuracy-dependent learning)

You observe only:
- utterance \(u_t\)
- option set \(O_t\)
- whether listener succeeded \(y_t\in\{0,1\}\) (and ideally which choice they made)

Update the posterior over \(m_{g,i}\) via Bayes:
\[
p(m \mid \text{history}) \propto p(m)\prod_t p(u_t, y_t \mid m, i,O_t)
\]

Factor as:
\[
p(u_t, y_t \mid m) = p(u_t \mid m,i,O_t)\;p(y_t \mid u_t,i,O_t,m)
\]

- \(p(u_t\mid m,i,O_t)\) is the speaker model \(S(\cdot)\) (or a simplified emission model, see below).
- \(p(y_t\mid u_t,\dots)\) ties learning to accuracy.

### Key move: make failure *negative evidence* about the current convention
A simple accuracy likelihood is:
\[
p(y=1\mid u,i,O,m)=L_0(i\mid u,O)
\]
(i.e., success probability is the listener’s model probability).

But to make it depend on \(m\), you want the success probability to reflect whether \(u\) is “on-convention” (because miscalibration means your assumed mapping is off). One way:

\[
p(y=1\mid u,i,O,m)= L_0(i\mid u,O)\cdot \exp\!\Big(-\tfrac{1}{2\tau^2}\|z(u)-m\|^2\Big)
\]

So:
- if you used an utterance far from the convention you’re trying to establish, success is less likely;
- if failure happens after an on-convention utterance, that’s evidence the convention \(m\) is wrong (or that listener noise is higher), pushing the posterior away.

### Inference
Exact posterior is not analytic because of the RSA term, but you can do:
- **Laplace approximation** (optimize for MAP \(m^\*\), approximate covariance by Hessian inverse), or
- **particle filter** per game/referent (a few dozen particles in \(\mathbb{R}^d\) works).

Posterior concentration (shrinking covariance) then *emerges* from repeated evidence under either method. Consequently, expected step size in \(m\) updates decreases automatically.

---

## 5) Why “step size decrease” emerges

Under conjugate-ish conditions (Gaussian prior + approximately Gaussian likelihood around MAP), posterior variance decreases roughly like \(1/n\). Even with Laplace/particles, you’ll see:
- early observations move MAP a lot,
- later ones move it less,
because the posterior is sharper.

Crucially, you don’t set a learning-rate schedule; it is implied by posterior curvature / particle weights.

---

## 6) Why “change direction after error” emerges

When \(y=0\), the likelihood term \(p(y=0\mid u,\dots,m)=1-p(y=1\mid \cdot)\) downweights values of \(m\) that made success likely for the utterance you actually produced.

That has two effects:

1. **Posterior shifts away from the region supporting the failed utterance**  
   If the utterance was close to the current \(m\), failure pushes \(m\) away (or inflates uncertainty if you also infer noise). Next utterance chosen from \(S(u\mid m)\) will tend to point in a different direction in embedding space.

2. **Uncertainty/noise can increase after failure (optional, but still Bayesian)**  
   Put a prior on listener noise \(\beta\) (or an added noise parameter), infer it too:
   \[
   \beta_g \sim p(\beta),\quad p(y_t\mid u_t,\beta)=\text{Bernoulli}(L_0^\beta(i\mid u_t,O_t))
   \]
   Failure increases posterior mass on lower \(\beta\) (noisier listener), which makes the RSA term push toward more separated/distinct utterances—again changing direction without an explicit “explore-after-error” rule.

---

## 7) Practical simplification: avoid modeling the full speaker likelihood \(p(u\mid m)\)

Because free text makes \(p(u\mid m)\) hard, you can let *learning* depend only on the success likelihood and a lightweight “emission” model:

- Use the observed utterance embedding as a noisy observation of \(m\):
  \[
  z(u_t) \sim \mathcal{N}(m,\sigma^2 I)
  \]
- And use success/failure to gate how much weight that observation gets, via a mixture:
  \[
  p(z(u_t)\mid m,y_t)=
  \begin{cases}
  \mathcal{N}(z(u_t); m,\sigma^2I) & y_t=1\\
  \mathcal{N}(z(u_t); m,\sigma^2_{\text{fail}}I) & y_t=0
  \end{cases}
  \]
with \(\sigma_{\text{fail}}>\sigma\) and (optionally) learn both. This is still Bayesian; failure carries weaker/ambiguous evidence about the convention, so posterior moves less toward the failed point and may drift elsewhere when combined with other constraints (like discriminability).

If you also want failure to be *anti-evidence* (actively push away), use a robust likelihood such as a “repulsive” component:
\[
p(z(u_t)\mid m, y=0)=\text{const}\cdot (1-\exp(-\|z(u_t)-m\|^2/2\rho^2))
\]
(implemented stably as a contrastive term in the log-likelihood). That makes “change direction after error” very direct.

---

## 8) Fitting and checking emergent predictions

Fit global parameters (\(d,\alpha,\lambda,\sigma,\tau\), and possibly prior parameters) by maximizing predictive likelihood of:
- next utterance embedding (or choice among proposals),
- and next-round success.

Then test emergent patterns (held-out games):
- decreasing \(\|m_{t}-m_{t-1}\|\) (or decreasing distance between consecutive utterance embeddings),
- larger semantic displacement following \(y=0\) than \(y=1\),
- within-game differentiation (referent conventions separate more within option set),
- between-game divergence (different games settle on different \(m_{g,i}\)).

---

### Two quick questions (to pick the cleanest inference route)
1) Do you have the *listener’s selected image* on each trial, or only success/failure? (Having the selected distractor lets you use a multinomial likelihood, which is much sharper than a Bernoulli.)
2) Are the same images reused across many games? If yes, we can put a hierarchical prior \(m^{(0)}_i\) shared across games (global prototype + game-specific deviation), which makes between-game divergence a principled random effect.


**User**
We do have listener's selected image. In fact in some games there were multiple listeners and/or the speaker rotated among players. Yes some of the same images are reused across many games (we might in fact narrow down to the 12 most common images)


**GPT-5.2**
Great—having the *selected image* (not just success) and repeated images lets you write a fairly clean hierarchical Bayesian model with multi-agent calibration.

## 1) Core objects (continuous semantics + hierarchical conventions)

Embed:
- utterance \(u\to z(u)\in\mathbb{R}^d\) (e.g., SBERT then optional low-rank projection)
- image \(i\to x(i)\in\mathbb{R}^d\) (e.g., CLIP image embedding projected to same \(d\))

Hierarchical “convention” for each image:
- global prototype for image \(i\): \(\mu_i\)
- game-specific deviation: \(\delta_{g,i}\)
- (optional) speaker-/listener-specific deviation if players rotate: \(\delta_{p,i}\)

So the latent conventional point used in a particular interaction can be:
\[
m_{g,p,i} = \mu_i + \delta_{g,i} + \delta_{p,i}
\]
with Gaussian priors \(\delta_{g,i}\sim \mathcal N(0,\Sigma_g)\), \(\delta_{p,i}\sim \mathcal N(0,\Sigma_p)\). Between-game divergence is then literally \(\delta_{g,i}\).

## 2) Listener choice likelihood (uses selected image)

For a trial with option set \(O\), utterance \(u\), listener \(\ell\) chooses image \(c\in O\).

A good continuous listener is:
\[
P(c\mid u,O,\ell) = \text{softmax}_{j\in O}\Big(\beta_\ell \cdot \mathrm{sim}(z(u), x(j))\Big)
\]
where \(\beta_\ell\) is a listener-specific inverse temperature (calibration/“noise”). Multiple listeners are easy: each has their own \(\beta_\ell\) (and you can partial-pool them).

This already yields “distinct directions are favored” because the softmax penalizes confusable option sets.

## 3) Speaker model that couples utterances to the evolving convention (but still free text)

Because the action space is unbounded, do **proposal + rerank**:

- Propose \(K\) candidate utterances \(\{u_k\}\) from an LLM/captioner conditioned on (image \(i\), maybe previous successful utterances in that game).
- Score each candidate with a utility:
\[
U(u_k)= \alpha \log P_{\text{listener}}(i\mid u_k,O,\hat\ell)\;-\;\lambda C(u_k)\;-\;\frac{1}{2\sigma^2}\|z(u_k)-m_{g,p,i}\|^2
\]
- Choose with softmax over \(U\).

Here \(\hat\ell\) can be:
- the actual listener on that trial (if speaker knows who it is), or
- a speaker belief over possible listeners (mixture over \(\ell\)’s \(\beta_\ell\)) if partners rotate.

The \(\|z(u)-m\|\) term is the “coordination pressure”; it doesn’t hard-code shrinking—shrinking comes from posterior learning of \(m\).

## 4) Bayesian updating that depends on outcomes (and yields direction-change after errors)

You update posteriors over latent \(m_{g,p,i}\) (equivalently \(\delta_{g,i},\delta_{p,i}\)) using *the listener’s choice*.

The key likelihood term for updating is the categorical choice:
\[
\log P(c_t\mid u_t,O_t,\ell_t) = \beta_{\ell_t}\mathrm{sim}(z(u_t),x(c_t)) - \log\sum_{j\in O_t}\exp(\beta_{\ell_t}\mathrm{sim}(z(u_t),x(j)))
\]

How does this make “change direction after error” emerge?
- If the listener picked a distractor \(c_t\neq i_t\), the gradient of this log-likelihood pushes the model toward utterance regions that would have made \(i_t\) higher than the chosen distractor under the similarity geometry. When you propagate that back to the posterior over \(m\), the next utterance sampled/optimized under the updated posterior tends to move away from the failed region toward a more discriminative direction for that option set.
- If you also infer \(\beta_{\ell}\), errors will shift posterior mass toward lower \(\beta_\ell\) (noisier listener), which in turn makes the speaker utility favor more separative utterances.

### Inference method (practical)
Use one of:
- **Laplace/MAP per game-image**: optimize \(\delta_{g,i}\) (and \(\delta_{p,i}\)) by gradient ascent on log posterior; approximate covariance by Hessian inverse → posterior concentration gives decreasing step sizes.
- **Variational inference**: \(q(\delta_{g,i})=\mathcal N(\mu,\Sigma)\) with an ELBO; \(\Sigma\) will shrink where data are informative.
- **Particle filter** within each game if you want online updating; particles reweight by the categorical likelihood from the observed listener choices.

No explicit learning-rate schedules required: diminishing updates come from increasing posterior curvature / decreasing posterior variance.

## 5) Handling multiple listeners and rotating speakers

Because you sometimes have multiple listeners and rotating speakers, partial pooling is important:

- Listener calibration: \(\beta_\ell \sim \text{LogNormal}(\mu_\beta,\sigma_\beta)\)
- Speaker cost sensitivity: \(\lambda_s\) or production-noise per speaker, also hierarchical.
- Player-specific semantic quirks: \(\delta_{p,i}\) (optional; might be overkill unless you have lots of data per player).

Then:
- within-game convergence = learning \(\delta_{g,i}\) shared across whoever is speaking/listening in that game
- player idiosyncrasy = \(\delta_{p,i}\)
- miscalibration effects = inference over \(\beta_\ell\) and/or speaker beliefs about \(\beta\)

## 6) Narrowing to 12 common images (good idea)

With 12 images reused many times, you can make \(\mu_i\) very well estimated, which helps:
- separate “generic” semantics (global prototype) from game drift (\(\delta_{g,i}\))
- quantify between-game divergence cleanly: variance of \(\delta_{g,i}\) across games for each image
- test whether the effective semantic subspace is low-dimensional by fitting \(d\) and checking held-out predictive likelihood.

## 7) Minimal implementation plan

1. Compute embeddings: CLIP for images, SBERT for utterances; learn a linear map so they’re comparable (or use CLIP text encoder for utterances).
2. Fit the listener-choice model \(P(c\mid u,O,\ell)\) first (estimate \(\beta_\ell\), maybe a global projection).
3. Add hierarchical conventions \(\mu_i,\delta_{g,i},\delta_{p,i}\) and fit by maximizing posterior / ELBO using the full sequence data.
4. Add the speaker model (proposal+rerank) if you want to generate/predict utterances; otherwise you can still evaluate convergence/direction-change using posterior movement in \(m\) space given observed utterances and choices.

If you tell me (a) typical option-set size, and (b) whether utterances are short phrases vs full sentences, I can recommend a specific embedding choice (CLIP text vs SBERT) and an inference stack (Pyro/NumPyro vs Stan vs PyTorch VI).


**User**
Option set size is always 12; utterances range from single words to multi-sentence utterances


**GPT-5.2**
With fixed 12-way choice, you can lean heavily on a CLIP-style shared space (handles single words through paragraphs reasonably) plus a hierarchical Bayesian state model that’s updated by the observed 12-way listener choice.

## Recommended semantic backbone
- Use **CLIP image encoder** for images \(x(i)\).
- Use **CLIP text encoder** for utterances \(z(u)\). This avoids having to learn a cross-modal map and stays stable across very short and long texts.
- Optionally learn a low-rank projection \(P\in\mathbb{R}^{d\times D}\) on top of CLIP to capture the “game-relevant subspace”; choose \(d\) by predictive performance.

Similarity: cosine in the projected space.

## Listener choice model (observed selected image)
For trial \(t\) with option set \(O_t\) (size 12), utterance \(u_t\), listener \(\ell_t\), chosen image \(c_t\):
\[
P(c_t \mid u_t,O_t,\ell_t)=\text{softmax}_{j\in O_t}\big(\beta_{\ell_t}\,\cos(Pz(u_t),Px(j))\big)
\]
Hierarchical \(\beta_{\ell}\) captures some listeners being noisier. This piece is easy to fit and gives you calibrated trial-by-trial likelihood.

## Game-specific convention as a latent drift (what learns over repetitions)
Let each game \(g\) and image \(i\) have a latent “conventional point” \(m_{g,i}\) (in the same projected space). Put a hierarchical prior anchored at the global prototype for that image:
\[
\mu_i \sim \mathcal N(0,I),\quad m_{g,i}\sim \mathcal N(\mu_i,\Sigma_{\text{game}})
\]
Now connect utterances to the convention (this is what yields within-game convergence) with an emission model:
\[
P(Pz(u_t)\mid m_{g,i_t}) = \mathcal N(Pz(u_t); m_{g,i_t}, \sigma^2 I)
\]

### Accuracy-dependent updating (so “change after error” emerges)
Use the *listener choice* likelihood as additional evidence about whether the produced utterance was a good conventional signal. A simple way that makes failures act like “anti-evidence” is to weight the emission by how well the utterance worked:

Let
\[
s_t = P(i_t \mid u_t, O_t, \ell_t)
\]
(the model-implied probability of the true target under the listener choice model; you can compute this exactly from the 12-way softmax). Then define an effective emission precision:
\[
\sigma_t^2 = \sigma_{\min}^2 + \frac{\sigma_{\max}^2-\sigma_{\min}^2}{\text{clip}(s_t,\epsilon,1)}
\]
So:
- if \(s_t\) is high (clear success), the observation is treated as reliable and posterior over \(m_{g,i}\) tightens quickly;
- if \(s_t\) is low (error or near-error), the observation is treated as unreliable—posterior won’t move much toward that utterance, and the next utterance (which is generated/optimized from the posterior) will tend to move elsewhere.

This is still Bayesian updating; you’re not hard-coding “shift after error”, you’re letting the *likelihood width* be implied by communicative evidence.

If you want genuinely repulsive updates after errors (not just “ignore bad evidence”), add a second likelihood term for failures that downweights conventions near the failed utterance embedding (a contrastive/robust observation model). But often the reliability-weighting already produces a direction change once you combine it with the discriminability pressure below.

## Discriminability pressure that’s option-set specific (12-way)
To make the “direction” respond to the current distractors, include an option-set discriminability factor in the posterior update or in the speaker policy. E.g., define a utility-like likelihood:
\[
P(\text{good utterance}\mid m_{g,i},O_t,\ell_t)\propto \exp\big(\alpha \log P(i_t\mid u_t,O_t,\ell_t)\big)
\]
and treat it as an additional term when inferring \(m_{g,i}\). Intuitively: if an utterance led the listener to pick a specific distractor, posterior mass shifts toward conventions that would separate target from that distractor in this option geometry—this is what yields “change direction after error” in a targeted way.

## Inference that gives “shrinking” and “step size decrease” for free
Use per-(game,image) Gaussian posteriors with either:
- **Laplace updates** (MAP + Hessian) on \(m_{g,i}\), or
- **variational Gaussians** \(q(m_{g,i})=\mathcal N(\hat m,\hat\Sigma)\).

Posterior covariance \(\hat\Sigma\) will shrink as repeated reliable evidence accumulates, and MAP updates \(\|\hat m_t-\hat m_{t-1}\|\) will naturally decrease—no learning-rate schedule needed.

## Speaker generation (optional) with free text
If you want to actually *predict utterances*, do proposal+rERank:
1) propose \(K\) candidates from an LLM (condition on target image and maybe previous successful utterances in-game),
2) choose via softmax over:
\[
U(u)= \alpha \log P(i\mid u,O,\text{listener mixture}) - \lambda C(u) - \tfrac{1}{2\sigma^2}\|Pz(u)-\mathbb E[m_{g,i}]\|^2
\]
This yields:
- convergence: the last term concentrates utterances near the inferred convention,
- differentiation: the first term favors distinctiveness among the 12 options,
- modality/cost effects: \(C(u)\) can be length/characters/etc.

## What I’d implement first (fastest path)
1) CLIP-embed all images and utterances; fit the 12-way listener model with hierarchical \(\beta_\ell\).
2) Add the latent \(m_{g,i}\) layer and do variational inference to see if you get:
   - decreasing posterior variance over repetitions,
   - larger semantic displacement after low \(s_t\) (errors/near-errors),
   - within-game clustering of utterances in embedding space.

If you share how many repeats per image per game you typically have (e.g., 3? 10? 30), I can suggest whether Laplace-per-game or VI/particle filtering will be more stable.


