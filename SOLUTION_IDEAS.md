Straight Through Estimator was the easy option. We tried that and it didn't seem to be working too well. Other options we need to explore:

# Current Research Status: 2026-05-14

Phase 7 has four important natural `0..19` result-level results:

```text
exact_grid_seed_replication_negative
multisample_result_space_policy_gradient_stage0_alignment_negative
result_space_expected_answer_loss_alignment_negative
gradient_friendly_decoder_stage0_pass_stage1_exact_marginal_discovery_negative
```

Exact full-grid upstream-open result-boundary teaching can learn hard result
requests. Seed `2` produced a retained positive, and CLI seeds `4` and `5`
both relearned Stage 1 result requests near `1.0`. However, target-off
continuation did not robustly clear the strict `90%` retention gate for seeds
`4` and `5` (`87.0%` and `88.2%` best post-start retention). Semantic decoder
movement stayed exactly `0.0`.

The follow-up estimator-family gate implemented result-space REINFORCE and
tested exact-grid `K=16` multi-sample policy gradient with per-prompt and
leave-one-out baselines. The plumbing is live: result-proj/upstream gradients
were nonzero, semantic decoder gradient stayed exactly `0.0`, and the
per-prompt/leave-one-out baselines reduced advantage variance versus the old
global EMA baseline. But the policy-gradient estimate was anti-aligned with
the boundary-target ceiling at initialization (`result-proj cosine=-0.0945`;
upstream `cosine=-0.1108`), so vanilla multi-sample result-space PG should
not receive long-run training budget without first fixing gradient alignment.

The exact result-marginal answer-loss gate then showed that the raw
expected-cost objective itself is anti-aligned with the boundary-target
ceiling, not merely sampled poorly. Raw exact expected answer-loss gradients
had nonzero result-proj/upstream L2 and semantic decoder gradient exactly
`0.0`, but exact-vs-boundary cosine was negative (`-0.0978` result-proj,
`-0.1231` upstream). Sampled PG was strongly aligned with that raw exact
gradient (`0.9577` result-proj, `0.9736` upstream). Detached z-score
normalization weakly improved result-head cosine (`0.0764`) but still failed
the upstream gate (`-0.0007`), so Stage 1 exact-marginal training was skipped.

The gradient-friendly decoder gate showed that decoder geometry can flip the
local sign but still fail discovery. A contrastive-margin decoder made exact
expected-cost gradients positively aligned with the boundary ceiling
(`0.1204` result-proj cosine, `0.0484` upstream cosine), with forced
true/oracle exact accuracy still `1.0` and semantic decoder gradient exactly
`0.0`. But exact-grid result-marginal training with that decoder frozen
collapsed to a low-entropy wrong result policy (`0.0750` learned-best hard
result accuracy; final exact-match `0.085`).

Helpful now:

- exact full-grid coverage;
- upstream-open result-boundary teaching;
- target-off retention diagnostics as stability probes, not novelty claims;
- the boundary-target branch as a supervised ceiling/control for new
  estimator-family comparisons;
- three-way fixed-grid gradient-agreement diagnostics against exact
  result-marginal and boundary-target controls for any new estimator;
- qualitatively different learning signals: surrogate/shadow-calculator
  gradients, synthetic gradients/direct feedback alignment, learned
  shadow-gradient modules, or a new estimator that first passes the alignment
  gate and remains stable during early training.
- decoder/loss-geometry checks only if they introduce a genuinely stronger
  backward channel than weak local expected-cost sign alignment.

Not helpful as next steps:

- oracle/readout reruns for natural `0..19`;
- random-resampled boundary-target repeats;
- frozen linear or frozen MLP result-head variants;
- the skipped MLP rescue from the full-grid task;
- more target-off retention reruns that do not introduce a new mechanism;
- canonical-query/protocol stabilization as if exact-grid retention had
  robustly replicated.
- vanilla multi-sample result-space PG long runs while the Stage 0
  PG-vs-boundary cosine remains negative or near zero.
- raw exact result-marginal expected-cost long runs, or learned-baseline
  methods that merely estimate the same raw expected-cost gradient.
- more decoder-only calibration branches that only sharpen forced true results
  or weakly improve ordinary expected-cost geometry.

The key implication is that result-space parameterization alone was not enough,
and ordinary answer-loss expected-cost optimization is not rescued by weak
decoder calibration. The boundary-target branch proves that the natural result
request is representable and teachable, but the next useful branch should
change the backward signal rather than only reducing score-function variance or
reshaping the same expected-cost objective.

Phase 7 update after the first biased-backward-channel tests: plain
output-projection boundary feedback and a fit-once linear shadow-feedback map
both failed Stage 1 discovery. The linear shadow map produced very strong
initial alignment with the boundary ceiling (`0.9983` result-proj cosine,
`0.9854` upstream cosine), but its 200-step frozen-map smoke reached only
`0.040` final exact match. Future shadow-gradient work should use heldout
validation and/or online-trained shadow modules, with early Stage 1 lift above
the `0.16` boundary-feedback baseline before long runs.

Heldout validation has now shown that same-batch linear shadow alignment is
not a reliable gate: a deterministic `320/80` split kept train result-proj
cosine near `0.9981`, but heldout result-proj cosine fell to `0.2622`. Next
shadow-gradient work should add result-policy state and online training rather
than reusing a fixed linear map.

Selected next direction:

```text
heldout-validated or online-trained shadow-gradient modules
```

Rationale: the current result-space action is representable, and one decoder
variant made the local expected-cost gradient point in the right signed
direction, but exact expected-cost training still collapsed. The next branch
should supply an explicitly biased/useful backward signal at the calculator
boundary while retaining the exact-grid boundary-ceiling diagnostic as a gate.

# Alternatives to the Straight-Through Estimator

Great follow-up — and the right question to ask, because STE is just the *cheapest* solution to a much more general problem. Let me first sharpen what the problem actually is, then give you the menu.

## First, clarify which neurons are hard to train

In your calculator setup there are two populations:

- **Downstream neurons** (receiving the calculator's output): these are *easy*. The calculator's output is just an input feature to them — backprop works normally as long as you treat that input as a constant (stop-gradient).
- **Upstream neurons** (the ones whose activations get fed *into* the calculator): these are the hard ones. The calculator blocks the gradient, so they have no learning signal telling them *how to encode their representation into something the calculator can use*.

Every method below is really about getting a learning signal to those upstream encoder neurons. STE is one answer. Here are the others.

---

## The full menu of alternatives

### 1. **REINFORCE / Score-function estimators (the unbiased gradient family)**

Treat the calculator's input as a *stochastic action* sampled from a distribution parameterized by the upstream neurons. Then the policy-gradient identity gives you:

$$\nabla_\theta \mathbb{E}[L] = \mathbb{E}[L \cdot \nabla_\theta \log p_\theta(a)]$$

You don't need the calculator to be differentiable — you just need to evaluate the loss after running it. This is **unbiased**, unlike STE. The catch is **enormous variance**, which is why people developed control-variate variants: NVIL, MuProp, REBAR, RELAX. If your calculator outputs are discrete (digits, tokens), this is the most theoretically clean option.

Phase 7 update: vanilla result-space REINFORCE with exact-grid `K=16`
multi-sample advantages did not pass the gradient-alignment gate against the
boundary-target ceiling. Do not treat longer vanilla REINFORCE schedules as the
next obvious move. If returning to score-function estimators, add a stronger
control variate or critic and first check PG-vs-boundary cosine on the fixed
grid.

### 2. **Gumbel-Softmax / Concrete relaxation**

If the upstream neurons need to produce *discrete* inputs to the calculator (e.g., digit tokens), you can replace the discrete sampling with a continuous, temperature-controlled relaxation. You then anneal temperature → 0 during training so the forward pass eventually matches the discrete one. This is biased but lower-variance than REINFORCE. A clean fit for "encode a number into the calculator" tasks.

### 3. **Differentiable surrogate / "shadow" emulator**

Train a small neural network to *mimic* the calculator end-to-end, then use the surrogate during backward passes while using the real calculator on the forward pass. This is essentially the same idea as differentiable physics emulators in scientific ML. The surrogate provides smooth gradients; the real calculator provides exact outputs. The risk is gradient–output mismatch — your gradients point the encoder toward "what fools the surrogate," not "what the real calculator wants."

### 4. **Zeroth-order / finite-difference gradient estimation**

Query the calculator at perturbed inputs and estimate the gradient numerically:

$$\hat{g} \approx \frac{1}{\sigma} \mathbb{E}_\epsilon[L(x+\sigma\epsilon)\,\epsilon]$$

This is what FOGZO and Evolution Strategies (ES) do. Unbiased in expectation, dimension-scaling cost. Works for *any* black-box module, including non-differentiable ones. Salimans et al.'s ES paper showed this can train neural networks competitively if you parallelize aggressively.

### 5. **Reinforcement learning framing**

Take this seriously: a calculator wired into the network is structurally identical to a tool-using policy. Treat the upstream layer as a *policy* whose action is "the calculator query," and the downstream task loss as the reward. PPO, actor-critic, or even DQN-style methods all apply. This is the principled version of #1 with engineering polish.

### 6. **Synthetic gradients / Decoupled Neural Interfaces (DNI)**

Train a small auxiliary network to *predict* what the gradient at the calculator boundary should be, given the upstream activations. The encoder gets fed this predicted gradient instead of a real one. Jaderberg et al. showed this allows asynchronous training and works through non-differentiable modules. Conceptually: you're learning the gradient itself.

Phase 7 update: a simple online MLP shadow-gradient module is now implemented
as a heldout warmup diagnostic. It uses per-example-scaled answer injection
gradients plus current result logits as state. Hidden size `64` produced useful
heldout alignment (`0.7167` result-proj, `0.7601` upstream) but overfit the
fit split (`0.2683/0.2202` train-heldout gaps); hidden size `16` reduced the
gap but missed the result-proj threshold (`0.6255`). Do not launch Stage 1
from this simple form. Next synthetic-gradient attempts should explicitly
address the target/state itself. Validation early stopping alone was tested
next with a separate heldout-test split; the selected checkpoint reached only
`0.6449` heldout-test result-proj cosine. Fit-split per-result z-score target
normalization improved heldout alignment but still missed the full gate: best
near miss was `h16` with `0.7259` result-proj cosine, `0.7549` upstream
cosine, and a `0.1723` result train-heldout gap. Raw appended policy-state
features did not fix this; `h32` reached heldout `0.7037/0.7611` but gaps
widened to `0.2853/0.2131`. Fit-split per-feature z-score standardization
also failed: the best simpler-state `h32` run reached only `0.6691/0.7028`
heldout with large gaps, while policy-state standardization was worse.
Directional shadow losses then improved the heldout direction signal but still
missed the full gate: `cosine` h16/h32 reached `0.7646/0.8007` and
`0.7937/0.8270`, but result train-heldout gaps stayed around `0.20`.
Gap-penalized validation selection exposed the tradeoff but did not solve it:
penalty `4` kept heldout above threshold but gap stayed `0.1673`, while
penalty `5` lowered gap to `0.1511` and lost heldout cosine. Training-time
dropout regularization also failed to close the gap: h32/dropout `0.1`
reached heldout `0.7920/0.8248`, but gaps stayed `0.2039/0.1564`; h16/dropout
`0.2` reached `0.7642/0.7983`, but gaps stayed `0.1977/0.1530`. A more
stable target construction, explicit gap/norm training loss, or a different
learned-gradient state remain the next plausible
branches. A lightweight target-stabilization attempt, per-example unit-norm
targets before z-scoring, also reproduced the same overfit profile: h32/cosine
reached `0.7936/0.8270` heldout with `0.2025/0.1545` gaps. So the next target
branch needed to be more structural than row-wise norm removal. Fit-split
result-prototype targets were then tested; h32/cosine improved heldout result
cosine to `0.8040`, but gaps were still `0.1909/0.1557`, and gap-penalized
h16 selection still had result gap `0.1705`. Prototype averaging by result
class is therefore not enough; the remaining plausible branches are different
state conditioning or explicit train-time gap/norm objectives. Appending the
raw calculator result-projection input as state was tested next; it improved
upstream heldout cosine to `0.8372` for h16/cosine, but result gap stayed
`0.1958`, so raw boundary activations alone are also not enough. A train-time
validation prediction-loss term was then added to the warmup objective; h32
kept high heldout cosine but result gaps stayed near `0.199`, while h16/weight
`1.0` reduced gaps but lost heldout/norm quality. Future synthetic-gradient
work should therefore use a more direct split-gradient gap/norm objective,
Jacobian-conditioned state, or a richer target construction rather than more
ordinary prediction-loss regularization.

### 7. **Target propagation (and Difference Target Propagation)**

Instead of propagating *gradients* backward, propagate *target activations*. The layer above the calculator computes "the activation I would have wanted from the calculator's output," and the upstream encoder is trained to map onto an inverse of that target. No gradients need to flow through the calculator at all. Bengio and collaborators developed this as a biologically-plausible alternative to backprop, but it's natural fit for non-differentiable modules.

### 8. **Forward-Forward (Hinton, 2022)**

Eliminate backprop entirely. Each layer is trained with a *local* "goodness" objective — high activity for positive (real) data, low for negative (corrupted) data. Since there's no global backward pass, the calculator's non-differentiability is irrelevant. The downside: Forward-Forward is still under-developed for large models, and you'd be doing two research projects at once.

### 9. **Direct Feedback Alignment (DFA)**

Replace the transposed-weight feedback path of backprop with **fixed random matrices**. Surprisingly, networks still learn. For your case, you could route a random feedback projection from the loss directly to the upstream neurons, bypassing the calculator entirely. Cheap, parallelizable, and indifferent to what's in between.

### 10. **Equilibrium Propagation / Predictive Coding**

Energy-based learning rules where the network settles into an equilibrium and weights are updated based on activity differences between two phases. These don't rely on differentiable forward operators in the standard sense. They're more exotic and currently confined to small-scale experiments, but they're a real alternative.

### 11. **Two-phase / curriculum training**

Pragmatic engineering trick: train initially with a *fully differentiable* approximation of the calculator (e.g., a learned MLP). Once the encoder has learned a sensible interface, freeze the encoder and swap in the real calculator. Then fine-tune downstream layers only. This sidesteps the gradient problem entirely by not having it during the hard learning phase.

### 12. **Implicit differentiation**

If your "calculator" is actually defined by an optimization problem or fixed-point equation (Z3 solver, root-finder, LP), you can differentiate through the *solution* using the implicit function theorem — no need to differentiate the solver's internals. This is what OptNet, DEQ, and JAX's `implicit_diff` do. Doesn't help for true black-box programs, but it's the right tool if your "calculator" is actually a constrained optimizer.
