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
work therefore tried a direct split-gradient objective next. That objective
finally cleared Stage 0B: h32 with validation-gradient weight `0.5` and norm
weight `0.1` reached heldout `0.8068/0.8083`, gaps `0.1227/0.1343`, and norms
`1.1276/1.0736`. But using the calibrated module as a fixed Stage 1 feedback
source failed at weights `1.0/0.01/0.001`. The useful remaining idea is not
ordinary prediction regularization; it is on-policy shadow refresh,
trust-region feedback that checks refreshed gradient agreement,
Jacobian-conditioned state, or a richer target that survives model movement.
A simple fixed-module feedback L2 clamp was tested next: clamps `3.5` and
`10` stopped the feedback norm explosion but did not improve Stage 1 accuracy,
so plain output-norm clamping is not enough. Periodic on-policy refresh every
`50` steps then restored excellent current-model gradient agreement, but the
model still collapsed to a single result and ended at `0.025` exact match.
Soft result-policy entropy/batch-diversity stabilization was tested next.
Low diversity weight did not stop hard collapse, while high diversity weight
with a feedback clamp kept roughly `9` effective hard results at step `200`
but reached only `0.070` final exact match. The remaining synthetic-gradient
problem is therefore not just avoiding one-result collapse; the constraint has
to connect diverse requests to per-example improvement. A direct optimizer-step
trust region was also tested: capping realized AdamW parameter deltas at
`0.05`/`0.10` stabilized norms but still missed the early-lift baseline. A
hard-path answer-loss acceptance gate was then tested; it accepted only `3%`
of proposed steps and still ended at `0.050` exact. A hard-answer-loss line
search over proposed step scales `1,0.5,0.25,0.1,0` accepted only `2.5%` of
steps; it lifted best snapshot slightly to `0.0925` but finished at only
`0.060` exact. A first Jacobian-conditioned state then appended local
result-output `J^T answer_grad` scores. With feature z-scoring, h32 cleared
Stage 0B (`0.9073/0.9011` heldout result/upstream), but refreshed Stage 1
still ended at only `0.055` exact. A hard improvement-assignment target then
finally produced lift: weight `10` reached `0.170` final exact with refreshed
shadow and `0.400` without shadow by step `200`. This is promising but not
yet the final scalable method because it scores forced result classes during
training. A plain target-off handoff failed: decaying assignment weight
`10 -> 0` over `200` steps with natural answer loss on peaked at `0.370`, then
collapsed to about `0.105` after shutoff. Next work should test longer
always-on convergence, seed replication, stronger handoff bridges, and cheaper
assignment approximations rather than more soft diversity or state-only shadow
features.
Longer always-on assignment established that this is a real but unfinished
training path: no-shadow weight `10` reached `0.915` final exact on the
original 1600-step seed, while two replication seeds ended at `0.860` and
`0.820` final exact and one peaked at `0.920` before drifting down. This is
strong evidence that forced-result improvement assignment can teach the
natural result interface, but it is still prescriptive and expensive because
it scores forced result classes every step. The next useful ideas are cheaper
assignment construction, checkpoint/stability selection, a target-off bridge
stronger than plain decay, and testing whether the same signal works when the
calculator path is not the only available path.
That direct non-bottleneck transfer test is now negative. In additive
`calculator_bottleneck_mode=none`, answer loss can improve through the normal
residual path while the learned result request stays near chance. Hard
assignment weight `10` did not fix that: final exact was `0.700`, but final
calculator-result accuracy was only `0.0325`, result-policy accuracy was
`0.0275`, and assignment targets were almost never the true sum by step `800`.
Future non-bottleneck ideas need an explicit causal-use pressure, staged
bottleneck-to-additive handoff, or a way to compute improvement targets that
does not get corrupted by the bypass path.
A first causal-use pressure was tested as a hinge on
`zero_injection_loss - normal_loss`. It is cheap and non-prescriptive, but by
itself it was not enough: weights `10/50` with margin `0.5` created a large
causal gap while calculator-result accuracy stayed near chance and answer
accuracy fell. This suggests the next non-bottleneck approach should not only
make ablations worse; it must connect the causal path to correct result-level
utility, for example by staged bottleneck-to-additive training or a better
assignment target.
A first staged bottleneck-to-additive handoff then showed that this route can
work if the learned calculator policy is protected during handoff. Loading a
strong bottleneck checkpoint into an additive model without freezing preserved
the result policy only at step `0` (`0.9125`) and destroyed it by step `50`
(`0.0300`), after which the model solved mostly through the neuron path.
Freezing the embeddings, pre-hook block, and result action head kept final
calculator-result accuracy at `0.9200` and reached `0.9475` normal snapshot
accuracy with injection-zero only `0.0175`. This is the first strong
non-bottleneck calculator-dependence result, but it is a staged and frozen
handoff, not from-scratch non-prescriptive discovery. Useful next work:
replicate seeds, try controlled unfreezing, and replace the prescriptive
bottleneck policy-creation phase with a cheaper or less supervised mechanism.
Replication refined that picture: the strong source checkpoint transferred to
another additive seed with final eval `0.9525`, but weaker source checkpoints
did not give a high-accuracy additive readout by 800 steps despite preserving
learned calculator-result accuracy. So the next handoff work should not just
repeat frozen transfer; it should select or improve source checkpoints, train
the downstream readout more robustly, or unfreeze carefully without destroying
the calculator policy.
Longer downstream adaptation helped the weaker sources: continuing `src5_add5`
for another 800 steps improved final eval to `0.8175`, and continuing
`src4_add2` improved to `0.6050`, with injection-zero still near chance. This
means weak-source handoff is partly an adaptation problem, but the strong
source still reached `~0.95` faster and higher. Useful next handoff ideas are
checkpoint selection, better readout adaptation, and controlled unfreezing
rather than simply extending every frozen run.
The first simple controlled-unfreeze attempt was negative. Continuing adapted
weak-source checkpoints with all policy parameters unfrozen at LR `3e-4`
collapsed learned calculator-result accuracy to `0.3000` and `0.2525`.
Unfreezing therefore needs retention constraints or selective parameter
movement; ordinary low-LR answer-loss continuation is not enough.
Adding an explicit result-policy KL anchor made unfreezing useful: with anchor
weight `10`, `src4_add2` improved to `0.7475` final eval and `src5_add5`
improved to `0.9525`, while learned calculator-result accuracy stayed around
`0.80`. This is still staged and anchored, but it is a concrete path for
adapting a non-bottleneck calculator without immediately destroying the policy.
A fast off-ramp did not work: decaying that anchor from `10` to `0` over
`200/400` unfreeze steps left the policies usable at shutoff, but final
calculator-result accuracy fell to `0.5950` and `0.3850`. Future anchor work
should use slower/floored/gated schedules or selective unfreezing, not assume
that the policy becomes self-sustaining once downstream readout improves.
The anchor can be much weaker while still useful. Constant KL anchors `1.0`
and `0.1` preserved final calculator-result accuracy around `0.77-0.81` and
kept injection-zero near chance; the best reduced-strength cells reached
`0.8325` and `0.9750` final eval. This suggests a retention regularizer may be
lightweight enough to scale, though it remains staged and active.
The first threshold probe suggests `0.01` is too weak for clean retention:
answer accuracy remained useful, but one cell's final calculator-result
accuracy fell to `0.6425` and anchor agreement to `0.7050`. A practical
retention schedule should probably floor around the `0.1` region or gate on
calculator-result accuracy rather than decaying all the way to zero.
A floored schedule now exists: `--result-policy-anchor-floor` lets the anchor
decay to a lightweight nonzero floor. Anchor `1.0 -> 0.1` preserved calculator
dependence and strong `src5` answer accuracy, but did not beat simply keeping
anchor `0.1` constant. The next interesting version is adaptive/gated
retention, not another fixed floor.
Freezing only the result-space action head is not enough: with `result_proj`
locked and only upstream trainable, the transferred policy still collapsed.
That points to the upstream representation feeding the head as the fragile
state, so retention has to constrain behavior or the whole policy path.
Behavior-gated anchoring is now available: the anchor can stay at a low base
weight and jump when anchor agreement falls below a threshold. A first gate
(`0.01`, agreement `<0.9 -> 0.1`) improved over constant `0.01`, but did not
beat constant `0.1`. The knob is useful, but the gate needs a smarter metric or
continuous schedule to matter.
Calculator-accuracy-gated anchoring was the first smarter metric test. It
behaved adaptively, and `src5_add5` reached `0.9825` final eval, but
thresholds `0.80/0.82` still left `src4_add2` below the fixed `0.1` anchor
result. Do not keep sweeping simple discrete thresholds as if that alone is
the recipe; the next retention idea should be continuous/adaptive, combine
calculator accuracy with answer utility, or change which policy-path
parameters are allowed to move.
Continuous anchor control is now implemented through
`--result-policy-anchor-gate-mode linear` and
`--result-policy-anchor-gate-band`. The first linear calculator-accuracy gate
(`0.01`, threshold `0.85`, band `0.10`, max `0.1`) improved `src4_add2`
slightly beyond fixed `0.1` while using mean anchor weight `0.0385`, but
`src5_add5` slipped slightly below the best fixed/discrete gate. Treat it as a
useful retention controller, not as the final scalable method. Further anchor
work should add answer-utility awareness or change what parameters move,
rather than just sweeping bands.
Selective policy-backbone freezing is also available through
`--freeze-calculator-policy-backbone`. It freezes embeddings and pre-hook
blocks while leaving the result action head trainable. On adapted weak
bottleneck-to-additive handoffs, it preserved learned calculator accuracy
without any anchor and improved answer accuracy over the frozen-adapted
baseline, but it still trailed lightweight anchored unfreezing. This supports
the diagnosis that upstream representation drift is the dangerous failure
mode; future selective-unfreeze ideas should combine stable policy backbones
with better readout/action-head adaptation or a small utility-aware anchor.
A tiny fixed result-policy anchor on top of policy-backbone freezing did not
help: anchor agreement already stayed near `1.0`, and final answer accuracy
slightly trailed no-anchor policy-backbone freezing. This says the missing
piece in that branch is not action-head retention. Look instead at downstream
readout adaptation under stable policy, answer-utility-aware retention, or
better source policies.
Longer stable-policy readout adaptation gave a split result: `src5_add5`
reached `0.9500` final eval after 1600 steps with learned calc still `0.8325`,
but `src4_add2` only reached `0.7550` despite learned calc `0.8550`. This
supports source-quality selection/acquisition as a next bottleneck: if the
source policy's representation is handoff-friendly, stable readout adaptation
can work without an anchor; if it is not, more time alone does not fix it.
Simply lowering the hard improvement-assignment source weight is not the fix:
on hostile seed `10`, weight `5` weakened source acquisition to `0.6750`
final eval and its best tested additive handoff reached only `0.3425` at
600 steps. Future source work should train or select for actual 500/600-step
handoff behavior, not just soften the existing assignment objective.
A first learned selector audit also failed to replace the handoff gate:
leave-family-out ridge over early handoff trace features reached only `5/8`
winner accuracy even at step `500`, below raw early exact at `6/8`. The next
practical source-acquisition tool should log the real additive handoff probe
during source training on cloned state, then select checkpoints by that score.

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
