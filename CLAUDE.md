# Overview

This repo is intended as a research sandbox. The thesis being researched is related to connecting non-differentiable tools into a neural network such that both the inputs and outputs to the tool are connected directly to the internal workings of the network. We are trying to see if we can get a simple calculator tool to sit within a neural network and whether the network can learn to use it.

# Progress

So far, we've found that if we use an "oracle" approach to give the calculator the correct output for whatever math question comes in, the downstream nodes can absolutely learn to answer the question (no surprise). However, because tools are typically non-differentiable, the upstream neurons have not yet shown an ability to learn how to provide inputs into the calculator such that the network's ability to do math succeeds. We have only heavily tried STE, but there are many other possible approaches available.

We have a lot of other ideas to try here SOLUTION_IDEAS.md

# Critical Research Guardrail

Do not rediscover or re-present oracle calculator success as progress. Since
Phase 1, the project has known that downstream answer components can solve the
task when given correct calculator outputs or oracle operands. Oracle runs,
oracle-at-eval recovery, injection-zero controls, and forced-random controls are
wiring checks only.

The only central research question is whether the upstream/model-side interface
can learn to provide useful calculator inputs, and whether that learned
calculator-query protocol is retained when direct operand supervision or other
teacher signals are removed. Future work should prioritize learned-interface
metrics: operand/pair exact match, calculator-result accuracy from learned
actions, private all-pair protocol decoding, learned-vs-true action-loss gaps,
aux/supervision weight exactly `0.0`, and retention across checkpoints/seeds.

Before running any oracle-only experiment, ask whether it is strictly needed to
validate new wiring. If the wiring has already been validated for the current
configuration, skip oracle-only reruns and move directly to learned-interface
teaching/retention.

Do not rediscover target-off retention as a novel research result. The project
has already tested the general pattern many times: teach or scaffold a
calculator interface, remove the scaffold, and ask whether answer loss retains
or completes the protocol. Phase 4 established seed-robust aux-zero retention
for the identifiable `sum_left_operand` true-operand protocol. Phase 5 showed
upstream-open answer-only continuations can preserve or complete partially
taught identifiable protocols, while no-handoff answer-only discovery still
fails. Phase 6 established relaxation/local-target-off retention for
answer-derived identifiable bridges, including deterministic Concrete across
seeds. Future target-off/retention runs should only be done when they test a
genuinely new interface, objective, action parameterization, or stability
question; do not spend tasks merely re-proving that retention-after-teaching is
possible.

# Current Phase 7 Finding

Phase 7 has a supervised natural `0..19` result-level ceiling, but not yet a
robust answer-loss discovery result.

Exact full-grid upstream-open result-boundary teaching can learn hard natural
result requests. Seed `2` produced a single-seed retained positive
(`0.9675` Stage 1 hard result accuracy; `0.8800` best post-start target-off
retention), and CLI seeds `4` and `5` relearned Stage 1 requests near exact
(`1.0000` and `0.9975`).

However, the strict retention replication gate failed. Seeds `4` and `5`
retained only `87.0%` and `88.2%` of their selected Stage 1 hard result
accuracy at the best post-start target-off checkpoints, below the required
`90%` threshold. This is `exact_grid_seed_replication_negative`, not a robust
retained-positive replication.

The next estimator-family test also produced a useful negative:
`multisample_result_space_policy_gradient_stage0_alignment_negative`.
Result-space REINFORCE is now implemented and wired: `K=16` exact-grid
multi-sample policy gradient produced nonzero result-proj/upstream gradients,
semantic decoder gradient stayed exactly `0.0`, and per-prompt/leave-one-out
baselines reduced advantage variance versus the old global EMA baseline. But
the fixed-grid policy-gradient estimate was anti-aligned with the known
boundary-target ceiling (`result-proj cosine=-0.0945`, upstream
`cosine=-0.1108`), so Stage 1 long training was intentionally skipped.

The exact result-marginal answer-loss gate has now resolved that ambiguity as
another negative: `result_space_expected_answer_loss_alignment_negative`.
Raw exact expected answer-loss over result classes produced nonzero result-proj
and upstream gradients with semantic decoder gradient exactly `0.0`, but both
were anti-aligned with the boundary-target ceiling (`result-proj
cosine=-0.0978`, upstream `cosine=-0.1231`). The sampled PG gradient was
strongly aligned with this raw exact expected-cost gradient (`result-proj
cosine=0.9577`, upstream `cosine=0.9736`), so the previous PG failure was not
mainly finite-sample variance. Detached z-score cost normalization weakly
improved the result-head cosine (`0.0764`) but still failed the upstream-open
gate (`-0.0007`). Stage 1 exact-marginal training was intentionally skipped.

The gradient-friendly decoder gate produced a mixed but ultimately negative
result for ordinary expected-cost discovery:
`gradient_friendly_decoder_stage0_pass_stage1_exact_marginal_discovery_negative`.
A contrastive-margin decoder made the Stage 0 exact expected-cost gradient
locally positive against the boundary ceiling (`result-proj cosine=0.1204`,
upstream `cosine=0.0484`) while forced true/oracle exact accuracy stayed
`1.0` and downstream semantic decoder gradient stayed exactly `0.0`.
However, Stage 1 exact-grid result-marginal training with that decoder frozen
still collapsed to a low-entropy wrong result policy (`0.0750` learned-best
hard result accuracy in the training curve; final exact-match `0.085`).
Local decoder-gradient sign improvement alone did not rescue answer-loss
discovery.

The first explicitly biased backward-channel task produced another useful
negative. Output-projection boundary feedback passed the fixed-grid Stage 0
alignment gate (`result-proj cosine=0.2772`, upstream `0.4382`, semantic
decoder gradient `0.0`), but Stage 1 discovery reached only `0.155` best
snapshot calculator-result accuracy and `0.160` final exact match. A
fixed-random direct-feedback seed failed the Stage 0 result-head gate
(`result-proj cosine=-0.0036`). Label:
`boundary_feedback_stage0_output_projection_alignment_pass_stage1_discovery_negative`.

A fit-once linear shadow-feedback module was then tested. It fits a linear map
from answer-loss injection gradients to boundary result-logit gradients once at
initialization, freezes that map, and trains without recomputing boundary
targets. Stage 0 induced gradients were almost perfectly aligned with the
boundary ceiling (`result-proj cosine=0.9983`, upstream `0.9854`, semantic
decoder gradient `0.0`), but the 200-step Stage 1 early-lift smoke failed
badly (`0.070` best snapshot calculator-result accuracy; `0.040` final exact
match). Label:
`linear_shadow_feedback_stage0_alignment_pass_stage1_early_lift_negative`.

A heldout split then showed why same-batch linear shadow alignment was too
weak as a gate. With a deterministic `320/80` exact-grid split, the fit split
still aligned almost perfectly (`result-proj cosine=0.9981`, upstream
`0.9845`), but heldout result-proj cosine fell to `0.2622` with a `0.7359`
train-heldout result-cosine gap. Label:
`heldout_linear_shadow_feedback_stage0_generalization_negative`.

The first online MLP shadow-feedback warmup used per-example-scaled answer
injection gradients plus current result logits as shadow inputs. Hidden size
`64` reached heldout result/upstream cosines `0.7167/0.7601`, but train-heldout
gaps were too large (`0.2683/0.2202`). Hidden size `16` reduced the result
gap, but heldout result cosine fell to `0.6255`. Label:
`online_mlp_shadow_feedback_stage0b_partial_alignment_no_clean_gate`.

Validation-selected early stopping was then added with a separate validation
split and untouched heldout test split. The selected `h64` checkpoint at step
`60` reached heldout-test result/upstream cosines `0.6449/0.7266`, with
train-test gaps `0.3201/0.2414`; the final unselected checkpoint was also
below the result-head threshold (`0.6955`). Label:
`online_mlp_shadow_feedback_validation_selection_negative`.

Fit-split per-result z-score target normalization was then added to the online
MLP shadow module. It improved raw heldout-test alignment but still did not
clear the full gate. The best near miss was hidden size `16`, with heldout-test
result/upstream cosines `0.7259/0.7549`, relative norms `1.4146/1.1848`, and
train-heldout gaps `0.1723/0.1458`; the result gap remained above `0.15`.
Label: `online_mlp_shadow_feedback_target_normalization_partial_no_go`.

Richer raw policy-state features were then appended to the target-normalized
online MLP input: result logits, probabilities, log-probabilities, and entropy
alongside the answer-gradient feature. This did not rescue the gate. Hidden
size `32` reached heldout-test result/upstream cosines `0.7037/0.7611`, but
train-heldout gaps widened to `0.2853/0.2131`; hidden size `16` missed the
result threshold (`0.6862`). Label:
`online_mlp_shadow_feedback_policy_state_raw_features_negative`.

Fit-split per-feature z-score normalization was then added for the online MLP
shadow input. It fits feature statistics only on the fit split and applies
them to train/validation/heldout features before the shadow MLP. It did not
rescue either feature state. With raw policy-state features, hidden sizes
`16/32` reached only `0.5942/0.3997` and `0.4340/0.4023` heldout
result/upstream cosines. With the simpler answer-gradient plus result-logit
state, `h32` reached `0.6691/0.7028` with gaps `0.2830/0.2658`; `h16` had a
small result gap but missed upstream badly (`0.6436/0.4763`). Label:
`online_mlp_shadow_feedback_feature_standardization_negative`.

Directional shadow losses were then added for the online MLP warmup:
`cosine` and `mse_plus_cosine` optimize normalized-target direction rather
than only componentwise MSE. This materially improved the simple
answer-gradient plus result-logit state, but still did not clear the full gate.
With target normalization and validation selection, `cosine` h16/h32 reached
heldout result/upstream cosines `0.7646/0.8007` and `0.7937/0.8270`, but
result train-heldout gaps stayed around `0.20`; h8 reduced capacity but missed
the heldout cosine threshold (`0.5990/0.5859`). Label:
`online_mlp_shadow_feedback_directional_loss_partial_no_go`.

Gap-penalized validation selection was then added to the directional-loss
online MLP gate. It subtracts a train-validation cosine-gap penalty from the
validation min-cosine checkpoint score while keeping the heldout split
untouched. It exposed a sharp tradeoff, but did not clear the gate. For
`cosine` h16, gap penalty `4.0` selected step `70` with heldout
`0.7165/0.7439`, but result gap was still `0.1673`; penalty `5.0` selected
step `60` and reduced gaps to `0.1511/0.1220`, but heldout fell to
`0.6872/0.6979`. Label:
`online_mlp_shadow_feedback_gap_penalized_selection_tradeoff_no_go`.

Training-time dropout regularization was then added to the online MLP shadow
module, with explicit `AdamW` weight decay exposed in the diagnostic config.
On the useful target-normalized `cosine` + `injection_grad_logits` branch,
dropout `0.1/0.2` at h16/h32 preserved the heldout direction signal but did
not close the generalization gap. The best heldout cosine was h32/dropout
`0.1` at `0.7920/0.8248`, with gaps `0.2039/0.1564`; h16/dropout `0.2`
reached `0.7642/0.7983`, with gaps `0.1977/0.1530`. Label:
`online_mlp_shadow_feedback_dropout_regularization_no_go`.

Per-example target-direction normalization was then added as a lightweight
target-stabilization test before fit-split target z-scoring. It made the
target rows unit-norm before the online MLP warmup, then evaluated induced
model gradients against the original boundary target. This did not change the
failure mode. On the same `cosine` branch, h32 reached heldout
`0.7936/0.8270`, with gaps `0.2025/0.1545`; h16 reached
`0.7650/0.8010`, with gaps `0.1983/0.1546`. Label:
`online_mlp_shadow_feedback_target_unit_norm_no_go`.

Fit-split result-prototype target stabilization was then added. It averages
boundary target gradients by the boundary-best result class on the fit split,
then trains/evaluates the online MLP against those class prototypes while the
heldout model-gradient gate still compares induced gradients against the
original boundary ceiling. This improved the tradeoff slightly, but still did
not clear the gate. h32/`cosine` reached the best heldout result cosine so far
in this branch (`0.8040/0.8243`) but had gaps `0.1909/0.1557`. h16/`cosine`
plus prototype gap-penalized selection reached heldout `0.7540/0.7855`, with
gaps `0.1705/0.1409`; result gap remained above `0.15`. Label:
`online_mlp_shadow_feedback_target_prototype_partial_no_go`.

The shadow state was then expanded with the actual calculator result-projection
input vector (`injection_grad_logits_result_input`). This was a qualitatively
different learned-gradient state from raw policy statistics: it gives the
shadow MLP the boundary representation that `result_proj` consumes. It
improved upstream heldout alignment, but still failed the result-head
generalization gate. h16/`cosine` reached heldout `0.7676/0.8372`, with gaps
`0.1958/0.1269`; h32/`cosine` reached `0.7895/0.8294`, with gaps
`0.2079/0.1533`. Gap-penalized selection on h16 kept step `100` for penalties
`3/4/5` and did not reduce the result gap. Label:
`online_mlp_shadow_feedback_result_input_state_negative`.

Train-time validation prediction-loss regularization was then added to the
online MLP shadow warmup. The module now can add a separate validation-split
shadow prediction loss to each fit update while preserving the untouched
heldout-test split for the final Stage 0B gate. This produced another useful
negative. h32 with validation-loss weight `0.5/1.0` preserved high heldout
cosines (`0.7953/0.8233` and `0.7915/0.8195`), but result gaps stayed near
`0.199`. h16 with weight `1.0` reduced gaps to `0.1595/0.1150`, but heldout
fell to `0.7274/0.7381` and relative norms rose to `1.3346/1.2494`. Label:
`online_mlp_shadow_feedback_validation_loss_regularization_no_go`.

Direct validation model-gradient regularization was then added to the online
MLP shadow warmup. This is the first clean Stage 0B pass in the online shadow
line: h32 with validation-gradient weight `0.5` and norm weight `0.1` reached
heldout result/upstream cosines `0.8068/0.8083`, train-heldout gaps
`0.1227/0.1343`, and relative norms `1.1276/1.0736`. h32 without the norm
term and h16 variants also cleared the cosine/gap gate, though h16 had
larger relative norms. However, a fixed calibrated online MLP module did not
produce Stage 1 early lift. Shadow weights `1.0/0.01/0.001` reached only
`0.075/0.005/0.035` final exact match, with best snapshots
`0.0525/0.0400/0.0550`, all below the `0.16` boundary-feedback baseline.
Label:
`online_mlp_shadow_feedback_validation_gradient_stage0b_pass_stage1_fixed_module_negative`.

A fixed-module feedback norm clamp was then added for Stage 1 apply. This
tested whether the validation-gradient module failed only because its feedback
norm exploded as the model moved. The clamp worked mechanically: with
`--shadow-feedback-apply-max-norm 3.5` and `10`, applied feedback stayed at
the requested norm instead of growing to tens of thousands. But Stage 1 still
did not lift: both clamped runs ended at `0.075` final exact match with best
snapshot `0.0525`, unchanged from the unclamped weight-`1.0` run. Label:
`online_mlp_shadow_feedback_apply_norm_clamp_stage1_negative`.

Periodic on-policy online-shadow refresh was then added for Stage 1. With
`--shadow-feedback-refresh-every 50`, the h32 validation-gradient module was
refit against the current model at steps `50/100/150/200`. Refresh worked as
a gradient-alignment mechanism: heldout result/upstream cosines after refresh
were `0.9820/1.0000`, `0.9971/0.9999`, `0.9978/0.9991`, and `0.9716/0.9997`,
with tiny train-heldout gaps. But Stage 1 still did not lift; final exact
match was `0.025` and the best snapshot was only `0.0475`. Label:
`online_mlp_shadow_feedback_on_policy_refresh_alignment_pass_stage1_negative`.

Soft result-policy stabilization was then tested as the first direct
training-dynamics constraint. The new result-space entropy/batch-diversity
bonus is non-prescriptive: it does not say which result is correct for any
example, only discourages collapse of the learned result policy. Low diversity
weight `1.0` with refresh still collapsed to one hard result, both without a
feedback clamp (`0.015` final exact, `0.0475` best snapshot) and with clamp
`10` (`0.005` final, `0.040` best). A high diversity ceiling, weight `100`
with clamp `10`, kept hard result usage broader (`9.14` effective hard
results at step `200`) and improved final exact to `0.070`, with best snapshot
`0.080`, but still remained far below the `0.16` output-projection feedback
baseline. Label:
`result_policy_soft_diversity_stabilization_stage1_negative`.

An actual optimizer-step trust region was then added. Unlike gradient
clipping, this snapshots trainable parameters, lets AdamW propose a step, and
scales the realized parameter delta back to a configured L2 radius. With
refreshed h32 validation-gradient shadow feedback plus feedback clamp `10`,
max-delta caps `0.05` and `0.10` both bound every update: proposed deltas were
roughly `0.17-0.20`, then scaled to the cap. The gate stabilized shadow norms
and preserved refresh agreement, but did not produce discovery. Cap `0.05`
ended at `0.075` final exact with best snapshot `0.060`; cap `0.10` ended at
`0.040` final exact with best snapshot `0.045`. Label:
`optimizer_step_trust_region_stage1_negative`.

An answer-loss acceptance gate then tested the stronger version of a trust
region: let AdamW propose a shadow-feedback step, evaluate hard-path answer
loss on the current batch, and revert the step if answer loss worsens beyond
a tolerance. This is non-prescriptive because it uses the real answer loss as
an accept/reject signal, not a forced calculator result label. With refreshed
h32 validation-gradient shadow feedback and feedback clamp `10`, tolerances
`0.0` and `0.1` both accepted only `6/200` steps (`3%`). They stabilized the
run but did not lift: final exact match was `0.050`, best snapshot `0.070`,
still below the `0.16` boundary-feedback baseline. Label:
`answer_loss_step_acceptance_stage1_negative`.

A line-search repair gate then tried scaled versions of the proposed
optimizer step (`1,0.5,0.25,0.1,0`) and kept the scale with the best
hard-path answer loss. This also remained negative. With the same refreshed
h32 validation-gradient module and feedback clamp `10`, only `5/200` scaled
steps were accepted (`2.5%`). Best snapshot improved slightly to `0.0925` at
step `25`, but final exact was only `0.060`. Label:
`answer_loss_line_search_step_repair_stage1_negative`.

An output-Jacobian-conditioned shadow state was then added. The new feature
mode appends the local result-signal-to-injection `J^T` answer-loss scores to
the answer-gradient/logit state. This is a genuine new state, not another
schedule tweak. Raw h32 nearly cleared the Stage 0B gate
(`0.7957/0.8237` heldout result/upstream cosines); with fit-split feature
z-scoring it cleared strongly (`0.9073/0.9011`, gaps `0.0639/0.0736`).
However, the refreshed Stage 1 smoke with clamp `10` still failed: final
exact was `0.055`, best snapshot `0.065`, and final learned calculator
accuracy `0.0475`. Label:
`output_jacobian_shadow_feature_stage0b_pass_stage1_negative`.

The first hard assignment-style usage constraint then produced a partial
positive. The new improvement-assignment target scores forced result classes,
assigns only answer-loss-improving result targets, and caps per-result
assignments so the policy cannot satisfy the loss by single-class collapse.
With refreshed h32 shadow feedback and clamp `10`, assignment weight `1`
collapsed (`0.0475` final exact), while weight `10` crossed the old `0.16`
early-lift baseline (`0.170` final, `0.2425` best snapshot). The ablation
without shadow was much stronger: assignment weight `10` alone reached
`0.400` final exact by step `200`. Label:
`hard_improvement_assignment_stage1_lift_partial`.

A target-off retention gate then decayed assignment weight `10 -> 0` over
`200` steps while keeping natural answer loss on, and continued to step `400`.
The interface did not retain. Exact match peaked at `0.370` around step `175`,
was still `0.3475` at the shutoff step `200`, but collapsed to `0.105` by
step `250` and ended at `0.1075`. Label:
`hard_improvement_assignment_decay_retention_negative`.

Longer always-on hard improvement assignment then produced a mixed partial
positive. With no shadow feedback and assignment weight `10` kept on, CLI seed
`2` reached `0.860` final exact by `800` steps and `0.915` final exact by
`1600` steps, with a best snapshot of `0.9475` at step `1300`. CLI seed `4`
ended at `0.860` final exact and `0.870` last snapshot by `1600` steps; CLI
seed `5` ended lower at `0.820` final exact and `0.8325` last snapshot, but
peaked at `0.920` around step `1500` before drifting down. Oracle stayed
`1.000`, injection-zero stayed near chance, and operand exact remained low,
so the learned result-space calculator path is doing the work. Label:
`hard_improvement_assignment_convergence_seed_replication_mixed_partial`.
Do not claim scalable or non-prescriptive success from this: the target still
scores forced result classes every step, did not retain under plain decay, and
needs a cheaper assignment approximation plus a stronger handoff.

The first non-bottleneck transfer gate for hard improvement assignment was
negative. A small code change allowed `calculator_action_head=result_space`
with the ordinary `ste` estimator so the additive path
(`calculator_bottleneck_mode=none`) can be tested without the strict answer
decoder. On the natural exact grid, the answer-only additive baseline reached
`0.615` final exact and a best snapshot of `0.9725`, but injection-zero was
also high (`0.560` at the best snapshot) and calculator-result accuracy stayed
near chance. Adding assignment weight `10` reached `0.700` final exact and
`0.820` best snapshot, but learned calculator-result accuracy stayed near
chance (`0.0275` final, `0.0575` best training-curve result-policy accuracy)
and assignment target accuracy collapsed to `0.0033` by step `800`. Label:
`non_bottleneck_hard_assignment_transfer_negative`. The bottleneck assignment
signal does not transfer as-is when a neuron path can solve around the
calculator; future non-bottleneck work needs a causal calculator-use pressure
or staged handoff, not just the bottleneck assignment objective.

A first explicit non-bottleneck causal-use pressure was then tested and was
also negative. The new objective logs `calculator_causal_gap =
zero_injection_loss - normal_loss` and applies a hinge requiring the
zero-injection path to be worse by a margin. This is non-prescriptive and costs
one extra zero-injection forward, not a forced-result sweep. On top of additive
answer loss plus assignment weight `10`, causal-gap weights `10` and `50`
with margin `0.5` did create a large final causal gap (`1.27` and `0.84`),
but calculator-result accuracy stayed near chance (`0.000` and `0.0425`
final), best result-policy accuracy stayed at `0.030` and `0.045`, and final
exact fell to `0.560` and `0.4225` versus `0.700` without the gap objective.
Label: `non_bottleneck_causal_gap_pressure_negative`. The objective can
damage the bypass path without teaching correct calculator requests; next
non-bottleneck work needs a target or handoff that ties causal dependence to
true result-level calculator utility.

A staged bottleneck-to-additive handoff then produced the first strong
non-bottleneck calculator-dependence partial positive. A new
`compatible_model` checkpoint load scope copies only shape-compatible tensors
from a bottleneck run into an additive model, and `--freeze-calculator-policy`
freezes the embeddings, pre-hook block, and result action head while allowing
the additive output projection and downstream/readout layers to train. Without
freezing, the transferred bottleneck policy started at `0.9125`
calculator-result accuracy but collapsed to `0.0300` by step `50`; final
normal was `0.8075`, injection-zero was `0.7675`, and learned calculator
accuracy was only `0.0250`. With the policy frozen, the additive model reached
`0.940` final eval exact and `0.9475` best snapshot exact by step `800`, while
injection-zero stayed `0.0175`, forced-random stayed `0.0500`, oracle reached
`0.9600`, and learned calculator-result accuracy stayed `0.9200`. Label:
`bottleneck_to_additive_freeze_policy_handoff_partial_positive`. This is real
non-bottleneck calculator-path use, but not yet the final goal: it is staged,
inherits a bottleneck-trained policy, and freezes that policy during handoff.
Next work should replicate seeds, test unfreezing schedules, and search for a
more scalable/non-prescriptive way to create or preserve the policy.

Frozen handoff replication then produced a useful mixed result:
`bottleneck_to_additive_freeze_policy_source_quality_mixed`. The strong
source checkpoint from the seed-2 1600-step bottleneck run replicated across a
new additive seed: `src2_add2` reached `0.9400` final eval and `0.9475` best
normal, while `src2_add4` reached `0.9525` final eval and `0.9325` best
normal; both kept injection-zero near chance (`0.0175/0.0200`) and learned
calculator-result accuracy high (`0.9200/0.9150`). But weaker source
checkpoints did not yield high downstream accuracy even though their frozen
action policies stayed high: `src4_add2/src4_add4` ended at only
`0.3025/0.3375` final eval with learned calc `0.8725/0.8575`, and
`src5_add5` ended at `0.5550` with learned calc `0.8000`. The handoff is
therefore robust for a good source policy/representation, but source
checkpoint quality and/or result-embedding geometry still matter. Next work
should test source checkpoint selection/quality metrics, stronger downstream
readout adaptation, and controlled unfreezing rather than blindly repeating
frozen transfers.

Longer downstream adaptation then showed that the weak-source handoffs were
not completely dead, but also not solved:
`bottleneck_to_additive_longer_downstream_adaptation_partial`. Continuing the
weak frozen additive runs for another 800 steps from their additive final
weights preserved the frozen calculator policies and kept causal controls near
chance. `src5_add5` improved from `0.5550` to `0.8175` final eval, with
injection-zero `0.0000`, forced-random `0.0425`, oracle `0.8075`, and learned
calc `0.8000`. `src4_add2` improved from `0.3025` to `0.6050` final eval,
with injection-zero `0.0025`, forced-random `0.0625`, oracle `0.5725`, and
learned calc `0.8725`. The result weakens the claim that poor 800-step
handoffs are hard failures, but source/readout quality still matters because
neither weak-source continuation matched the strong-source `~0.95` result by
the same total 1600-step adaptation budget.

A first controlled unfreeze probe was negative:
`bottleneck_to_additive_low_lr_unfreeze_policy_collapse_negative`. Starting
from the adapted weak-source checkpoints, removing `--freeze-calculator-policy`
and continuing with low global LR `3e-4` for 400 steps damaged the learned
calculator policy in both cells. `src4_add2` learned calc fell from `0.8725`
to `0.3000`, final eval fell from `0.6050` to `0.5200`, and forced-random
rose to `0.1200`. `src5_add5` learned calc fell from `0.8000` to `0.2525`,
final eval stayed roughly flat (`0.8175 -> 0.8100`), and forced-random rose to
`0.1125`. Normal accuracy can partly survive through downstream/bypass-like
adaptation, but low-LR unfreezing is not a safe policy-preserving handoff.
Next unfreezing work needs explicit policy retention regularization, a much
more selective unfreeze, or a mechanism that validates calculator-result
accuracy while adapting.

Explicit result-policy anchoring then produced a useful controlled-unfreeze
partial positive:
`bottleneck_to_additive_policy_anchor_unfreeze_partial`. A new
`--result-policy-anchor-weight` objective snapshots the initial fixed-grid
result-space policy and penalizes KL/MSE drift during training. With LR
`3e-4`, all policy parameters unfrozen, and KL anchor weight `10`, the adapted
weak-source handoffs avoided the collapse seen in plain unfreeze.
`src4_add2` improved from `0.6050` frozen-adapted final eval to `0.7475`,
with learned calc `0.8075`, anchor agreement `0.9800`, injection-zero
`0.0100`, and oracle `0.7875`. `src5_add5` improved from `0.8175` to
`0.9525`, with learned calc `0.7950`, anchor agreement `0.9850`,
injection-zero `0.0000`, and oracle `0.9375`. This is not a scalable
from-scratch solution because it anchors a staged policy, but it shows that
controlled unfreezing can improve non-bottleneck handoff if policy retention
is explicitly protected.

The first anchor off-ramp was negative:
`bottleneck_to_additive_anchor_decay_offramp_negative`. Keeping the same
adapted weak-source checkpoints, LR `3e-4`, and full policy unfreeze, but
linearly decaying the KL anchor from `10` to `0` over the first `200` of `400`
steps caused post-shutoff drift. At step `200`, the transferred policies were
still usable (`src4_add2` calc `0.8300`, `src5_add5` calc `0.8225`), but by
the final checkpoint they fell to `0.5950` and `0.3850`. Final eval was
`0.5925` for `src4_add2` and `0.6750` for `src5_add5`, worse than the
constant-anchor partial positive and, for `src5_add5`, worse than the frozen
adapted baseline. The anchor can protect unfreezing while present; this
decay schedule does not yet create a self-sustaining non-bottleneck policy.

A reduced-strength constant-anchor sweep was positive:
`bottleneck_to_additive_reduced_anchor_strength_partial`. With the same
adapted checkpoints, LR `3e-4`, and full policy unfreeze, constant KL anchors
of `1.0` and `0.1` both preserved useful calculator policies. Anchor `1.0`
ended at final eval `0.7775/0.9925` for `src4_add2/src5_add5`, with final
calculator-result accuracy `0.8050/0.7925`. Anchor `0.1` ended at
`0.8325/0.9750`, with final calculator-result accuracy `0.8075/0.7725`.
Injection-zero stayed near chance (`0.0000-0.0250`). This weakens the concern
that the non-bottleneck handoff requires a very large anchor, but it is still
not a from-scratch or anchor-free method.

Do not rerun these as next steps unless debugging new code:

- oracle/readout checks for natural `0..19`;
- random-resampled upstream-open boundary-target repeats;
- frozen linear or frozen MLP result-head boundary teaching;
- the MLP rescue from the full-grid task;
- more target-off retention reruns that do not introduce a genuinely new
  mechanism or diagnose the observed seed fragility.
- vanilla multi-sample result-space policy-gradient long runs without first
  fixing the Stage 0 gradient-alignment problem.
- raw exact expected-cost/result-marginal training, or learned-baseline
  variants that merely estimate the same raw expected-cost gradient.
- more decoder-only calibration branches that only aim to make forced true
  results sharper or weakly improve the same ordinary expected-cost geometry.
- plain output-projection boundary feedback with the same Stage 1 objective and
  schedule, or fixed-random DFA long runs that do not first pass result-head
  Stage 0 alignment.
- frozen fit-once linear shadow feedback with the same exact-grid calibration
  and weight/schedule.
- same-batch linear shadow alignment as a sufficient Stage 0 gate.
- simple online MLP shadow warmups with only injection-gradient plus result-logit
  state at `h64` or `h16`, `lr=1e-3`, `100` steps, unless adding a real
  anti-overfit or target-normalization change.
- validation-selected early stopping alone on that same simple online MLP
  shadow target/state as a Stage 1 go signal.
- the same per-result z-score target-normalized `h64/h32/h16/h8`, `lr=1e-3`,
  `100`-step validation-selected Stage 0B sweep as novelty.
- raw appended `injection_grad_policy_state` features with per-result target
  z-score, `h16/h32`, `lr=1e-3`, `100` steps as novelty.
- fit-split per-feature z-score standardization on either
  `injection_grad_logits` or `injection_grad_policy_state` with the same
  target-normalized `h16/h32`, `lr=1e-3`, `100`-step Stage 0B gate as novelty.
- plain `cosine` or `mse_plus_cosine` online MLP shadow losses on
  `injection_grad_logits` with per-result target normalization, h8/h16/h32,
  `lr=1e-3`, `100` steps as novelty.
- gap-penalized validation selection on that same directional-loss
  `injection_grad_logits` setup with penalties `1/3/4/5` as novelty.
- simple dropout-only regularization on that same directional-loss
  `injection_grad_logits` setup with dropout `0.1/0.2`, h16/h32, and
  `weight_decay=0.01` as novelty.
- per-example unit-norm target transforms on that same directional-loss
  `injection_grad_logits` setup with h16/h32 and `cosine`/`mse_plus_cosine`
  as novelty.
- fit-split result-prototype target averaging on that same directional-loss
  `injection_grad_logits` setup with h16/h32, `cosine`/`mse_plus_cosine`, and
  gap-selection penalties `3/4/5` as novelty.
- appending the raw calculator result-projection input to the same
  target-normalized directional-loss state (`injection_grad_logits_result_input`)
  with h16/h32, `cosine`/`mse_plus_cosine`, and h16 gap-selection penalties
  `3/4/5` as novelty.
- train-time validation prediction-loss regularization on the same
  directional-loss `injection_grad_logits` setup with validation-loss weights
  `0.5/1.0`, h16/h32, `lr=1e-3`, `100` steps, and ordinary heldout
  min-cosine selection as novelty.
- direct validation model-gradient regularization on the same
  `injection_grad_logits`, target-normalized h16/h32 setup with
  validation-gradient weight `0.5`, norm weights `0/0.1`, and fixed-module
  Stage 1 weights `1.0/0.01/0.001` as novelty.
- fixed calibrated online-MLP shadow feedback with simple apply feedback L2
  clamps `3.5` or `10` on the same h32 validation-gradient module as novelty.
- periodic on-policy refresh every `50` steps with the same h32
  validation-gradient module, `shadow_feedback_weight=1.0`, no apply clamp,
  and 200-step early-lift budget as novelty.
- soft result-policy entropy/batch-marginal diversity stabilization on top of
  the same refreshed h32 validation-gradient module, including low diversity
  weight `1.0` with/without clamp `10` and high diversity weight `100` with
  clamp `10`, as novelty.
- actual optimizer-step L2 trust-region caps `0.05` or `0.10` on top of the
  same refreshed h32 validation-gradient module with feedback clamp `10` and
  200-step early-lift budget as novelty.
- hard-path answer-loss step acceptance on the same refreshed h32
  validation-gradient module with feedback clamp `10`, tolerances `0.0/0.1`,
  and 200-step early-lift budget as novelty.
- hard-path answer-loss line search over scales `1,0.5,0.25,0.1,0` on the same
  refreshed h32 validation-gradient module with feedback clamp `10` and
  200-step early-lift budget as novelty.
- output-Jacobian shadow feature mode
  `injection_grad_logits_output_jacobian` with h16/h32 raw features or h32
  fit-split feature z-scoring, validation-gradient `0.5`, norm `0.1`, refresh
  every `50`, clamp `10`, and 200-step early-lift budget as novelty.
- hard improvement-assignment weights `1` or `10` on the same exact-grid
  seed-2/seed-4 run for only 200 steps as novelty. Weight `10` is the first
  useful Stage 1 lift in this branch; next test retention, longer convergence,
  seeds, or lower-cost/scalable assignment construction.
- hard improvement-assignment weight `10` decayed linearly to zero over
  `200` steps with `answer_loss_weight=1`, no shadow feedback, and 400-step
  budget as novelty; this failed target-off retention.
- always-on hard improvement-assignment weight `10` for 800 or 1600 steps on
  the same exact-grid seeds as novelty. This branch now has a mixed seed
  replication result: useful convergence lift, but no retention/scaling claim.
- non-bottleneck additive result-space `ste` with answer loss plus hard
  improvement-assignment weight `10` for 800 steps on the same seed as novelty.
  It learned answer accuracy mostly through the neuron path while calculator
  result accuracy stayed near chance.
- non-bottleneck additive answer loss plus assignment weight `10` with
  calculator causal-gap hinge margin `0.5` and weights `10` or `50` for 800
  steps on the same seed. It created a loss gap but not correct calculator
  requests.
- compatible bottleneck-to-additive checkpoint loading without freezing the
  calculator policy on the same seed; it immediately destroyed the transferred
  result policy.
- frozen-policy bottleneck-to-additive handoff on the same seed/checkpoint as
  novelty. This is now a partial positive; next tests should vary seeds,
  checkpoints, unfreeze schedules, or the way the policy is acquired.
- the completed frozen-policy handoff replication matrix cells
  `src2_add2`, `src2_add4`, `src4_add2`, `src4_add4`, or `src5_add5` at
  800 steps as novelty. They established strong-source replication and weak
  source sensitivity.
- the completed weak-source frozen continuation cells `src4_add2` and
  `src5_add5` for one extra 800-step continuation as novelty. They established
  that longer downstream adaptation helps but does not fully erase source
  quality sensitivity.
- low-LR `3e-4` full-policy unfreeze for 400 steps from the adapted
  `src4_add2` or `src5_add5` checkpoints as novelty. It collapsed learned
  calculator-result accuracy.
- result-policy KL anchor weight `10`, LR `3e-4`, 400-step full unfreeze from
  the adapted `src4_add2` or `src5_add5` checkpoints as novelty. This is now
  a partial positive; next anchored-unfreeze work should vary anchor schedule,
  selective unfreezing, or source acquisition.
- result-policy KL anchor weight `10` decayed linearly to zero over `200`
  steps, LR `3e-4`, 400-step full unfreeze from the adapted `src4_add2` or
  `src5_add5` checkpoints as novelty. It lost calculator-result accuracy after
  anchor shutoff.
- result-policy KL anchor weights `1.0` or `0.1`, LR `3e-4`, 400-step full
  unfreeze from the adapted `src4_add2` or `src5_add5` checkpoints as novelty.
  Both preserved useful calculator-result accuracy with low injection-zero.

Next best step: improve shadow generalization by changing the target
construction or learned-gradient update path so local gradient agreement
becomes useful training dynamics. Plausible branches include a step-level
mechanism that constructs better directions rather than selecting from mostly
bad proposed shadow steps, longer always-on convergence and seed tests for
the hard improvement-assignment target only if the test changes stability or
selection, a target-off handoff with a stronger natural answer-loss bridge
than plain decay, a lower-cost assignment approximation that does not enumerate
all result classes every step, a non-bottleneck version of the hard-assignment
gate only if it adds explicit causal calculator-use pressure or a staged
bottleneck-to-additive handoff that is stronger than a plain zero-injection
loss-gap hinge, seed replication/unfreeze schedules for the frozen-policy
bottleneck-to-additive handoff that specifically changes source checkpoint
selection, downstream adaptation beyond just one longer continuation, or
unfreezing with an even weaker/floored/gated policy-retention schedule or
selective parameter set,
a Jacobian-conditioned state more substantial than the result-output
`J^T answer_grad` feature, or a richer target construction that remains valid
after upstream movement.
Keep the exact-grid boundary-ceiling
diagnostic as the Stage 0 gate for any new mechanism, require a heldout warmup
pass before Stage 1, and require early Stage 1 lift above the `0.16`
boundary-feedback baseline before long runs. Do not move directly to
canonical-query/protocol stabilization as if Phase 7 retention had robustly
replicated, and do not treat ordinary expected-cost/score-function training as
rescued by decoder calibration alone.

For details, see `factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md`.

# Navigation
You can find a valuable set of fact sheets in factSheets/, which keeps track of all the learnings of past experiments by experiment phase
Under aiAgentWorkHistory, we have all the work performed in the past.
Under aiAgentProjectTasks, we have all the intended work to be done by the researchers, completed ones in the completed folder.
You can find the overarching experiment's purpose here: OVERARCHING_EXPERIMENT_PURPOSE.md
You can find a heavy set of ideas we want to try here: SOLUTION_IDEAS.md

# After contributing
- Whenever doing experiments and learning new information, fill out information in the associated phase's fact sheet
- Fill out any work history in aiAgentWorkHisotry that you've accomplished
- Move your task file to the completed folder (if it's in fact fully completed)
- Commit and push
