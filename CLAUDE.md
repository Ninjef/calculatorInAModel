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

Next best step: improve shadow generalization by changing the target
construction or learned-gradient update path so local gradient agreement
becomes useful training dynamics. Plausible branches include a step-level
trust region that validates per-step improvement rather than only bounding
parameter distance, hard/assignment-style usage constraints that tie diversity
to per-example improvement rather than soft marginal entropy,
Jacobian-conditioned state rather than raw activations, or a richer
target construction that remains valid after upstream movement.
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
