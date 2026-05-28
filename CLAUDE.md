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

Next best step: improve shadow generalization by changing scaling or the loss,
not simply appending raw policy features. Plausible branches include feature
standardization, explicit regularization, a different synthetic-gradient loss,
or a more stable target construction. Keep the exact-grid boundary-ceiling
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
