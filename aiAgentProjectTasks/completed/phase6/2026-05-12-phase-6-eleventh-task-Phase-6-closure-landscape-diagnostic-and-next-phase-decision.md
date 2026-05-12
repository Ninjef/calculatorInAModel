# Phase 6 Eleventh Task: Closure Landscape Diagnostic And Next Phase Decision

## Mission

Close Phase 6 with a small, decisive diagnostic instead of starting another
large training branch.

Phase 6 has already produced the central identifiable-setting positive:

```text
In the strict semantic_decoder_only sum_left_operand setup, deterministic
hard-forward / soft-backward Concrete answer-loss training can discover hard
calculator actions without true operand labels, hard-best CE, oracle operands
during training, or semantic decoder movement. The resulting hard protocol
retains after the relaxation is turned off.
```

The latest natural sum-only work removed the sum-only decoder/readout blocker,
but the same deterministic Concrete bridge did not learn a correct calculator
result protocol:

```text
oracle/readout gate: passed at 1.000
natural learned result accuracy: about 0.11
natural learned-result minus best-result gap: about 5.57
```

This task should answer:

```text
Was Phase 6's success mainly a success of identifiable action landscapes, and
does the natural sum-only failure come from underidentification/diffuse
operand-action gradients rather than a broken decoder or broken relaxation
implementation?
```

The output should be a Phase 6 closure decision and a recommended Phase 7
direction, not a new claim based on oracle success.

## Why This Is The Next Best Task

Helpful findings to preserve:

- Phase 4 made the calculator protocol identifiable with `sum_left_operand` and
  proved a true learned calculator-query protocol can be taught and retained.
- Phase 5 showed answer-only continuations can preserve or complete partial
  protocols, but plain no-handoff answer-only training does not discover the
  protocol.
- Phase 6 hard-best local targets proved that frozen answer-decoder NLL can
  construct an extremely sharp interface target in the identifiable task.
- Phase 6 strict random-upstream local-target training solved the interface
  with only the semantic decoder loaded, so the method did not rely on inheriting
  an oracle-trained upstream representation.
- Phase 6 deterministic Concrete is the most end-goal-relevant positive:
  answer loss itself trained the model-side calculator interface, with no direct
  operand labels, no hard-best local CE, and no oracle operands during bridge
  training.
- Deterministic Concrete replicated across effective seeds `2`, `4`, and `5`,
  retained after the relaxation was off, and tolerated modest upstream movement.
- The sum-only product decoder gate proved that the natural `0..19` decoder can
  now be healthy enough for interpretation.

Less helpful directions right now:

- More oracle-only training in any setting where the gate already passes.
- More constant-weight hard-best local-target teaching in `sum_left_operand`.
  It works and is now a control, not the next question.
- More simple linear local-target decay. The decay boundary task already showed
  the handoff does not cleanly solve with that schedule.
- More independent-head exact expected answer-loss sweeps. Expected cost fell,
  but hard actions collapsed to wrong protocols.
- More literal stochastic Gumbel runs before fixing the known NaN/variance
  failure mode.
- Scaling to `operand_max=99` before natural `0..19` has a plausible discovery
  mechanism.
- Repeating natural deterministic Concrete with small schedule variations before
  measuring whether the natural gradient is aiming at a useful result-level
  action group at all.

The fastest path to the overall project goal is:

```text
explain the identifiable-vs-natural gap -> close Phase 6 honestly -> start a
new phase only if the next step needs a different objective or action
parameterization.
```

## Read First

Read:

```text
CLAUDE.md
OVERARCHING_EXPERIMENT_PURPOSE.md
SOLUTION_IDEAS.md
docs/canonical_diagnostics.md
aiAgentProjectTasks/2026-05-10-phase-6-overarching_plan-Identifiable-local-interface-discovery.md
factSheets/PHASE_4_EXPERIMENT_FACT_SHEET.md
factSheets/PHASE_5_EXPERIMENT_FACT_SHEET.md
factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md
aiAgentWorkHistory/phase6/2026-05-12-relaxed-bridge-replication-stochastic-upstream.md
aiAgentWorkHistory/phase6/2026-05-12-sum-only-interaction-decoder-natural-bridge.md
```

Inspect as needed:

```text
src/model.py
scripts/overfit_one_batch.py
scripts/run_phase6_relaxed_bridge_replication_stochastic_upstream.py
scripts/run_phase6_sum_only_semantic_decoder_gate.py
scripts/run_phase6_natural_sum_only_relaxed_bridge.py
scripts/run_full_enum_action_loss_diagnostic.py
scripts/diagnose_calculator_protocol.py
scripts/diagnose_private_protocol.py
```

## Critical Guardrail

Do not rediscover oracle success.

Oracle-at-eval, forced true result classes, injection-zero controls, and
forced-random controls are wiring/readout checks only. Phase 6 is already past
that point in both the identifiable setting and the product-interaction natural
sum-only setting.

The closure decision must be based on learned-interface and action-landscape
evidence:

- learned pair/calculator-result accuracy in the identifiable setting;
- learned calculator-result accuracy in natural sum-only;
- full-enum best-pair and best-result-group metrics;
- pair-target entropy versus result-group entropy;
- one-step relaxed-gradient movement toward the true pair or best result group;
- learned-minus-best NLL gaps;
- semantic decoder movement exactly `0.0`;
- objective weights confirming no true operand CE, no hard-best local CE, no
  full-enum expected-loss objective, no anchor, and no oracle operands during
  bridge training.

## Stage 0: Evidence Table From Existing Runs

Do not rerun completed training.

Create a compact evidence table from existing summaries and fact sheets. At
minimum compare:

### Identifiable Deterministic Concrete Positive

Use:

```text
runs/2026-05-11_phase6_relaxed_bridge_replication_stochastic_upstream/summary.json
factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md
```

Required rows:

- deterministic Stage 1 best selected metrics for seeds `2`, `4`, and `5`;
- relaxation-off Stage 2 retained metrics for seeds `2`, `4`, and `5`;
- upstream-open stress selected and retained metrics;
- final objective weights;
- semantic decoder delta.

### Natural Product-Decoder Negative

Use:

```text
runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/summary.json
runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/diagnostic_summary.json
factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md
```

Required rows:

- Stage 0 product decoder gate metrics;
- Stage 1 natural deterministic Concrete best selected metrics;
- full-enum best-result group true sum;
- learned-result best fraction;
- learned-result minus best-result gap;
- injection-zero and forced-random controls;
- semantic decoder delta;
- final objective weights.

If any run summary is absent locally, use the fact sheet and work history as the
source of truth and state that no rerun was done.

## Stage 1: Paired Full-Enum Landscape Diagnostic

Run a cheap paired diagnostic on the same `0..19` action space for:

```text
A. identifiable sum_left_operand with the standard Stage 0B semantic decoder
B. natural sum-only with answer_decoder_interaction=product and the selected
   product decoder checkpoint
```

Prefer extending `scripts/run_full_enum_action_loss_diagnostic.py` or adding a
small dedicated Phase 6 closure runner. Do not build a broad new framework.

For each setting, report on the exhaustive all-400 prompt grid where practical:

- best action pair equals true operands;
- tie-aware true-pair best fraction;
- best calculator-result group equals true sum;
- true pair rank;
- mean pair target entropy;
- mean effective action pairs;
- mean effective result count;
- mean same-true-sum near-best pair count;
- true-pair probability under the soft action target;
- true-result-group probability under the soft action target;
- top-1/top-3/top-5 action mass;
- learned hard pair exact where meaningful;
- learned calculator-result accuracy.

Expected interpretation:

```text
If sum_left_operand has near-one-pair effective targets while natural sum-only
has many near-equivalent same-result pairs, the natural negative should be
treated as an underidentification/action-parameterization boundary, not as a
decoder health failure.
```

## Stage 2: Paired One-Step Relaxed-Gradient Diagnostic

Run one-step deterministic hard-forward / soft-backward Concrete gradient
checks in both settings from the appropriate strict `semantic_decoder_only`
initialization.

Use the same optimizer shape unless the code path requires a setting-specific
override:

```text
calculator_estimator=gumbel_concrete_interface
relaxed_calculator_mode=deterministic
relaxed_calculator_hard_forward=true
relaxed_calculator_temperature=2.0
freeze_semantic_decoder=true
freeze_upstream_encoder=true
answer_loss_weight=1.0
aux_operand_loss_weight=0.0
adaptive_interface_loss_weight=0.0
local_target_loss_weight=0.0
expected_answer_loss_weight=0.0
input_proj_anchor_weight=0.0
oracle_train=false
oracle_warmup_steps=0
```

Report before and after one optimizer step:

### Identifiable Metrics

- probability of the true/best pair;
- hard learned pair exact;
- hard learned calculator-result accuracy;
- gradient cosine versus diagnostic hard-best pair CE, if available;
- input-proj/upstream/semantic decoder deltas.

### Natural Sum-Only Metrics

- probability mass assigned to action pairs whose calculator result equals the
  true sum;
- probability mass assigned to the best result group;
- hard learned calculator-result accuracy;
- gradient cosine versus a diagnostic best-result-group objective, if practical;
- input-proj/upstream/semantic decoder deltas.

Run at least three initialization seeds if the one-step natural result is noisy.
This is still a diagnostic, not a training sweep.

Interpretation rules:

- If identifiable true-pair probability moves up but natural true-result-group
  probability does not, Phase 6 should close with a clear natural
  underidentification/gradient-routing blocker.
- If natural true-result-group probability moves up but full training still
  fails, Phase 7 should focus on optimization dynamics, annealing, entropy, or
  continuation schedules.
- If natural true-result-group probability moves strongly up and the existing
  training failure was caused by a clear bug, fix the bug and run only the
  minimal natural bridge retry needed to confirm.

## Stage 3: Closure Decision

Write a Phase 6 closure section in:

```text
factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md
```

It should include:

- the strongest supported Phase 6 claim;
- what Phase 6 did not support;
- why the natural sum-only bridge failed after the decoder gate was repaired;
- which branches are now controls rather than live directions;
- whether the next work item should be Phase 7;
- the recommended Phase 7 first task.

Use this decision template:

```text
Supported:
- answer-derived local targets can teach strict calculator protocols in the
  identifiable task;
- deterministic hard-forward / soft-backward Concrete answer-loss training can
  discover and retain the identifiable hard protocol without direct operand
  labels or oracle operands during bridge training;
- the positive replicates across seeds and tolerates modest upstream movement.

Not supported:
- natural sum-only result-level discovery with the current independent operand
  heads and deterministic Concrete schedule;
- literal stochastic Gumbel training under tested settings;
- direct expected answer-loss optimization over independent heads;
- scaling to larger natural arithmetic before resolving the natural
  underidentification/action-parameterization boundary.
```

## Recommended Phase 7 Direction

If the paired diagnostics confirm that natural sum-only failure is primarily
underidentification or diffuse result-level gradients, recommend a new phase
focused on one of these directions:

```text
1. Result-space interface parameterization:
   train a calculator-query policy whose primary learned action is the result
   class or a constrained/canonical query representation, then map to valid
   calculator calls.

2. Joint-pair or structured action head:
   replace independent operand heads when the correct result is represented by
   many diagonal action pairs that an independent product policy cannot use
   cleanly.

3. Communication-constrained natural task:
   keep natural addition, but block upstream from cheaply computing the sum
   itself so the calculator path is the only reliable route to the answer.

4. Target-propagation / local critic objective:
   learn a local boundary objective that is result-level rather than true-pair
   level, avoiding direct operand labels while still producing a low-variance
   signal.
```

Do not recommend starting Phase 7 with `operand_max=99` scaling unless Stage 1
and Stage 2 unexpectedly show the natural `0..19` gradient is already sharp and
well-routed.

## Deliverables

Produce:

```text
runs/2026-05-12_phase6_closure_landscape_diagnostic/summary.json
runs/2026-05-12_phase6_closure_landscape_diagnostic/summary.md
aiAgentWorkHistory/phase6/2026-05-12-phase-6-closure-landscape-diagnostic.md
```

Update:

```text
factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md
```

Then move this task file to:

```text
aiAgentProjectTasks/completed/phase6/
```

Commit and push.

## Stop Conditions

Stop after the closure decision. Do not launch a new Phase 7 training branch in
this task.

Only run a minimal natural bridge retry if the paired diagnostic reveals a
clear implementation bug in the previous natural Stage 1 run. If that happens,
document the bug, fix it, and keep the retry to one effective seed before
deciding whether Phase 6 needs a follow-up.
