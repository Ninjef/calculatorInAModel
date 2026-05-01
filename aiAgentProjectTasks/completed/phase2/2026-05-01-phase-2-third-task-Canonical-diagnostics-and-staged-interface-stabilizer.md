# Canonical Diagnostics and Staged Interface Stabilizer

## Mission

Resolve the ambiguity created by reusing phase-1 "Track" diagnostic scripts in phase 2, then run one staged adaptive-interface stabilizer under the strict `answer_decoder` bottleneck.

The previous phase-2 task showed:

- the downstream counterfactual target signal still identifies the true result class;
- frozen-upstream training improves answer/calculator-result accuracy relative to the failed adaptive baseline;
- lower LR and entropy reduce collapse severity but do not produce stable learned actions;
- soft-result target alone collapses badly;
- learned actions never beat true actions and never become Track 4 best.

This task should make the diagnostic contract explicit before adding more adaptive variants. Then it should test whether a small, decaying true-operand stabilizer can bootstrap the interface without being counted as final success evidence.

## Part 1: Canonicalize Diagnostic Usage

Create phase-neutral diagnostic entrypoints or documentation so future phase-2 tasks do not need to guess what "Track 3" and "Track 4" mean.

Required outcome:

- Keep backward-compatible wrappers for existing `scripts/run_phase1_*` or `scripts/run_track*` paths.
- Add or document canonical phase-neutral names for:
  - causal calculator protocol diagnostics;
  - action-loss diagnostics.
- The canonical docs must state what each diagnostic is for:
  - causal dependence and bottleneck classification;
  - forced-result sweep;
  - learned-vs-true-vs-random/shuffled action NLL comparison.
- The docs must state what these diagnostics are not:
  - not unit tests;
  - not phase-1-only task files;
  - not proof of success unless paired with calculator-dependent accuracy and learned-action improvement.
- Update phase-2 task/report language to prefer the canonical diagnostic names while still mentioning legacy Track 3/Track 4 labels where useful.

Do not refactor the diagnostic internals unless needed for clean entrypoints or docs.

## Part 2: Staged True-Operand Stabilizer Experiment

Run a strict-bottleneck adaptive-interface experiment with a small decaying true-operand auxiliary loss used only as a stabilizer.

Starting checkpoint:

```text
runs/2026-04-30_175805_513968_model-c-oracle-op0-19-answer_decoder/model-c-2digit-seed2/final_weights.pt
```

Keep this regime:

```text
digits=2
operand_max=19
calculator_operand_vocab_size=20
n_layer=2
n_head=1
n_embd=16
mlp_expansion=1
calculator_hook_after_layer=1
calculator_read_position=operands
calculator_bottleneck_mode=answer_decoder
calculator_estimator=adaptive_interface
freeze_semantic_decoder=true
```

Use the most informative previous stabilizer as the base:

```text
input_proj_lr=3e-4
upstream_lr=3e-4
adaptive_interface_target_mode=hard_pair
```

Run a small ladder:

```text
aux_operand_loss_weight in {0.01, 0.03}
aux_operand_loss_decay_steps in {250, 500}
aux_operand_loss_floor=0.0
```

Purpose:

- Determine whether a short supervised nudge can put `input_proj` into the basin where the counterfactual adaptive objective maintains or improves true calculator actions.
- Do not claim success from auxiliary supervision alone. A positive result must preserve or improve learned actions after the aux weight has decayed away.

## Validation Gates

For every candidate checkpoint, report:

- answer exact match;
- operand exact match;
- calculator result accuracy;
- adaptive target result accuracy;
- learned-target agreement;
- final adaptive interface CE;
- final aux operand loss and aux weight;
- `input_proj` parameter delta;
- frozen semantic decoder deltas;
- injection-zero, forced-zero, forced-random, and oracle-at-eval counterfactuals;
- canonical causal diagnostic classification and bottleneck label;
- canonical action-loss gaps.

A checkpoint is interesting only if it improves over:

```text
runs/2026-05-01_070903_085855_model-c-op0-19-adaptive_interface-answer_decoder/model-c-2digit-seed2
```

and over the lower-LR stabilizer:

```text
runs/2026-05-01_080805_707417_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed2
```

on at least one of:

- operand exact match;
- calculator result accuracy;
- learned-target agreement;
- action-loss learned-minus-true gap;
- learned-best fraction.

## Required Reporting

Write a phase-2 work history with:

- exact commands and run paths;
- code/doc changes for canonical diagnostics;
- the semantic decoder checkpoint path;
- all run configs;
- whether aux loss had decayed to zero at evaluation;
- metric table comparing all staged stabilizer runs to the failed baseline and lower-LR run;
- counterfactual table;
- causal diagnostic summary;
- action-loss summary;
- go/no-go recommendation.

Also update:

```text
factSheets/PHASE_2_EXPERIMENT_FACT_SHEET.md
```

## Success Criteria

A positive result must show calculator-dependent answer accuracy and improved learned calculator actions under the strict bottleneck after the true-operand auxiliary stabilizer has decayed away.

A useful negative result is acceptable if it clarifies whether direct operand bootstrapping helps the adaptive objective maintain a better interface, or whether the interface still collapses once direct supervision is removed.
