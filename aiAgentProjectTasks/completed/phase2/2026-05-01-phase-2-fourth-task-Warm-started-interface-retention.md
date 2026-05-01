# Warm-Started Interface Retention Under Strict Bottleneck

## Mission

Test whether the adaptive interface can preserve a learned true-operand protocol after that protocol has first been made real by a supervised warm start.

The previous phase-2 task showed:

- canonical diagnostics are now phase-neutral and should be used in new reports;
- the strict `answer_decoder` semantic decoder remains healthy;
- oracle-at-eval and forced true-sum sweeps still show that correct calculator results solve the task downstream;
- true operand actions remain much better than learned/random/shuffled actions under the canonical action-loss diagnostic;
- small decaying true-operand auxiliary losses (`0.01`, `0.03`) did not bootstrap stable learned actions after decay;
- learned actions still never became action-loss best.

The next question is sharper:

```text
If the interface is warm-started into a nontrivial true-operand protocol, can the adaptive objective maintain that protocol after direct operand supervision is removed?
```

Do not count supervised warm-start performance as success. The success evidence must come from the adaptive-only retention phase after the true-operand supervision weight is zero.

## Part 1: Add or Use a Staged Training Path

Implement the smallest clean training mechanism needed to run a staged interface-retention experiment.

Preferred behavior:

- Stage A: supervised true-operand warm start.
- Stage B: decay true-operand supervision to zero while adaptive objective remains active.
- Stage C: adaptive-only continuation with supervision weight exactly `0.0`.

Keep this narrowly scoped. Do not refactor unrelated training code.

If the current `scripts/overfit_one_batch.py` options can already express the stages cleanly, use them. If not, add minimal CLI/config support such as:

```text
--aux-operand-loss-weight
--aux-operand-loss-decay-steps
--aux-operand-loss-floor 0.0
--freeze-upstream-encoder
--input-proj-only-warm-start or equivalent staged parameter freezing
--stage-boundaries or equivalent explicit staged schedule
```

The implementation must record enough per-step or per-stage metadata to verify:

- when supervision was nonzero;
- when supervision reached zero;
- which parameters were trainable in each stage;
- whether final evaluation and diagnostics occurred after supervision was zero.

## Part 2: Warm-Start Retention Experiment

Starting checkpoint:

```text
runs/2026-04-30_175805_513968_model-c-oracle-op0-19-answer_decoder/model-c-2digit-seed2/final_weights.pt
```

Keep the strict bottleneck regime:

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
adaptive_interface_target_mode=hard_pair
```

Primary run:

```text
Stage A:
  trainable: calculator_hook.input_proj only, unless existing code makes this impractical
  aux_operand_loss_weight=1.0
  adaptive_interface_loss_weight=0.0 or low enough not to prevent warm-start learning
  duration=300 to 500 steps

Stage B:
  trainable: calculator_hook.input_proj only
  aux_operand_loss_weight decays from Stage A value to 0.0
  adaptive_interface_loss_weight=1.0
  duration=500 to 1000 steps

Stage C:
  trainable: calculator_hook.input_proj only
  aux_operand_loss_weight=0.0
  adaptive_interface_loss_weight=1.0
  duration=500 to 1000 steps
```

Run at least one variant that unfreezes the upstream encoder only after the input projection has demonstrated retention, or explicitly report why that variant should wait.

Suggested small ladder if time permits:

```text
warm_start_aux_weight in {0.3, 1.0}
warm_start_steps in {300, 500}
adaptive_only_steps in {500, 1000}
freeze_upstream_encoder=true for the first pass
```

Prefer a focused ladder over many noisy variants.

## Required Checkpoints

Save and evaluate checkpoints at:

- warm-start handoff: end of Stage A;
- supervision-zero handoff: end of Stage B;
- final adaptive-only checkpoint: end of Stage C.

For each checkpoint, report:

- answer exact match;
- operand exact match;
- calculator result accuracy;
- adaptive target result accuracy;
- learned-target agreement;
- adaptive interface CE;
- aux operand loss and current aux weight;
- operand entropy/confidence;
- `input_proj` parameter delta from the semantic decoder checkpoint;
- frozen semantic decoder deltas;
- trainable parameter groups at that stage.

## Canonical Diagnostics

Use the canonical names in commands and reports:

```bash
PYTHONPATH=. python3 scripts/run_causal_calculator_protocol_diagnostics.py ...
PYTHONPATH=. python3 scripts/run_action_loss_diagnostic.py ...
```

It is fine for outputs to continue using legacy directory names such as `track3_diagnostics` and `track4_action_loss` for compatibility.

For the warm-start handoff, supervision-zero handoff, and final checkpoint, report:

- injection-zero, forced-zero, forced-random, and oracle-at-eval counterfactuals;
- canonical causal diagnostic classification and bottleneck label;
- forced-result sweep learned-best fraction;
- forced-result sweep true-sum-best fraction;
- canonical action-loss true, learned, random, and shuffled NLL;
- learned-minus-true, random-minus-true, and shuffled-minus-true gaps;
- true-best and learned-best fractions.

## Comparison Baselines

Compare against:

```text
runs/2026-05-01_070903_085855_model-c-op0-19-adaptive_interface-answer_decoder/model-c-2digit-seed2
runs/2026-05-01_080805_707417_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed2
runs/2026-05-01_085013_079390_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-aux0.03-auxdecay500/model-c-2digit-seed2
```

The last path is the best previous aux stabilizer by answer/calculator-result accuracy.

## Decision Criteria

A positive result must show all of:

- nontrivial true-operand protocol at warm-start handoff;
- final evaluation occurs after true-operand supervision is exactly `0.0`;
- final operand exact match and calculator result accuracy remain materially above previous adaptive-interface baselines;
- learned-target agreement improves over the lower-LR baseline;
- canonical action-loss learned-minus-true gap improves over the lower-LR baseline;
- learned-best fraction becomes nonzero or learned actions otherwise clearly move toward true actions;
- answer accuracy is calculator-dependent under counterfactuals and oracle-at-eval remains high.

A useful negative result is acceptable if it distinguishes between:

- failure to learn the warm-started true-operand protocol at all;
- successful warm start followed by collapse during decay;
- successful decay followed by collapse during adaptive-only continuation;
- retention by `input_proj` alone but collapse when upstream is unfrozen.

## Required Reporting

Write a phase-2 work history with:

- exact commands and run paths;
- code changes, if any;
- semantic decoder checkpoint path;
- staged run configs;
- stage boundaries and trainable parameter groups;
- whether and when supervision reached zero;
- metric table for warm-start handoff, supervision-zero handoff, and final checkpoint;
- comparison table against the failed baseline, lower-LR baseline, and best previous aux stabilizer;
- counterfactual table;
- canonical causal diagnostic summary;
- canonical action-loss summary;
- parameter delta table;
- go/no-go recommendation.

Also update:

```text
factSheets/PHASE_2_EXPERIMENT_FACT_SHEET.md
```

## Success Criteria

This task succeeds if it answers whether the adaptive objective can retain a true-operand calculator interface after supervised warm-start support has been removed.

Do not claim adaptive-interface success from Stage A alone. Stage A is only a setup condition. The final claim must be based on Stage C behavior under the strict `answer_decoder` bottleneck.
