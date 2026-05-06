# Full-Action Enumeration Teacher Before Upstream Unfreezing

## Mission

The low-variance replay/EMA action-loss continuation did not make self-training robust:

```text
Two of three selected-continuation runs were best at step 0.
Only one seed had a transient canonical learned-minus-true gap improvement.
Learned-best action-loss fraction stayed 0.0.
```

But prior diagnostics still show that answer-NLL contains useful action signal:

```text
Candidate pools often contain actions that beat the current learned action.
Oracle-at-eval remains high.
Injection-zero remains near zero.
```

The next question is whether sampled-candidate variance is the remaining obstacle. In this small phase-2 regime, the complete action space is only:

```text
calculator_operand_vocab_size = 20
full action pairs = 20 x 20 = 400
```

So stop sampling candidates. Enumerate all action pairs, score every pair by frozen answer-decoder NLL, and train the interface toward a full soft action distribution.

## Important Guardrail: Still Do Not Unfreeze Upstream Yet

This task is a teacher-quality test, not an upstream-training task.

Do not unfreeze upstream unless the full-enumeration teacher first gives robust evidence that answer-NLL-derived targets can improve learned calculator actions across seeds under the strict frozen-upstream regime.

## Starting Points

Stage B handoff:

```text
runs/2026-05-01_112523_133504_model-c-op0-19-adaptive_interface-inlr0.003-uplr0.003-answer_decoder-aux1-auxdecay500/model-c-2digit-seed2/final_weights.pt
```

Action-loss-selected dense checkpoints:

```text
runs/2026-05-03_112750_450950_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed3/checkpoint_snapshots/step_00100_weights.pt
runs/2026-05-03_114747_070474_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed4/checkpoint_snapshots/step_00550_weights.pt
runs/2026-05-03_114747_345486_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed5/checkpoint_snapshots/step_01050_weights.pt
```

Best prior action-loss self-training checkpoint:

```text
runs/2026-05-03_154959_116705_model-c-op0-19-action_loss_weighted_interface-inlr0.0003-uplr0.0003-alrand4-altop1-alloc1-alt1-answer_decoder/model-c-2digit-seed4/final_weights.pt
```

Latest negative low-variance replay/EMA work history:

```text
aiAgentWorkHistory/phase2/2026-05-05-low-variance-action-loss-continuations.md
```

## Fixed Regime

Keep the strict phase-2 bottleneck:

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
freeze_semantic_decoder=true
freeze_upstream_encoder=true
trainable=calculator_hook.input_proj only
answer_loss_weight=1.0
aux_operand_loss_weight=0.0
input_proj_anchor_weight=0.0
```

Primary claims must use checkpoints with:

```text
final_aux_operand_loss_weight exactly 0.0
final_input_proj_anchor_weight exactly 0.0
freeze_upstream_encoder=true
trainable_parameter_groups=[calculator_hook.input_proj]
```

Do not add true-operand auxiliary supervision.
Do not use true operands or true sums to construct action-loss training targets.
Do not unfreeze upstream in the primary experiments.

## Part 1: Implement Full-Enumeration Action-Loss Interface

Add a narrowly named estimator, for example:

```text
calculator_estimator=action_loss_full_enum_interface
```

Required behavior:

- For each prompt in a batch, enumerate every `(a,b)` pair in `[0, 19] x [0, 19]`.
- Force each pair through the frozen answer decoder and compute answer NLL.
- Convert the 400 losses into a soft target distribution.
- Train `calculator_hook.input_proj` toward the marginal A and B distributions implied by that full target.
- Use only answer NLL through the frozen decoder to rank/weight action pairs.
- Do not include true operands, true sums, or labels in target construction.

Recommended knobs:

```text
action_loss_full_enum_temperature
action_loss_full_enum_top_mass_floor or min_probability_floor if needed
action_loss_full_enum_chunk_size
```

Keep chunking explicit so the diagnostic can run on CPU/MPS without memory surprises.

## Part 2: Diagnostics Before Training

Add or extend a candidate diagnostic to report full-enumeration action landscape statistics for selected checkpoints:

- best full-enum action NLL;
- learned action NLL;
- true action NLL, for reporting only;
- learned-minus-true and learned-minus-best gaps;
- true-best and learned-best fractions;
- soft target true-A/true-B marginal mass, for reporting only;
- entropy/effective number of action pairs.

This diagnostic may report true operands, but training target construction may not use them.

## Part 3: Selected-Checkpoint Continuations

Run three 500-step full-enum continuations from the action-loss-selected dense checkpoints:

| Run | Start checkpoint | Steps | Input LR | Objective | Aux | Anchor |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| FullEnum-selected-cont-seed1 | dense seed1 step 100 | 500 | `0.0003` | full-enum action-loss | `0.0` | `0.0` |
| FullEnum-selected-cont-seed2 | dense seed2 step 550 | 500 | `0.0003` | full-enum action-loss | `0.0` | `0.0` |
| FullEnum-selected-cont-seed3 | dense seed3 step 1050 | 500 | `0.0003` | full-enum action-loss | `0.0` | `0.0` |

Use:

```text
--snapshot-every 50
--checkpoint-every 50
```

Select snapshots by canonical learned-minus-true action-loss gap, not oracle-at-eval or built-in normal exact.

## Part 4: Stage-B Comparison Only If Selected Starts Improve

Only if at least two selected-checkpoint continuations improve, run three Stage-B-started full-enum variants:

| Run | Start | Steps | Input LR | Objective | Aux | Anchor |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| FullEnum-stageB-seed1 | Stage B | 1000 | `0.0003` | full-enum action-loss | `0.0` | `0.0` |
| FullEnum-stageB-seed2 | Stage B | 1000 | `0.0003` | full-enum action-loss | `0.0` | `0.0` |
| FullEnum-stageB-seed3 | Stage B | 1000 | `0.0003` | full-enum action-loss | `0.0` | `0.0` |

## Part 5: Required Diagnostics

Use oracle-at-eval only as a wiring guardrail.

For each primary final and selected snapshot, report:

- built-in eval exact;
- learned-target agreement if available;
- full-enum action diagnostic summary;
- canonical action-loss true, learned, random, shuffled NLL;
- learned-minus-true, random-minus-true, shuffled-minus-true gaps;
- true-best and learned-best fractions;
- canonical causal classification and bottleneck label;
- injection-zero, forced-zero, forced-random, oracle-at-eval;
- forced-result sweep learned-best and true-sum-best fractions;
- private-protocol all-pair operand exact and calculator-result accuracy;
- group behavior for carry/no-carry and large/small operands for the best run.

## Decision Criteria

A useful positive result requires at least two of three selected-checkpoint continuations to improve over their starting checkpoints on:

```text
canonical learned-minus-true action-loss gap
operand exact or calculator-result accuracy
```

and preserve:

```text
injection-zero near zero
oracle-at-eval high
aux exactly 0.0
anchor exactly 0.0
upstream frozen
```

A stronger result requires either:

```text
learned-best action-loss fraction becomes meaningfully nonzero
```

or:

```text
private-protocol decoding shows clearer true-operand-like structure than the best current self-training checkpoint
```

If full enumeration succeeds, create a follow-up task for upstream distillation from the full-enum teacher. If it fails, treat this as evidence that answer-NLL-derived action targets are not enough under the current bottleneck/readout geometry, and pivot to better interface parameterization or curriculum rather than upstream unfreezing.

## Required Reporting

Write a phase-2 work history with:

- exact commands and run paths;
- code changes;
- proof that aux stayed exactly `0.0`;
- proof that anchor stayed exactly `0.0`;
- trainable parameter groups;
- full-enum selected-checkpoint continuation table;
- Stage-B-started comparison table if run;
- full-enum action landscape summary;
- canonical action-loss summary;
- causal diagnostic summary;
- private-protocol summary;
- recommendation on whether to proceed to upstream distillation.
