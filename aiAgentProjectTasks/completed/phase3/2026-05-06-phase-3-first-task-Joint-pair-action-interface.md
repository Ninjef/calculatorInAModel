# Joint Pair-Action Interface Under Strict Bottleneck

## Mission

Phase 2 ended with a specific failure mode:

```text
Full-enum answer-NLL can score all 20 x 20 calculator action pairs.
But the training objective collapses those pair scores into independent A/B marginals.
Learned-best action-loss fraction stays near zero, and private-protocol accuracy
plateaus around a partial, B-noisy protocol.
```

This task tests the most direct Phase 3 hypothesis:

```text
Representing the calculator action jointly as a pair distribution will preserve
the action-loss teacher's pair structure and improve learned calculator actions
under the strict frozen-upstream bottleneck.
```

This is not an upstream-unfreezing task. It is a joint-interface test.

## Starting Points

Use the same Phase 2 strict decoder setup.

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

Best Phase 2 full-enum selected snapshot for comparison:

```text
/Users/jarnold/Documents/Codex/2026-05-05/please-work-in-this-repo-users/diagnostics/private_full_enum/selected_seed2_step200/private_protocol_summary.json
```

Key comparison numbers from Phase 2:

```text
selected seed2 step 200 all-pair answer exact: 0.5300
selected seed2 step 200 operand exact: 0.5675
selected seed2 step 200 calculator result accuracy: 0.5750
canonical learned-minus-true action gap: 2.0201
learned-best action-loss fraction: 0.0
```

## Fixed Regime

Keep the strict Phase 2 bottleneck unless this task explicitly says otherwise:

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
answer_loss_weight=1.0
aux_operand_loss_weight=0.0
input_proj_anchor_weight=0.0
```

Primary claims must use checkpoints with:

```text
final_aux_operand_loss_weight exactly 0.0
final_input_proj_anchor_weight exactly 0.0
freeze_upstream_encoder=true
trainable parameter groups limited to the calculator interface head
```

Do not use true operands or true sums to construct training targets.
Do not unfreeze upstream in this task.

## Part 1: Implement a Joint Pair Head

Add a narrow interface mode for a joint action distribution.

Suggested config knobs:

```text
calculator_action_head=independent_operands|joint_pair
calculator_estimator=action_loss_full_enum_joint_interface
action_loss_full_enum_temperature
action_loss_full_enum_min_probability_floor
action_loss_full_enum_chunk_size
```

Recommended first implementation:

- Keep the existing independent A/B path unchanged.
- In `joint_pair` mode, read the A residual at the final A digit and the B residual at the final B digit.
- Concatenate the two read vectors.
- Project the concatenated vector to `operand_vocab_size * operand_vocab_size` pair logits.
- Decode the argmax pair into `(a_pred, b_pred)`.
- Feed `a_pred + b_pred` through the existing hard calculator result path.
- Preserve existing trace fields where possible, and add pair-specific trace fields.

Useful new trace fields:

```text
pair_pred
pair_confidence
pair_entropy
pair_logp
pair_true_rank, diagnostic/reporting only
pair_true_probability, diagnostic/reporting only
```

Keep this implementation boring. Do not add low-rank/factorized pair heads until the direct joint head has a baseline.

## Part 2: Train on Full-Enum Pair Targets Directly

Add the first joint estimator:

```text
calculator_estimator=action_loss_full_enum_joint_interface
```

Required behavior:

- Enumerate all `20 x 20 = 400` action pairs.
- Force each action pair through the frozen answer decoder.
- Compute answer NLL for each pair.
- Convert pair losses to a soft pair target distribution.
- Train the joint pair logits against the full pair target distribution.
- Do not marginalize into independent A/B targets.
- Do not use true operands or true sums for target construction.

The training loss should be pair-level CE/KL:

```text
target_pair_distribution = softmax(-answer_nll / temperature)
loss = -sum(target_pair_distribution * log_softmax(pair_logits))
```

Continue to log the old full-enum metrics for comparison, but add joint metrics:

```text
action_loss_full_enum_joint_target_loss
action_loss_full_enum_pair_entropy
action_loss_full_enum_effective_pairs
action_loss_full_enum_learned_pair_nll
action_loss_full_enum_true_pair_nll, reporting only
action_loss_full_enum_best_pair_nll
action_loss_full_enum_learned_minus_true_gap
action_loss_full_enum_learned_minus_best_gap
action_loss_full_enum_learned_best_fraction
action_loss_full_enum_true_best_fraction
```

## Part 3: Diagnostics Before Primary Runs

Before running the full training ladder, add or extend diagnostics to support joint heads:

- `scripts/diagnose_calculator_protocol.py`
- `scripts/run_action_loss_diagnostic.py`
- `scripts/run_full_enum_action_loss_diagnostic.py`
- `scripts/diagnose_private_protocol.py`

Required diagnostic additions:

- Pair exact match.
- Result-equivalent pair accuracy.
- Pair entropy/effective pair count.
- Learned pair NLL, true pair NLL, and best pair NLL.
- Learned-best and true-best fractions.
- Tie-aware learned-best fraction, if straightforward.
- True-pair rank and probability for reporting only.

Tie-aware note:

```text
For addition, many pairs can have indistinguishable or near-indistinguishable answer NLL.
Report whether the learned pair is within a small NLL tolerance of the best pair,
for example <= best_nll + 1e-3 and <= best_nll + 1e-2.
```

## Part 4: Smoke Test on One Seed

Run one short selected-checkpoint continuation before the full ladder:

| Run | Start checkpoint | Steps | Input LR | Objective | Aux | Anchor |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| Joint-smoke-selected-seed2 | dense seed2 step 550 | 200 | `0.0003` | full-enum joint pair | `0.0` | `0.0` |

Use:

```text
--snapshot-every 50
--checkpoint-every 50
--snapshot-samples 128
```

Proceed to the primary ladder only if:

- the run completes without breaking canonical diagnostics;
- injection-zero remains near zero;
- oracle-at-eval remains high;
- the joint head produces nontrivial pair distributions rather than a constant action collapse.

## Part 5: Primary Selected-Checkpoint Continuations

Run three 500-step selected-checkpoint continuations:

| Run | Start checkpoint | Steps | Input LR | Objective | Aux | Anchor |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| Joint-selected-cont-seed1 | dense seed1 step 100 | 500 | `0.0003` | full-enum joint pair | `0.0` | `0.0` |
| Joint-selected-cont-seed2 | dense seed2 step 550 | 500 | `0.0003` | full-enum joint pair | `0.0` | `0.0` |
| Joint-selected-cont-seed3 | dense seed3 step 1050 | 500 | `0.0003` | full-enum joint pair | `0.0` | `0.0` |

Use dense checkpoints:

```text
--snapshot-every 50
--checkpoint-every 50
```

Select snapshots by canonical learned-minus-best or learned-minus-true action-loss gap, plus pair diagnostics. Do not select by eval exact alone.

## Part 6: Stage-B Comparison Gate

Only if at least two selected-checkpoint continuations improve over their starts, run Stage-B starts:

| Run | Start | Steps | Input LR | Objective | Aux | Anchor |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| Joint-stageB-seed1 | Stage B | 1000 | `0.0003` | full-enum joint pair | `0.0` | `0.0` |
| Joint-stageB-seed2 | Stage B | 1000 | `0.0003` | full-enum joint pair | `0.0` | `0.0` |
| Joint-stageB-seed3 | Stage B | 1000 | `0.0003` | full-enum joint pair | `0.0` | `0.0` |

The point of this gate is to avoid spending compute if joint training only rescues already-selected checkpoints.

## Part 7: Required Comparisons

Compare against:

- Stage B handoff.
- Prior lower-LR retention checkpoints.
- Best Phase 2 action-loss self-training checkpoint.
- Best Phase 2 full-enum selected snapshot.
- Independent-head full-enum finals and selected snapshots.

For each primary final and selected snapshot, report:

- Built-in eval exact.
- Canonical causal normal exact.
- Injection-zero, forced-zero, forced-random, oracle-at-eval.
- Canonical causal classification and bottleneck label.
- Forced-result sweep learned-best and true-sum-best fractions.
- Canonical action-loss learned, true, random, and shuffled NLL.
- Learned-minus-true and learned-minus-best gaps.
- Learned-best, true-best, and tie-aware learned-best fractions.
- Private-protocol all-pair answer exact.
- Pair exact match.
- Result-equivalent pair accuracy.
- Calculator-result accuracy.
- Group behavior for carry/no-carry, large/small operands, and symmetric pairs.
- Pair entropy/effective pair count.
- Proof of aux `0.0`, anchor `0.0`, frozen upstream, and interface-only trainable parameters.

## Decision Criteria

A useful positive result requires at least two of three selected-checkpoint continuations to improve over their starts on:

```text
learned-minus-best or learned-minus-true action-loss gap
pair exact or result-equivalent pair accuracy
private-protocol calculator-result accuracy
```

while preserving:

```text
injection-zero near zero
oracle-at-eval high
aux exactly 0.0
anchor exactly 0.0
upstream frozen
```

A strong positive result would show at least one of:

```text
learned-best or tie-aware learned-best fraction becomes meaningfully nonzero
result-equivalent pair accuracy exceeds the best independent-head Phase 2 checkpoint
private-protocol all-pair accuracy beats the best Phase 2 full-enum snapshot by a clear margin
```

A useful negative result should explain which part failed:

- The joint head collapsed to a constant pair.
- The full-enum teacher remained too broad even for pair-level CE.
- The learned pair improved NLL but not calculator-result accuracy.
- Pair-level training helped selected starts but not Stage-B starts.
- Diagnostics show the bottleneck was unhealthy, making the result invalid.

## Required Reporting

Write a Phase 3 work history:

```text
aiAgentWorkHistory/phase3/YYYY-MM-DD-joint-pair-action-interface.md
```

Create or update a Phase 3 fact sheet:

```text
factSheets/PHASE_3_EXPERIMENT_FACT_SHEET.md
```

Report:

- exact code changes;
- exact commands and run paths;
- smoke-test result;
- selected-checkpoint continuation table;
- Stage-B comparison table if run;
- canonical causal diagnostic table;
- canonical action-loss table;
- full-enum joint diagnostic table;
- private-protocol table;
- comparison against best Phase 2 checkpoints;
- explicit go/no-go recommendation for the next Phase 3 task.

## Follow-Up Decision

If joint pair training improves the interface, create the next task for either:

```text
joint pair head + controlled upstream unfreezing
```

or:

```text
joint pair head + identifiability curriculum
```

If it fails, create the next task for Track B:

```text
addition-only answer loss is underidentified; test a task/tool curriculum where
operand identity is rewarded directly by the task structure.
```

