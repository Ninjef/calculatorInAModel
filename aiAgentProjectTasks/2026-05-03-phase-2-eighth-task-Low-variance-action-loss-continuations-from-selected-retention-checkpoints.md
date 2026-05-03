# Low-Variance Action-Loss Continuations From Selected Retention Checkpoints

## Mission

The previous phase-2 task established two useful facts:

```text
1. Dense checkpoint selection works: action-loss-selected snapshots beat final
   checkpoints in all three lower-LR Stage C dense runs on canonical
   learned-minus-true action-loss gap.

2. Candidate action search contains real answer-NLL signal: learned/top-k/local/
   random candidate pools often contain actions that beat the current learned
   action without using true operands.
```

But the first `action_loss_weighted_interface` objective was not robust:

```text
Self-training seed2 improved substantially, but seed1 and seed3 did not.
Learned-best action-loss fraction stayed 0.0 in all three self-training runs.
```

The next question is whether the signal becomes trainable if we reduce target
variance and start from action-loss-selected retained interfaces rather than
always starting directly from Stage B.

## Important Settled Fact: Do Not Re-Prove Downstream Oracle Use

We already know the downstream answer decoder works when supplied true operands.
This has been demonstrated repeatedly across phase 2.

For this task, `oracle_at_eval` is only a cheap invariant/smoke test:

```text
Use it to confirm the frozen decoder and bottleneck wiring are still intact.
Do not spend research time analyzing or rediscovering that true operands work.
If oracle_at_eval unexpectedly drops, report it as a wiring/regression problem.
Otherwise move on.
```

The real frontier is improving learned calculator actions without true-operand
supervision.

## Starting Points

Stage B handoff checkpoint:

```text
runs/2026-05-01_112523_133504_model-c-op0-19-adaptive_interface-inlr0.003-uplr0.003-answer_decoder-aux1-auxdecay500/model-c-2digit-seed2/final_weights.pt
```

Best prior lower-LR replication:

```text
runs/2026-05-01_122402_250571_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed3/final_weights.pt
```

Action-loss-selected dense checkpoints:

```text
runs/2026-05-03_112750_450950_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed3/checkpoint_snapshots/step_00100_weights.pt
runs/2026-05-03_114747_070474_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed4/checkpoint_snapshots/step_00550_weights.pt
runs/2026-05-03_114747_345486_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed5/checkpoint_snapshots/step_01050_weights.pt
```

Best current self-training checkpoint:

```text
runs/2026-05-03_154959_116705_model-c-op0-19-action_loss_weighted_interface-inlr0.0003-uplr0.0003-alrand4-altop1-alloc1-alt1-answer_decoder/model-c-2digit-seed4/final_weights.pt
```

Useful scripts:

```text
scripts/overfit_one_batch.py
scripts/run_action_loss_candidate_diagnostic.py
scripts/run_action_loss_diagnostic.py
scripts/run_causal_calculator_protocol_diagnostics.py
scripts/diagnose_private_protocol.py
```

## Fixed Regime

Keep the strict bottleneck:

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

Do not unfreeze upstream in this task.
Do not add true-operand auxiliary supervision.
Do not use true operands to construct action-loss training targets.

## Part 1: Implement a Lower-Variance Action-Loss Variant

Add one narrow variant or extension to `action_loss_weighted_interface`.

Preferred candidate:

```text
calculator_estimator=action_loss_replay_interface
```

Suggested behavior:

- Maintain a small per-prompt or per-example replay/cache of candidate action
  targets scored by answer NLL.
- Refresh the candidate set periodically, not every gradient step.
- Train `calculator_hook.input_proj` toward a stabilized soft target over cached
  candidate pairs.
- Include the current learned action, top-k actions, local perturbations, and
  random actions in the refresh pool.
- Use only answer NLL through the frozen decoder to rank/weight candidates.
- Do not include true operands or true sums in the target construction.

If replay/cache is too large for one pass, implement a simpler low-variance
variant first, such as:

```text
action_loss_candidate_refresh_every=N
action_loss_candidate_ema_beta=B
```

The point is to reduce per-step target churn and see whether the answer-loss
signal becomes optimizable across seeds.

## Part 2: Start From Selected Retention Checkpoints

Run action-loss continuation from the three action-loss-selected dense
checkpoints, not only from Stage B.

Primary continuations:

| Run | Start checkpoint | Steps | Input LR | Objective | Aux | Anchor |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| AL-selected-cont-seed1 | dense seed1 step 100 | 500 | `0.0003` | low-variance action-loss | `0.0` | `0.0` |
| AL-selected-cont-seed2 | dense seed2 step 550 | 500 | `0.0003` | low-variance action-loss | `0.0` | `0.0` |
| AL-selected-cont-seed3 | dense seed3 step 1050 | 500 | `0.0003` | low-variance action-loss | `0.0` | `0.0` |

Use dense snapshots for these continuations:

```text
--snapshot-every 50
--checkpoint-every 50
```

Selection should be by canonical action-loss learned-minus-true gap, not by
oracle-at-eval.

## Part 3: Compare Against Stage-B-Started Self-Training

If Part 2 looks promising, run three Stage-B-started variants with the same
low-variance objective:

| Run | Start | Steps | Input LR | Objective | Aux | Anchor |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| AL-lowvar-stageB-seed1 | Stage B | 1000 | `0.0003` | low-variance action-loss | `0.0` | `0.0` |
| AL-lowvar-stageB-seed2 | Stage B | 1000 | `0.0003` | low-variance action-loss | `0.0` | `0.0` |
| AL-lowvar-stageB-seed3 | Stage B | 1000 | `0.0003` | low-variance action-loss | `0.0` | `0.0` |

Only run these after the selected-checkpoint continuations show at least one
clear positive and no obvious systematic degradation.

## Part 4: Required Diagnostics

Use oracle-at-eval only as a guardrail.

For each primary final and selected snapshot, report:

- built-in eval exact;
- learned-target agreement if available;
- canonical action-loss true, learned, random, shuffled NLL;
- learned-minus-true, random-minus-true, shuffled-minus-true gaps;
- true-best and learned-best fractions;
- canonical causal classification and bottleneck label;
- injection-zero, forced-zero, forced-random, oracle-at-eval;
- forced-result sweep learned-best and true-sum-best fractions;
- private-protocol all-pair operand exact and calculator-result accuracy;
- group behavior for carry/no-carry and large/small operands for the best run.

For oracle-at-eval, one table entry is enough unless it fails.

## Decision Criteria

A useful positive result requires at least two of three selected-checkpoint
continuations to improve over their starting checkpoints on:

```text
canonical learned-minus-true action-loss gap
operand exact or calculator-result accuracy
```

and to preserve:

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
private-protocol decoding shows clearer true-operand-like structure than the
best current self-training checkpoint.
```

A useful negative result should distinguish:

- selected checkpoints are good stopping points but bad continuation starts;
- candidate signal exists but per-step target churn prevents optimization;
- replay/EMA stabilizes loss but not true operand/calculator-result structure;
- improvements remain seed-specific and not robust enough for upstream unfreezing.

## Required Reporting

Write a phase-2 work history with:

- exact commands and run paths;
- code changes;
- proof that aux stayed exactly `0.0`;
- proof that anchor stayed exactly `0.0`;
- trainable parameter groups;
- selected-checkpoint continuation table;
- Stage-B-started comparison table if run;
- canonical action-loss summary;
- causal diagnostic summary;
- private-protocol summary;
- explicit go/no-go recommendation for upstream unfreezing.

Update:

```text
factSheets/PHASE_2_EXPERIMENT_FACT_SHEET.md
```

## Success Criteria

This task succeeds if it determines whether the answer-loss signal becomes
reliably trainable when target variance is reduced and continuations start from
action-loss-selected retained interfaces.

Do not spend this task re-establishing that true operands make the downstream
decoder work. That is a known invariant, not the research question.
