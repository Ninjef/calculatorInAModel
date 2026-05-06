# Matched Retention Ladder for Joint Identity Curriculum

## Mission

Finish the evidence gap left by the Track B/C bridge.

Phase 3 has now learned three important things:

```text
1. A direct joint pair head trained only from full-enum answer NLL does not learn
   a useful pair protocol from the frozen Track A readout.
2. A true-pair identity curriculum with upstream gradients can create retained
   nonzero pair structure after aux identity pressure reaches exactly 0.0.
3. A new interface-objective decay switch works, and a one-seed calibration hints
   that turning off the underidentified full-enum interface target after handoff
   may reduce later collapse.
```

The next task is not to invent another mechanism yet. It is to complete the matched replication ladder that tells us whether the signal is real, seed-lucky, or washed out by continued post-handoff interface training.

The output should be a decision-quality result:

```text
Either the decayed-interface condition reproducibly preserves a retained joint
identity protocol, or Phase 3 should move on to a sharper identifiability task
instead of continuing to tune this curriculum.
```

## Current Evidence

Known Track B aux-zero handoff positive:

```text
runs/2026-05-06_091027_975713_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed204/checkpoint_snapshots/step_00150_weights.pt
```

Key retained diagnostics:

| Metric | Value |
| --- | ---: |
| Canonical normal exact | `0.2500` |
| Injection-zero exact | `0.0078` |
| Forced-random exact | `0.0156` |
| Oracle-at-eval exact | `0.9414` |
| Canonical pair exact | `0.2305` |
| Canonical result-equivalent pair accuracy | `0.2813` |
| Private all-pair answer exact | `0.2400` |
| Private pair exact | `0.1925` |
| Private calculator-result accuracy | `0.2575` |
| Pair-logit effective pairs | `398.9657` |

Matched one-seed calibration from the interface-decay implementation:

| Variant | Step | Interface weight | Aux weight | Normal | Injection-zero | Forced-random | Oracle | Pair exact | Calc result acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Constant interface | 150 | `1.0000` | `0.0000` | `0.1484` | `0.0000` | `0.0234` | `0.8906` | `0.1172` | `0.1719` |
| Constant interface | 225 | `1.0000` | `0.0000` | `0.0547` | `0.0156` | `0.0078` | `0.9531` | `0.0156` | `0.0547` |
| Decayed interface | 150 | `0.0000` | `0.0000` | `0.1250` | `0.0000` | `0.0234` | `0.8906` | `0.1172` | `0.1484` |
| Decayed interface | 200 | `0.0000` | `0.0000` | `0.2344` | `0.0078` | `0.0391` | `0.9609` | `0.1797` | `0.2422` |
| Decayed interface | 225 | `0.0000` | `0.0000` | `0.1484` | `0.0156` | `0.0078` | `0.9531` | `0.1094` | `0.1641` |

Run paths:

```text
runs/2026-05-06_093654_713217_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed213

runs/2026-05-06_094254_634251_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-ifacedecay150-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed213
```

This calibration is not a positive result by itself. It is a reason to finish the matched ladder before trying a slower decay, anchors, entropy, larger models, or new objectives.

## Fixed Controls

Preserve these unless the task explicitly says otherwise:

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
answer_loss_weight=1.0
input_proj_anchor_weight=0.0
semantic decoder checkpoint:
runs/2026-05-03_114747_070474_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed4/checkpoint_snapshots/step_00550_weights.pt
```

For every retained-checkpoint claim:

```text
aux identity weight must be exactly 0.0
input_proj_anchor_weight must be exactly 0.0
injection-zero must remain near zero
forced-random must remain near zero
oracle-at-eval must remain high
trainable parameter groups must be reported
```

For decayed-interface claims, also prove:

```text
adaptive_interface_loss_weight at the selected checkpoint is exactly 0.0
final_adaptive_interface_loss_weight is exactly 0.0
```

## Required Implementation

Do not add a new training mechanism in this task.

Allowed small implementation work:

- If useful, add a lightweight summary/selection helper that reads `diagnostic_snapshots.csv`, `training_curve.csv`, and `metrics.json` for a set of runs and emits the selected checkpoint table.
- If such a helper is added, keep it script-level, focused, and covered by a small test or by checked sample output.

Do not add anchors, entropy bonuses, new heads, new bottlenecks, or slower aux decay until this matched ladder is summarized.

## Experiment Plan

Complete a matched three-seed ladder.

The existing seed-211 pair may count as one matched seed:

```text
--seed 211
```

Run at least two fresh matched seed pairs:

```text
--seed 221
--seed 231
```

If runtime is acceptable, add one extra pair:

```text
--seed 241
```

### Stage 1: Constant Full-Enum Interface Objective

For each fresh seed, run:

```text
python3 scripts/overfit_one_batch.py \
  --variant model-c \
  --digits 2 \
  --steps 225 \
  --batch-size 64 \
  --eval-samples 256 \
  --operand-max 19 \
  --calculator-operand-vocab-size 20 \
  --calculator-estimator action_loss_full_enum_joint_interface \
  --calculator-action-head joint_pair \
  --semantic-decoder-checkpoint runs/2026-05-03_114747_070474_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed4/checkpoint_snapshots/step_00550_weights.pt \
  --adaptive-interface-loss-weight 1.0 \
  --input-proj-lr 0.001 \
  --upstream-lr 0.0003 \
  --calculator-read-position operands \
  --calculator-bottleneck-mode answer_decoder \
  --n-layer 2 \
  --n-head 1 \
  --n-embd 16 \
  --mlp-expansion 1 \
  --calculator-hook-after-layer 1 \
  --action-loss-full-enum-temperature 1.0 \
  --action-loss-full-enum-chunk-size 64 \
  --aux-operand-loss-weight 5.0 \
  --aux-operand-loss-decay-steps 150 \
  --aux-operand-loss-grad-upstream \
  --snapshot-every 25 \
  --checkpoint-every 25 \
  --snapshot-samples 128 \
  --seed SEED \
  --log-every 25
```

### Stage 2: Decay Interface Objective to Zero at Handoff

For the same seeds, run the matched condition:

```text
python3 scripts/overfit_one_batch.py \
  --variant model-c \
  --digits 2 \
  --steps 225 \
  --batch-size 64 \
  --eval-samples 256 \
  --operand-max 19 \
  --calculator-operand-vocab-size 20 \
  --calculator-estimator action_loss_full_enum_joint_interface \
  --calculator-action-head joint_pair \
  --semantic-decoder-checkpoint runs/2026-05-03_114747_070474_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed4/checkpoint_snapshots/step_00550_weights.pt \
  --adaptive-interface-loss-weight 1.0 \
  --adaptive-interface-loss-decay-steps 150 \
  --adaptive-interface-loss-floor 0.0 \
  --input-proj-lr 0.001 \
  --upstream-lr 0.0003 \
  --calculator-read-position operands \
  --calculator-bottleneck-mode answer_decoder \
  --n-layer 2 \
  --n-head 1 \
  --n-embd 16 \
  --mlp-expansion 1 \
  --calculator-hook-after-layer 1 \
  --action-loss-full-enum-temperature 1.0 \
  --action-loss-full-enum-chunk-size 64 \
  --aux-operand-loss-weight 5.0 \
  --aux-operand-loss-decay-steps 150 \
  --aux-operand-loss-grad-upstream \
  --snapshot-every 25 \
  --checkpoint-every 25 \
  --snapshot-samples 128 \
  --seed SEED \
  --log-every 25
```

## Selection Rule

Use the window:

```text
steps 125, 150, 175, 200, 225
```

For each run, select the highest `pair_exact_match` checkpoint among rows where:

```text
aux_operand_loss_weight == 0.0
```

For decayed-interface runs, additionally require:

```text
adaptive_interface_loss_weight == 0.0
```

Tie-breakers:

```text
1. lower injection_zero_exact_match
2. lower forced_random_exact_match
3. higher oracle_exact_match
4. higher calculator_result_accuracy
```

Report both:

- per-run selected checkpoints;
- the trajectory from handoff through the final step.

## Required Diagnostics

Run full diagnostics for each selected checkpoint if the three-seed ladder is small enough to keep runtime reasonable. If runtime becomes a blocker, run full diagnostics for:

```text
1. the best constant-interface selected checkpoint;
2. the best decayed-interface selected checkpoint;
3. any additional selected checkpoint that crosses canonical pair exact >= 0.25.
```

Diagnostics:

- Built-in eval exact from `metrics.json`.
- Canonical causal diagnostics with normal, injection-zero, forced-zero, forced-random, oracle-at-eval, and read-vector corrupt/swap interventions.
- Pair exact and result-equivalent pair accuracy.
- Full-enum learned, true, and best NLL.
- Learned-minus-true and learned-minus-best gaps.
- Learned-best and tie-aware learned-best fractions.
- Pair-logit entropy/effective pairs.
- Private-protocol all-pair answer exact, operand exact, pair exact, and calculator-result accuracy.
- Group behavior for carry/no-carry, large/small operands, and symmetric pairs.
- Proof of selected-checkpoint aux/interface/anchor weights.
- Proof of trainable parameter groups.

Use the canonical scripts:

```text
scripts/run_causal_calculator_protocol_diagnostics.py
scripts/run_full_enum_action_loss_diagnostic.py
scripts/diagnose_private_protocol.py
```

## Decision Criteria

### Strong Positive

At least two decayed-interface seeds have selected retained checkpoints with:

```text
canonical pair exact >= 0.35
private pair exact >= 0.35
private calculator-result accuracy >= 0.40
injection-zero <= 0.02
forced-random <= 0.05
oracle-at-eval >= 0.90
aux weight == 0.0
interface weight == 0.0
```

If this happens, run the slower aux-decay stabilizer next, because the mechanism is real but still below the best Phase 2 private-protocol strength.

### Weak Positive

At least two decayed-interface seeds beat their matched constant-interface run on retained pair exact or calculator-result accuracy at selected aux-zero checkpoints, while preserving the causal controls.

If this happens but strong-positive thresholds are missed, run exactly one stabilizer:

```text
--aux-operand-loss-decay-steps 300
--adaptive-interface-loss-decay-steps 300
--steps 375
```

Use the two best seeds only. Do not launch a broad grid.

### Negative

If the decayed-interface condition does not improve retention across seeds, stop tuning this auxiliary identity curriculum. The next phase task should move to a sharper identifiability environment where the answer signal itself identifies operands, such as multi-operation prompts or curriculum-only structured calculator returns.

## Reporting Requirements

Update:

```text
factSheets/PHASE_3_EXPERIMENT_FACT_SHEET.md
aiAgentWorkHistory/phase3/
```

The report must include:

- all run paths;
- selected checkpoint paths;
- snapshot trajectory tables for every matched seed;
- aggregate constant vs decayed comparison;
- full diagnostic summaries for selected checkpoints;
- explicit decision under the criteria above;
- recommendation for either the slower-decay stabilizer or moving to a sharper identifiability task.
