# Dense Replication and Retention Stabilization for Joint Identity Curriculum

## Mission

The Track B identity-curriculum smoke produced the first retained nonzero joint pair structure after auxiliary identity pressure reached exactly `0.0`, but the signal was transient and diffuse:

```text
At the aux-zero handoff, pair/result behavior improved substantially over the Track A joint smoke.
After additional zero-aux training, the retained pair protocol degraded.
Pair-logit entropy stayed near uniform, so the interface did not become a confident clean protocol.
```

This task should determine whether the Track B result is:

```text
1. a reproducible aux-zero handoff phenomenon;
2. a seed-lucky transient;
3. degraded by continued answer-NLL/full-enum interface training after aux reaches zero;
4. stabilizable with the smallest honest retention intervention.
```

The goal is not to maximize final-step eval by any means. The goal is to establish a reliable retained checkpoint, with identity-specific pressure exactly `0.0`, that remains calculator-dependent and either approaches the best Phase 2 private-protocol strength or clearly explains why it does not.

## Starting Evidence

Track B upstream-open smoke:

```text
runs/2026-05-06_091027_975713_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed204
```

Selected aux-zero checkpoint:

```text
checkpoint_snapshots/step_00150_weights.pt
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
| Full-enum learned-minus-true gap | `5.2297` |
| Full-enum learned-best fraction | `0.0234` |
| Private all-pair answer exact | `0.2400` |
| Private pair exact | `0.1925` |
| Private calculator-result accuracy | `0.2575` |
| Pair-logit effective pairs | `398.9657` |

Degradation after aux-zero handoff:

| Step | Aux weight | Normal | Pair exact | Calc result acc |
| ---: | ---: | ---: | ---: | ---: |
| 100 | `1.6667` | `0.2969` | `0.2500` | `0.3125` |
| 150 | `0.0000` | `0.3906` | `0.2188` | `0.3906` |
| 200 | `0.0000` | `0.1563` | `0.1563` | `0.1563` |
| 250 | `0.0000` | `0.0313` | `0.0313` | `0.0313` |
| 300 | `0.0000` | `0.0625` | `0.0313` | `0.0625` |

Frozen-upstream control:

```text
runs/2026-05-06_090608_992797_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.0003-uplr0.0003-fullt1-fullchunk64-answer_decoder-aux1-auxdecay150/model-c-2digit-seed203
```

Finding: pair-head-only training from the frozen Track A readout did not learn identity. Do not spend primary time on that variant unless it is only a control.

Best Phase 2 comparison targets:

```text
private all-pair answer exact: roughly 0.5300..0.5375
private operand/pair exact: roughly 0.5500..0.5675
private calculator-result accuracy: roughly 0.5750..0.5775
best learned-minus-true gaps: roughly 1.9814..2.5646
```

## Fixed Controls

Preserve these unless the task explicitly varies them:

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
input_proj_anchor_weight=0.0 unless explicitly testing an anchor
semantic decoder checkpoint:
runs/2026-05-03_114747_070474_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed4/checkpoint_snapshots/step_00550_weights.pt
```

For any retained-checkpoint claim:

```text
aux identity weight at selected checkpoint must be exactly 0.0
input_proj_anchor_weight must be exactly 0.0 unless the claim is explicitly anchor-based
injection-zero must remain near zero
forced-random must remain near zero
oracle-at-eval must remain high
trainable parameter groups must be reported
```

## Required Implementation

Add the smallest training scheduler needed to separate handoff quality from post-handoff degradation.

Recommended implementation:

- Add an optional schedule for the action/full-enum interface objective weight, reusing `adaptive_interface_loss_weight` as the initial weight.
- Example CLI shape:

```text
--adaptive-interface-loss-decay-steps N
--adaptive-interface-loss-floor F
```

- The schedule should mirror `auxiliary_operand_weight` semantics:

```text
if initial weight <= 0: scheduled weight = 0
if decay_steps <= 0: scheduled weight = initial weight
else linearly decay to floor by step N
```

- Log the scheduled interface weight in `training_curve.csv`.
- Save final scheduled interface weight in `metrics.json`.
- Preserve current behavior when the new args are omitted.
- Add focused tests for the schedule helper and one smoke assertion that a decayed interface weight reaches exactly `0.0`.

Rationale:

The current post-zero phase continues applying broad full-enum answer-NLL targets. That teacher is useful for answer-result behavior but is underidentified for true operand identity. A decay/off switch lets us ask whether identity retention is washed out by the post-zero interface objective rather than by the absence of identity pressure itself.

## Experiment Plan

Run a compact ladder before any long sweep.

### Stage 1: Reproduce the Aux-Zero Handoff

Run three seeds of the upstream-open identity curriculum using the previous successful shape:

```text
--calculator-estimator action_loss_full_enum_joint_interface
--calculator-action-head joint_pair
--input-proj-lr 0.001
--upstream-lr 0.0003
--aux-operand-loss-weight 5.0
--aux-operand-loss-decay-steps 150
--aux-operand-loss-grad-upstream
--adaptive-interface-loss-weight 1.0
--snapshot-every 25
--checkpoint-every 25
--snapshot-samples 128
--steps 225
```

Use at least three seeds, including the previous seed family if helpful, but do not reuse only the known lucky seed.

Selection window:

```text
steps 125, 150, 175, 200, 225
```

Primary selection metric:

```text
highest snapshot pair exact among checkpoints whose aux weight is exactly 0.0,
breaking ties by lower injection-zero and higher oracle-at-eval.
```

### Stage 2: Test Whether Full-Enum Interface Training Washes Out Identity

Run the same seeds with the new scheduled interface objective:

```text
--adaptive-interface-loss-weight 1.0
--adaptive-interface-loss-decay-steps 150
--adaptive-interface-loss-floor 0.0
```

This makes both identity pressure and full-enum interface pressure reach zero at the handoff. Continue training to at least step 225 with answer loss only after handoff.

Primary question:

```text
Does pair exact/private calculator-result accuracy persist better from step 150 to 225
when the underidentified full-enum interface target is removed?
```

### Stage 3: Only If Needed, Try One Stabilizer

If Stage 2 preserves handoff quality but still underperforms Phase 2, test exactly one conservative stabilizer:

Preferred first stabilizer:

```text
slower aux decay:
--aux-operand-loss-weight 5.0
--aux-operand-loss-decay-steps 300
--steps 375
--snapshot-every 25
--checkpoint-every 25
```

Keep the interface objective decay aligned with aux decay unless Stage 2 showed that full-enum pressure helps retention.

Do not launch a large grid. The point is to identify the retention mechanism, not to bury the signal in variants.

## Required Diagnostics

For every selected retained checkpoint:

- Built-in eval exact.
- Canonical causal diagnostics with:
  - normal;
  - injection-zero;
  - forced-zero;
  - forced-random;
  - oracle-at-eval;
  - read-vector corrupt/swap interventions.
- Pair exact and result-equivalent pair accuracy.
- Full-enum learned, true, and best NLL.
- Learned-minus-true and learned-minus-best gaps.
- Learned-best and tie-aware learned-best fractions.
- Pair-logit entropy/effective pairs.
- Private-protocol all-pair answer exact, operand exact, pair exact, and calculator-result accuracy.
- Group behavior for carry/no-carry, large/small operands, and symmetric pairs.
- Proof of selected-checkpoint aux/interface/anchor weights.
- Proof of trainable parameter groups.

Also report per-run snapshot trajectories from handoff through the end, not just the selected checkpoint.

## Decision Criteria

### Strong Positive

A strong positive requires at least two seeds with selected aux-zero checkpoints that:

```text
canonical pair exact >= 0.35
private pair exact >= 0.35
private calculator-result accuracy >= 0.40
injection-zero <= 0.02
forced-random <= 0.05
oracle-at-eval >= 0.90
```

and at least one of:

```text
learned-minus-true gap <= 4.0
learned-best fraction > 0.02
pair-logit effective pairs clearly below the Track B smoke's ~399
```

### Useful Mixed Result

A useful mixed result is enough if it cleanly distinguishes:

```text
the aux-zero handoff is reproducible but not retained;
post-zero full-enum interface pressure causes degradation;
answer-only post-handoff causes degradation;
the signal is seed-specific;
the signal improves Track A but cannot approach Phase 2.
```

### No-Go

No-go if:

```text
fewer than two seeds reproduce nonzero aux-zero pair exact above 0.15;
or calculator-dependence controls fail;
or all apparent gains disappear in private all-pair diagnostics;
or the only useful checkpoint requires nonzero aux/anchor weight.
```

## Required Reporting

Update `factSheets/PHASE_3_EXPERIMENT_FACT_SHEET.md` and add a Phase 3 work history with:

- exact code changes;
- exact commands and run paths;
- scheduler semantics and tests;
- selected checkpoint table by seed and variant;
- snapshot trajectory table around the aux-zero handoff;
- canonical causal table;
- full-enum action-loss table;
- private-protocol table;
- comparison against:
  - Track A joint smoke;
  - Track B step-150 smoke;
  - best Phase 2 selected/private-protocol checkpoints;
- clear go/no-go recommendation for the following Phase 3 task.

Commit and push after completing the task.
