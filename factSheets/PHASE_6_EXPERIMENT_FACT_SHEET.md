# Phase 6 Experiment Fact Sheet

## Mission

Phase 6 moves from protocol teaching/retention toward protocol discovery:

```text
Can an answer-derived local interface target, in the identifiable Phase 4/5
task, teach the upstream/model-side calculator-query protocol without using
direct true-operand supervision?
```

## Fixed Setup

Unless a task explicitly says otherwise, use:

```text
digits=2
operand_max=19
calculator_operand_vocab_size=20
n_layer=2
n_head=1
n_embd=16
mlp_expansion=1
calculator_hook_after_layer=1
answer_format=sum_left_operand
calculator_output_format=sum_left_operand
calculator_read_position=operand_spans
calculator_read_span_width=2
calculator_bottleneck_mode=answer_decoder
freeze_semantic_decoder=true
answer_loss_weight=1.0
aux_operand_loss_weight=0.0
adaptive_interface_loss_weight=0.0 unless explicitly testing a local objective
input_proj_anchor_weight=0.0
oracle_train=false
oracle_warmup_steps=0
```

Standard Stage 0B semantic decoder checkpoint:

```text
runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt
```

If absent locally, use the absolute path recorded in the Phase 4 fact sheet.

## Anti-Oracle Guardrail

Oracle operands, oracle-at-eval, forced true result classes, injection-zero
controls, and forced-random controls are wiring checks only. Phase 6 progress
must be evaluated by learned-interface behavior:

- learned operand exact;
- learned pair exact;
- learned calculator-result accuracy;
- private all-pair protocol decoding;
- full-enum learned-minus-true and learned-minus-best action-loss gaps;
- local target weight exactly `0.0` for retention claims;
- aux/direct true-operand supervision weight exactly `0.0`;
- semantic decoder movement exactly `0.0`;
- dense checkpoint selection whenever upstream or interface parameters move.

## 2026-05-10 Identifiable Full-Enum Local Target

Task:

```text
aiAgentProjectTasks/2026-05-10-phase-6-first-task-Identifiable-full-enum-local-target-sharpness-and-smoke.md
```

Runner:

```text
scripts/run_phase6_identifiable_full_enum_local_target.py
```

Run root:

```text
runs/2026-05-10_phase6_identifiable_full_enum_local_target
```

### Target Sharpness

Full-enum answer-derived targets are sharp in the identifiable
`sum_left_operand` setup.

| Checkpoint | Best=true | Tie-aware true-best | Mean true rank | Effective pairs | True-pair prob |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage 0B full-model load | `1.000` | `1.000` | `1.000` | `1.079` | `0.989` |
| Phase 4 retained positive | `1.000` | `1.000` | `1.000` | `1.079` | `0.989` |

Interpretation:

- The Phase 6 bet passed Stage 0: in the identifiable task, the frozen
  answer-decoder NLL identifies the true action pair sharply.
- This is much sharper than the old Phase 2/3 addition-only soft target regime,
  where effective pair counts around `29` reflected underidentification.

### Local-Target Smoke

Both smoke branches used:

```text
oracle_train=false
oracle_warmup_steps=0
aux_operand_loss_weight=0.0
input_proj_anchor_weight=0.0
local_target_loss_weight=1.0
local_target_mode=hard_best_pair
semantic_decoder_checkpoint_load_scope=full_model
freeze_semantic_decoder=true
```

| Branch | Best step | Lightweight normal/operand/pair/calc | Final eval exact | Final local target | Trainable groups |
| --- | ---: | ---: | ---: | ---: | --- |
| frozen upstream | `250` | `0.688 / 0.688 / 0.688 / 0.688` | `0.402` | `1.0` | `calculator_hook.input_proj` |
| upstream open | `600` | `0.680 / 0.680 / 0.680 / 0.680` | `0.359` | `1.0` | `calculator_hook.input_proj`, `upstream` |

Selected full diagnostics:

| Checkpoint | Canonical operand/pair/calc | Private operand/pair/calc | Full-enum learned-true/best gap | Learned-best | Oracle | Injection-zero | Forced-random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage 0B | `0.0156 / 0.0156 / 0.0352` | `0.0125 / 0.0125 / 0.0325` | `9.379 / 9.379` | `0.0078` | `1.000` | `0.000` | `0.0039` |
| frozen best step `250` | `0.566 / 0.566 / 0.566` | `0.5825 / 0.5825 / 0.585` | `1.813 / 1.813` | `0.594` | `1.000` | `0.000` | `0.0234` |
| frozen final | `0.398 / 0.398 / 0.414` | `0.4225 / 0.4225 / 0.4325` | `3.557 / 3.557` | `0.375` | `1.000` | `0.000` | `0.0234` |
| upstream-open best step `600` | `0.734 / 0.734 / 0.734` | `0.6825 / 0.6825 / 0.6825` | `1.041 / 1.041` | `0.633` | `1.000` | `0.000` | `0.0273` |
| upstream-open final | `0.336 / 0.336 / 0.348` | `0.3375 / 0.3375 / 0.345` | `3.193 / 3.193` | `0.336` | `1.000` | `0.000` | `0.0234` |

Parameter movement versus each branch's step `0` checkpoint:

| Checkpoint | input-proj L2 / max | upstream L2 / max | semantic decoder L2 / max |
| --- | ---: | ---: | ---: |
| frozen best step `250` | `2.3458 / 0.2006` | `0.0 / 0.0` | `0.0 / 0.0` |
| frozen final | `4.4895 / 0.3990` | `0.0 / 0.0` | `0.0 / 0.0` |
| upstream-open best step `600` | `1.6701 / 0.1556` | `0.3410 / 0.0160` | `0.0 / 0.0` |
| upstream-open final | `2.6288 / 0.2561` | `0.6235 / 0.0268` | `0.0 / 0.0` |

Decision:

- Stage 2 local-target-off retention was not run because no Stage 1 snapshot
  reached the required `>=0.90` fast-gate threshold.
- The result is a useful positive for target identifiability and local target
  signal: best selected protocol metrics materially exceed the Phase 5
  no-handoff best partial checkpoints (`~0.43` canonical operand/pair/calc).
- The result is not a strong Phase 6 positive because the learned protocol did
  not reach near-exact and drifted by final checkpoints while the local target
  was still on.

Recommendation:

Try optimization/parameterization improvements before moving to strict random
upstream discovery: stronger hard-best weight, LR variants around the
input-proj branch, a soft/Gumbel relaxation, or a joint-pair head adapted to
`operand_spans`. Do not run broad answer-only sweeps.

## 2026-05-10 Matched Local-Target Teaching And Retention

Task:

```text
aiAgentProjectTasks/2026-05-10-phase-6-second-task-Matched-local-target-teaching-and-retention-gate.md
```

Runner:

```text
scripts/run_phase6_matched_local_target_teaching.py
```

Run root:

```text
runs/2026-05-10_phase6_matched_local_target_teaching
```

### Parity Gate

The new `compare-local-target-to-aux` gate passed on a fixed 128-sample
Stage 0B full-model batch:

| Metric | Value |
| --- | ---: |
| hard-best pair equals true pair | `1.000` |
| hard-best A target equals true A | `1.000` |
| hard-best B target equals true B | `1.000` |
| hard-best local CE | `2.995222` |
| direct aux CE on same logits | `2.995222` |
| local-minus-aux CE | `0.0` |
| effective pairs | `1.078` |
| true-pair probability | `0.988` |
| semantic decoder grad/delta | `0.0 / 0.0` |

The local target construction enumerated answer NLL over all `20 x 20` action
pairs and did not use true operands to choose the target; true operands were
used only afterward for parity reporting and the aux-CE comparison.

### Matched Stage 1 Teaching

Branch A used the Phase 4-matched frozen-upstream teaching shape:

```text
answer_loss_weight=0.0
local_target_loss_weight=1.0
aux_operand_loss_weight=0.0
input_proj_anchor_weight=0.0
freeze_upstream_encoder=true
trainable=calculator_hook.input_proj only
input_proj_lr=0.03
steps=300
target_mode=hard_best_pair
```

Branch B was not run because Branch A passed the Stage 1 protocol gate.

| Stage 1 checkpoint | Fast-gate normal/operand/pair/calc | Canonical operand/pair/calc | Private answer/operand/pair/calc | Full-enum learned-true/best gap | Learned-best |
| --- | ---: | ---: | ---: | ---: | ---: |
| first gate, step `75` | `0.977 / 0.977 / 0.977 / 0.977` | not run | not run | not run | not run |
| first exact, step `125` | `1.000 / 1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000 / 1.000` | `0.0 / 0.0` | `1.000` |

## 2026-05-11 Gumbel/Concrete Hard-Forward Interface Bridge

Task:

```text
aiAgentProjectTasks/2026-05-11-phase-6-sixth-task-Gumbel-Concrete-hard-forward-interface-bridge.md
```

Run root:

```text
runs/2026-05-11_phase6_gumbel_concrete_interface_bridge
```

Code changes:

- Added `calculator_estimator=gumbel_concrete_interface`.
- Added deterministic and sampled Gumbel relaxed operand distributions, with
  hard-forward / soft-backward calculator signals:
  `hard_signal.detach() + soft_signal - soft_signal.detach()`.
- For `calculator_output_format=sum_left_operand`, the soft signal is
  `concat(p_sum, p_a)`, where `p_sum` is the convolution of independent
  operand distributions.
- Added CLI knobs for relaxed temperature schedule, mode, hard-forward switch,
  and entropy bonus.
- Added a Stage 0 gradient-gate helper:
  `scripts/run_phase6_gumbel_concrete_interface_bridge.py`.

### Stage 0 Gradient Gate

Command output:

```text
runs/2026-05-11_phase6_gumbel_concrete_interface_bridge/stage0/gradient_gate_temp2.json
```

Strict `semantic_decoder_only`, frozen-upstream, fixed 128-sample batch,
deterministic hard-forward relaxation at temperature `2.0`.

| Metric | Value |
| --- | ---: |
| oracle / injection-zero / forced-random | `1.000 / 0.000 / 0.000` |
| initial answer loss | `10.8585` |
| initial hard pair / calc | `0.000 / 0.0078` |
| initial entropy / effective pairs | `5.9915 / 399.999` |
| full-enum best=true | `1.000` |
| best-pair probability before / after one step | `0.002499 / 0.002520` |
| best-pair probability delta | `+0.0000209` |
| gradient cosine, relaxed answer vs hard-best CE | `+0.2345` |
| one-step input-proj delta L2 | `1.0896` |
| upstream delta L2 | `0.0` |
| semantic decoder grad / delta L2 | `0.0 / 0.0` |

Decision: gate passed. The relaxed answer-loss gradient moved probability
toward the full-enum best pair and was positively aligned with the diagnostic
hard-best CE gradient, while only `calculator_hook.input_proj` moved.

### Stage 1 Frozen-Upstream Relaxed Training

Branch A was run first and reached the fast-gate threshold, so Branches B-D
were skipped per the task's early-stop rule.

Shared setup:

```text
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
freeze_semantic_decoder=true
freeze_upstream_encoder=true
answer_loss_weight=1.0
aux_operand_loss_weight=0.0
adaptive_interface_loss_weight=0.0
expected_answer_loss_weight=0.0
input_proj_anchor_weight=0.0
input_proj_lr=0.03
steps=300
```

| Branch | Mode | Temperature | Entropy | Best snapshot normal/operand/pair/calc | Final snapshot | Final eval |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| A | deterministic | `2.0 -> 0.5` | `0.0` | step `200`: `1.000 / 1.000 / 1.000 / 1.000` | step `300`: `0.789 / 0.789 / 0.789 / 0.789` | `0.873` |
| B | skipped | `1.0 -> 0.25` | `0.0` | skipped after A gated | skipped | skipped |
| C | skipped | `1.0 -> 0.25` | decayed | skipped after A gated | skipped | skipped |
| D | skipped | gumbel | decayed | skipped after A gated | skipped | skipped |

Stage 1 selected checkpoint:

```text
runs/2026-05-11_phase6_gumbel_concrete_interface_bridge/stage1_branch_a_temp2_to_05/2026-05-11_154207_114387_model-c-op0-19-gumbel_concrete_interface-inlr0.03-uplr0.0003-rtemp2-rfinal0.5-rdecay300-answer_decoder-sum_left_operand/model-c-2digit-seed2/checkpoint_snapshots/step_00200_weights.pt
```

Parameter movement from Stage 1 step `0` to step `200`:

| Group | L2 | Max abs |
| --- | ---: | ---: |
| `calculator_hook.input_proj` | `26.7822` | `2.4764` |
| upstream encoder | `0.0` | `0.0` |
| semantic decoder | `0.0` | `0.0` |

### Stage 2 Relaxation-Off Retention

Both continuations used:

```text
calculator_estimator=adaptive_interface
semantic_decoder_checkpoint_load_scope=full_model
freeze_semantic_decoder=true
freeze_upstream_encoder=true
answer_loss_weight=1.0
aux_operand_loss_weight=0.0
adaptive_interface_loss_weight=0.0
expected_answer_loss_weight=0.0
input_proj_anchor_weight=0.0
input_proj_lr=0.0003
steps=1000
```

| Source | Step 0 snapshot | Best snapshot | Final snapshot normal/operand/pair/calc | Final eval |
| --- | ---: | ---: | ---: | ---: |
| first qualifying Stage 1 step `175` | `0.883 / 0.883 / 0.883 / 0.883` | step `50`: `1.000 / 1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000 / 1.000` | `1.000` |
| best Stage 1 step `200` | `0.992 / 0.992 / 0.992 / 0.992` | step `50`: `1.000 / 1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000 / 1.000` | `1.000` |

Selected retained checkpoint:

```text
runs/2026-05-11_phase6_gumbel_concrete_interface_bridge/stage2_retention_best_step200/2026-05-11_154745_323390_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt
```

Selected retained diagnostics:

| Diagnostic | Result |
| --- | ---: |
| canonical normal / oracle / injection-zero / forced-random | `1.000 / 1.000 / 0.0039 / 0.0313` |
| canonical operand / pair / calc | `1.000 / 1.000 / 1.000` |
| private answer / operand / pair / calc | `1.000 / 1.000 / 1.000 / 1.000` |
| full-enum learned / true / best NLL | `0.0002 / 0.0002 / 0.0002` |
| full-enum learned-minus-true / best gap | `0.0 / 0.0` |
| full-enum learned-best / true-best | `1.000 / 1.000` |

Parameter movement from retained step `0` to final, best-step continuation:

| Group | L2 | Max abs |
| --- | ---: | ---: |
| `calculator_hook.input_proj` | `0.8357` | `0.0820` |
| upstream encoder | `0.0` | `0.0` |
| semantic decoder | `0.0` | `0.0` |

Final objective weights for the retained selected checkpoint:

```text
aux_operand_loss_weight=0.0
adaptive/local target weight=0.0
expected_answer_loss_weight=0.0
relaxed objective inactive via calculator_estimator=adaptive_interface
relaxed entropy weight=0.0
input_proj_anchor_weight=0.0
```

Interpretation:

This is a strong Phase 6 positive for the hard-forward / soft-backward
relaxed bridge. Unlike the exact expected answer-loss branch, which reduced
expected cost but collapsed to wrong hard actions, the deterministic relaxed
answer-loss path trained hard learned actions to an exact protocol checkpoint
without true-operand labels, oracle operands during training, or hard-best CE.
The learned protocol then retained with the relaxation fully off.

## 2026-05-10 Matched Local-Target Teaching And Retention, Continued

| final, step `300` | `1.000 / 1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000 / 1.000` | `0.0 / 0.0` | `1.000` |

Stage 1 parameter movement versus its step `0` checkpoint:

| Checkpoint | input-proj L2 / max | upstream L2 / max | semantic decoder L2 / max |
| --- | ---: | ---: | ---: |
| step `125` | `91.983 / 3.732` | `0.0 / 0.0` | `0.0 / 0.0` |
| final | `209.383 / 8.844` | `0.0 / 0.0` | `0.0 / 0.0` |

### Local-Target-Off Retention

Two frozen-upstream Stage 2 continuations were run with:

```text
answer_loss_weight=1.0
local_target_loss_weight=0.0
adaptive_interface_loss_weight=0.0
aux_operand_loss_weight=0.0
input_proj_anchor_weight=0.0
freeze_upstream_encoder=true
input_proj_lr=0.0003
steps=1000
```

| Stage 2 start | Final fast-gate normal/operand/pair/calc | Canonical operand/pair/calc | Private answer/operand/pair/calc | Full-enum learned-true/best gap | Learned-best |
| --- | ---: | ---: | ---: | ---: | ---: |
| first gate step `75` | `1.000 / 1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000 / 1.000` | `0.0 / 0.0` | `1.000` |
| first exact/best step `125` | `1.000 / 1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000 / 1.000` | `0.0 / 0.0` | `1.000` |

The first-gate retention branch had a useful intermediate nuance: its selected
step `150` snapshot was canonical-exact, but private was `0.9975` and full-enum
learned-best was `0.984` with learned-minus-true/best gap `0.1065`. Its final
checkpoint closed those gaps to exact.

Retention parameter movement versus each Stage 2 source checkpoint:

| Stage 2 start | input-proj L2 / max | upstream L2 / max | semantic decoder L2 / max |
| --- | ---: | ---: | ---: |
| first gate step `75` | `2.694 / 0.177` | `0.0 / 0.0` | `0.0 / 0.0` |
| first exact/best step `125` | `3.277 / 0.220` | `0.0 / 0.0` | `0.0 / 0.0` |

Decision:

- Strong Phase 6 positive in the full-model branch: the answer-derived hard-best
  local target replaced direct true-operand labels for Stage 1 teaching, and the
  learned protocol was retained after the local target was exactly `0.0`.
- This is not strict random-upstream discovery; all successful branches used
  the Stage 0B full-model load with frozen upstream and trainable
  `calculator_hook.input_proj`.
- Compared with the Phase 6 first smoke (`0.566` frozen best canonical
  operand/pair/calc), the Phase 4-matched recipe reached exact protocol metrics
  and passed retention.

Recommendation:

Proceed to the next Phase 6 branch: either optional upstream-open retention from
the retained local-target-off checkpoint, or the stricter
`semantic_decoder_only` random-upstream local-target teaching branch with the
same parity gate and dense diagnostics. Do not rerun oracle-only controls.

## 2026-05-11 Strict Random-Upstream Local Target

Task:

```text
aiAgentProjectTasks/2026-05-11-phase-6-third-task-Strict-random-upstream-local-target-discovery.md
```

Runner:

```text
scripts/run_phase6_strict_random_upstream_local_target.py
```

Run root:

```text
runs/2026-05-11_phase6_strict_random_upstream_local_target
```

### Gates

The strict branch used:

```text
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
freeze_semantic_decoder=true
freeze_upstream_encoder=true
aux_operand_loss_weight=0.0
input_proj_anchor_weight=0.0
```

Gate A passed on the semantic-decoder-only baseline:

| Metric | Value |
| --- | ---: |
| built-in eval exact | `0.000` |
| oracle-at-eval exact | `1.000` |
| injection-zero exact | `0.000` |
| forced-zero exact | `0.000` gate / `0.0078` canonical |
| forced-random exact | `0.000` gate / `0.0039` canonical |
| learned operand/pair/calc | `0.000 / 0.000 / 0.0234` |
| semantic decoder delta | `0.0` |

Gate B passed on a fixed 128-sample batch under the same
`semantic_decoder_only` initialization:

| Metric | Value |
| --- | ---: |
| hard-best pair equals true pair | `1.000` |
| hard-best A/B targets equal true A/B | `1.000 / 1.000` |
| hard-best local CE | `2.995489` |
| direct aux CE on same logits | `2.995489` |
| local-minus-aux CE | `0.0` |
| effective pairs | `1.078` |
| true-pair probability | `0.988` |
| semantic decoder grad/delta | `0.0 / 0.0` |
| one local step input-proj/upstream delta L2 | `0.000058 / 0.0` |

The local target was selected from full-enum answer NLL, not true operand
labels. True operands were used only for parity reporting and aux-CE comparison.

### Strict Stage 1 Teaching

Branch A used:

```text
calculator_estimator=identifiable_full_enum_local_target
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
freeze_upstream_encoder=true
answer_loss_weight=0.0
local_target_loss_weight=1.0
aux_operand_loss_weight=0.0
input_proj_lr=0.03
steps=300
target_mode=hard_best_pair
```

Branch B was not run because Branch A passed the Stage 1 protocol gate.

| Stage 1 checkpoint | Fast-gate normal/operand/pair/calc | Canonical operand/pair/calc | Private answer/operand/pair/calc | Full-enum learned-true/best gap | Learned-best |
| --- | ---: | ---: | ---: | ---: | ---: |
| first gate step `75` | `0.977 / 0.977 / 0.977 / 0.977` | not run | not run | not run | not run |
| first exact/best step `125` | `1.000 / 1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000 / 1.000` | `0.0 / 0.0` | `1.000` |
| final step `300` | `1.000 / 1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000 / 1.000` | `0.0 / 0.0` | `1.000` |

Stage 1 parameter movement versus its step `0` checkpoint:

| Checkpoint | input-proj L2 / max | upstream L2 / max | semantic decoder L2 / max |
| --- | ---: | ---: | ---: |
| final step `300` | `209.383 / 8.844` | `0.0 / 0.0` | `0.0 / 0.0` |

### Strict Local-Target-Off Retention

Two frozen-upstream retentions were run from the first qualifying and best
qualifying Stage 1 snapshots:

```text
calculator_estimator=adaptive_interface
semantic_decoder_checkpoint_load_scope=full_model
answer_loss_weight=1.0
local_target_loss_weight=0.0
adaptive_interface_loss_weight=0.0
aux_operand_loss_weight=0.0
input_proj_anchor_weight=0.0
freeze_semantic_decoder=true
freeze_upstream_encoder=true
input_proj_lr=0.0003
steps=1000
```

| Stage 2 start | Final fast-gate normal/operand/pair/calc | Canonical operand/pair/calc | Private answer/operand/pair/calc | Full-enum learned-true/best gap | Learned-best |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage 1 first gate step `75` | `1.000 / 1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000 / 1.000` | `0.0 / 0.0` | `1.000` |
| Stage 1 best step `125` | `1.000 / 1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000 / 1.000` | `0.0 / 0.0` | `1.000` |

The first-gate retention branch had a useful intermediate nuance: its selected
step `150` snapshot was canonical-exact, but private was `0.9975` and full-enum
learned-best was `0.984` with learned-minus-true/best gap `0.1065`. Its final
checkpoint closed those gaps to exact.

Retention parameter movement versus each Stage 2 source checkpoint:

| Stage 2 start | input-proj L2 / max | upstream L2 / max | semantic decoder L2 / max |
| --- | ---: | ---: | ---: |
| Stage 1 first gate step `75` | `2.694 / 0.177` | `0.0 / 0.0` | `0.0 / 0.0` |
| Stage 1 best step `125` | `3.277 / 0.220` | `0.0 / 0.0` | `0.0 / 0.0` |

Decision:

- Strong strict-branch positive: the answer-derived hard-best local target
  taught and retained the true calculator-query protocol with only the frozen
  semantic decoder loaded from Stage 0B. The upstream encoder and input
  projection started new/random for Stage 1.
- This is local-target-assisted discovery, not pure answer-only discovery:
  Stage 1 used `local_target_loss_weight=1.0`.
- The retention claim is clean: final local/adaptive target weight `0.0`, aux
  operand supervision `0.0`, input-proj anchor `0.0`, and semantic decoder
  movement `0.0`.
- Compared with the Phase 6 full-model positive, the same matched recipe now
  succeeds without inheriting the oracle-trained upstream representation.

Recommendation:

Proceed to a targeted follow-up rather than rerunning this branch: either an
upstream-open strict variant if the research question requires upstream
movement, or a smaller step toward less local teaching such as decay/handoff
sensitivity from the semantic-decoder-only branch.

## 2026-05-11 Strict Local-Target Decay Boundary

Task:

```text
aiAgentProjectTasks/2026-05-11-phase-6-fourth-task-Strict-local-target-decay-and-minimum-teaching-boundary.md
```

Runner:

```text
scripts/run_phase6_strict_local_target_decay_boundary.py
```

Run root:

```text
runs/2026-05-11_phase6_strict_local_target_decay_boundary
```

### Gates

The new boundary runner reran the strict gates once because the runner path and
command construction changed. Both gates passed under
`semantic_decoder_checkpoint_load_scope=semantic_decoder_only`:

| Gate | Key result |
| --- | --- |
| Oracle wiring | oracle-at-eval `1.000`, injection-zero `0.000`, forced-random `0.000`, semantic decoder delta `0.0` |
| Local-target parity | hard-best pair equals true `1.000`, local CE equals aux CE exactly (`2.995489` / `2.995489`), semantic grad/delta `0.0` |

These remain wiring/parity gates only.

### Single-Stage Decay Ladder

All branches used:

```text
calculator_estimator=identifiable_full_enum_local_target
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
answer_loss_weight=1.0
initial_local_target_loss_weight=1.0
local_target_loss_floor=0.0
aux_operand_loss_weight=0.0
input_proj_anchor_weight=0.0
freeze_upstream_encoder=true
trainable=calculator_hook.input_proj only
input_proj_lr=0.03
steps=300
```

All branches decayed the local target to exactly `0.0` by the final checkpoint,
but none reached retained-protocol quality.

| Decay steps | Final eval | Best fast normal/operand/pair/calc | Canonical operand/pair/calc final | Private operand/pair/calc final | Full-enum learned-true gap | Learned-best |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `50` | `0.234` | `0.414 / 0.414 / 0.414 / 0.422` | `0.234 / 0.234 / 0.242` | `0.2225 / 0.2225 / 0.2325` | `3.349` | `0.180` |
| `75` | `0.271` | `0.492 / 0.492 / 0.492 / 0.500` | `0.309 / 0.309 / 0.320` | `0.2675 / 0.2675 / 0.2725` | `3.419` | `0.289` |
| `100` | `0.215` | `0.461 / 0.461 / 0.461 / 0.469` | `0.238 / 0.238 / 0.258` | `0.2225 / 0.2225 / 0.2375` | `3.790` | `0.234` |
| `150` | `0.592` | `0.594 / 0.594 / 0.594 / 0.594` | `0.543 / 0.543 / 0.543` | `0.585 / 0.585 / 0.585` | `2.814` | `0.516` |

Parameter deltas versus each Stage 1 step `0` checkpoint:

| Decay steps | input-proj L2 | upstream L2 | semantic decoder L2 |
| ---: | ---: | ---: | ---: |
| `50` | `72.128` | `0.0` | `0.0` |
| `75` | `67.461` | `0.0` | `0.0` |
| `100` | `73.023` | `0.0` | `0.0` |
| `150` | `64.050` | `0.0` | `0.0` |

Interpretation:

- The full-enum hard-best target remains sharp and parity-matched, but the
  combined answer/local objective with linear decay did not hand off cleanly.
- The best single-stage decay branch was `150`, but it remained a partial
  protocol, not a retained exact calculator-query protocol.
- This is a schedule/handoff-dynamics negative, not a target-identifiability
  negative.

### Minimum Two-Stage Handoff

Two new answer-only continuations were run from earlier prior strict Stage 1
snapshots. The prior strict task had already shown exact retention from
Stage 1 step `75`.

All new continuations used:

```text
calculator_estimator=adaptive_interface
semantic_decoder_checkpoint_load_scope=full_model
answer_loss_weight=1.0
local_target_loss_weight=0.0
adaptive_interface_loss_weight=0.0
aux_operand_loss_weight=0.0
input_proj_anchor_weight=0.0
freeze_upstream_encoder=true
input_proj_lr=0.0003
steps=1000
```

| Prior Stage 1 start | Source fast operand/pair/calc | Final eval | Best fast normal/operand/pair/calc | Canonical operand/pair/calc selected | Private operand/pair/calc selected | Full-enum learned-true gap | Learned-best |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| step `25` | `0.289 / 0.289 / 0.289` | `0.809` | `0.875 / 0.875 / 0.875 / 0.883` | `0.812 / 0.812 / 0.812` | `0.805 / 0.805 / 0.805` | `1.151` | `0.727` |
| step `50` | `0.398 / 0.398 / 0.398` | `0.848` | `0.922 / 0.922 / 0.922 / 0.922` | `0.863 / 0.863 / 0.863` | `0.845 / 0.845 / 0.845` | `0.702` | `0.844` |
| step `75` | `0.977 / 0.977 / 0.977` | prior exact pass | prior exact pass | `1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000` | `0.0` | `1.000` |

Parameter deltas for the new handoffs:

| Start | input-proj L2 | upstream L2 | semantic decoder L2 |
| ---: | ---: | ---: | ---: |
| step `25` | `4.222` | `0.0` | `0.0` |
| step `50` | `4.227` | `0.0` | `0.0` |

Decision:

- Single-stage strict local-target decay to exactly `0.0` failed for all tested
  decay windows up to `150` steps.
- Answer-only continuation can improve much earlier partial local-target
  checkpoints, but step `25` and step `50` did not become exact retained
  protocols.
- The shortest reliable boundary remains the prior strict Stage 1 step `75`
  handoff: roughly the first fast-gate checkpoint, not the earlier partial
  checkpoints.

Recommendation:

Treat this as a negative for simple linear single-stage decay and a useful
minimum-handoff boundary. Next work should redesign the handoff schedule
rather than rerunning oracle controls: for example hold the local target until
the fast protocol gate is near `0.9`, use a two-phase schedule with automatic
gate-triggered local-target removal, or test a smoother relaxation while
keeping the same strict diagnostics.

## 2026-05-11 Exact Expected Answer-Loss Interface Discovery

Task:

```text
aiAgentProjectTasks/2026-05-11-phase-6-fifth-task-Exact-expected-answer-loss-interface-discovery.md
```

Implementation:

- Added `calculator_estimator=full_enum_expected_answer_loss`.
- The objective enumerates all `20 x 20` action pairs, computes detached
  answer NLL costs, forms the independent model policy
  `p(a,b)=softmax(a_logits/T)*softmax(b_logits/T)`, and minimizes expected
  cost directly.
- Added knobs:
  `--expected-answer-loss-weight`,
  `--expected-answer-loss-policy-temperature`,
  `--expected-answer-loss-cost-normalization none|center|zscore`,
  `--expected-answer-loss-entropy-weight`,
  `--expected-answer-loss-entropy-decay-steps`, and
  `--expected-answer-loss-chunk-size`.
- This is not a local target: it does not construct hard-best or soft CE
  targets from answer losses. True operands are used only for diagnostics.

Run root:

```text
runs/2026-05-11_phase6_expected_answer_loss_interface_discovery
```

### Stage 0 Gradient And Objective Gate

Strict `semantic_decoder_only` gate passed on a fixed 128-sample batch:

| Metric | Value |
| --- | ---: |
| oracle-at-eval exact after one step | `1.000` |
| injection-zero / forced-random exact | `0.000 / 0.000` |
| initial expected answer loss | `8.2412` |
| best / true / learned NLL | `0.0003 / 0.0003 / 8.2883` |
| initial entropy / effective pairs | `5.9915 / 399.998` |
| initial true-pair probability | `0.0025` |
| initial hard learned pair exact | `0.000` |
| one-step input-proj delta L2 | `1.0895` |
| one-step upstream delta L2 | `0.0` |
| semantic decoder grad / delta L2 | `0.0 / 0.0` |

The gate proves the expected-loss objective reaches
`calculator_hook.input_proj` while semantic decoder and upstream parameters
stay fixed. No aux, anchor, oracle, or hard-best local-target construction was
active.

### Stage 1 Frozen-Upstream Ladder

All branches used:

```text
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
freeze_semantic_decoder=true
freeze_upstream_encoder=true
trainable=calculator_hook.input_proj only
answer_loss_weight=0.0
expected_answer_loss_weight=1.0
adaptive/local target weight=0.0
aux_operand_loss_weight=0.0
input_proj_anchor_weight=0.0
input_proj_lr=0.03
steps=300
```

| Branch | Policy temp | Entropy | Final expected NLL | Final entropy / effective pairs | Best lightweight operand/pair/calc | Final eval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | `1.0` | `0.0` | `4.3363` | `0.366 / 1.448` | `0.0156 / 0.0156 / 0.0313` | `0.0059` |
| B | `1.0` | `0.03` decayed | `4.3350` | `0.365 / 1.446` | `0.0156 / 0.0156 / 0.0313` | `0.0059` |
| C | `0.5` | `0.0` | `4.1432` | `0.105 / 1.112` | `0.0156 / 0.0156 / 0.0313` | `0.0059` |

Stage 1 conclusion:

- Expected answer loss decreased substantially and the policy distribution
  collapsed.
- The collapse was to wrong hard actions, not to the true calculator-query
  protocol.
- No branch reached the `>=0.90` fast-gate threshold, so Stage 2 retention was
  correctly skipped.

### Selected Checkpoint Diagnostics

Selected checkpoint:

```text
runs/2026-05-11_phase6_expected_answer_loss_interface_discovery/stage1_branch_c_temp05/2026-05-11_151203_589696_model-c-op0-19-full_enum_expected_answer_loss-inlr0.03-uplr0.0003-expanspolt0.5-expanschunk64-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt
```

| Diagnostic | Key result |
| --- | ---: |
| canonical normal / oracle / injection-zero / forced-random | `0.0039 / 1.000 / 0.0156 / 0.0039` |
| canonical operand / pair / calc | `0.0039 / 0.0039 / 0.0430` |
| private answer / operand / pair / calc | `0.005 / 0.005 / 0.005 / 0.050` |
| full-enum learned / true / best NLL | `4.3075 / 0.0003 / 0.0003` |
| full-enum learned-minus-true / best gap | `4.3072 / 4.3072` |
| full-enum learned-best / true-best | `0.000 / 1.000` |

Parameter deltas versus branch step `0`:

| Branch | input-proj L2 | upstream L2 | semantic decoder L2 |
| --- | ---: | ---: | ---: |
| A | `118.197` | `0.0` | `0.0` |
| B | `118.359` | `0.0` | `0.0` |
| C | `71.452` | `0.0` | `0.0` |

Decision:

- Negative for direct expected answer-loss discovery in the strict frozen
  upstream setup.
- The answer-loss landscape remains sharp (`true_best=1.000`), and the new
  expected-loss objective is wired correctly, but optimizing expected cost over
  the policy can place almost all mass on a wrong hard action.
- This differs from the hard-best local target, which reaches exact protocol
  metrics because it explicitly converts the landscape argmin into CE targets.

Recommendation:

Do not run Stage 2 or broad repeats of this exact independent-head expected
loss. The next best step is a hard-forward/soft-backward Gumbel/Concrete bridge
or an upstream-open expected-loss branch only if the goal is to test whether the
frozen random readout is the blocker. The current failure mode most directly
supports trying a relaxation that aligns training-time mass movement with the
hard argmax protocol.

## 2026-05-12 Relaxed Bridge Replication, Stochastic Gumbel, And Upstream-Open Stress

Task:

```text
aiAgentProjectTasks/2026-05-11-phase-6-seventh-task-Relaxed-bridge-replication-stochastic-and-upstream-open.md
```

Runner and run root:

```text
scripts/run_phase6_relaxed_bridge_replication_stochastic_upstream.py
runs/2026-05-11_phase6_relaxed_bridge_replication_stochastic_upstream
```

All training branches used the strict identifiable setup with
`freeze_semantic_decoder=true`, `oracle_train=false`,
`oracle_warmup_steps=0`, `answer_loss_weight=1.0`,
`aux_operand_loss_weight=0.0`, `adaptive/local_target_loss_weight=0.0`,
`expected_answer_loss_weight=0.0`, and `input_proj_anchor_weight=0.0`.

### Stage 0 Gradient Gates

Both deterministic and literal stochastic Gumbel one-step gates passed with
semantic decoder grad/delta exactly `0.0`.

| Mode | Gate seed | Best-pair prob delta | Gradient cosine | Input-proj delta | Upstream delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| deterministic | `6201` | `+2.0915e-05` | `0.2345` | `1.0896` | `0.0` |
| gumbel | `7201` | `+1.1708e-05` | `0.2743` | `1.0892` | `0.0` |
| gumbel | `7202` | `+2.7953e-05` | `0.1475` | `1.0896` | `0.0` |
| gumbel | `7203` | `+2.1150e-05` | `0.0944` | `1.0892` | `0.0` |

### Stage 1 Deterministic Concrete Replication

All three effective seeds reached the requested fast-gate threshold during
deterministic hard-forward / soft-backward Concrete training.

| Effective seed | First gate step | Best fast normal/operand/pair/calc | Final fast normal/operand/pair/calc | Final eval |
| ---: | ---: | ---: | ---: | ---: |
| `2` | `200` | `1.000 / 1.000 / 1.000 / 1.000` | `0.859 / 0.859 / 0.859 / 0.859` | `0.920` |
| `4` | `250` | `0.961 / 0.961 / 0.961 / 0.961` | `0.844 / 0.844 / 0.844 / 0.844` | `0.766` |
| `5` | `275` | `0.977 / 0.977 / 0.977 / 0.977` | `0.922 / 0.922 / 0.922 / 0.922` | `0.936` |

Selected Stage 1 diagnostics:

| Effective seed | Canonical operand/pair/calc | Private operand/pair/calc | Full-enum learned-true/best gap | Learned-best |
| ---: | ---: | ---: | ---: | ---: |
| `2` | `0.996 / 0.996 / 0.996` | `0.9975 / 0.9975 / 0.9975` | `0.0 / 0.0` | `1.000` |
| `4` | `0.965 / 0.965 / 0.965` | `0.950 / 0.950 / 0.950` | `0.3548 / 0.3548` | `0.9297` |
| `5` | `0.973 / 0.973 / 0.973` | `0.9675 / 0.9675 / 0.9675` | `0.1557 / 0.1557` | `0.9688` |

Nuance: the fast-gate replication was strong, but the stricter selected
Stage 1 full-diagnostic threshold (`>=0.98`) held cleanly only for effective
seed `2`. Seeds `4` and `5` were near-gated at the selected Stage 1 checkpoint.

### Stage 2 Relaxation-Off Retention

All three deterministic seeds retained or completed to exact fast-gate metrics
after switching to `calculator_estimator=adaptive_interface` with all
teacher/local/expected/relaxed objectives inactive.

| Effective seed | Source Stage 1 checkpoint | Final fast normal/operand/pair/calc | Final eval |
| ---: | --- | ---: | ---: |
| `2` | step `200` | `1.000 / 1.000 / 1.000 / 1.000` | `1.000` |
| `4` | step `250` | `1.000 / 1.000 / 1.000 / 1.000` | `1.000` |
| `5` | step `275` | `1.000 / 1.000 / 1.000 / 1.000` | `1.000` |

Selected Stage 2 diagnostics:

| Effective seed | Canonical operand/pair/calc | Private operand/pair/calc | Full-enum learned-true/best gap | Learned-best |
| ---: | ---: | ---: | ---: | ---: |
| `2` | `1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000` | `0.0 / 0.0` | `1.000` |
| `4` | `0.996 / 0.996 / 0.996` | `0.9975 / 0.9975 / 0.9975` | `0.0153 / 0.0153` | `0.9922` |
| `5` | `1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000` | `0.0 / 0.0` | `1.000` |

Final retained objective weights were all clean:

```text
answer_loss_weight=1.0
final_aux_operand_loss_weight=0.0
final_adaptive_interface_loss_weight=0.0
final_local_target_loss_weight=0.0
final_expected_answer_loss_weight=0.0
final_relaxed_calculator_entropy_weight=0.0
final_input_proj_anchor_weight=0.0
```

Interpretation: deterministic Concrete replication is positive after
relaxation-off retention. The answer-only retention phase completed the
near-gated seed `4` and `5` protocols to exact or near-exact diagnostics.

### Stage 3 Literal Stochastic Gumbel

Literal sampled Gumbel-Softmax training failed in both the primary branch and
the one allowed stabilization branch despite the positive stochastic gradient
gate.

| Branch | Effective seed | Best fast normal/operand/pair/calc | Final fast normal/operand/pair/calc | Final eval | Diagnostic learned-best |
| --- | ---: | ---: | ---: | ---: | ---: |
| primary | `2` | `0.000 / 0.023 / 0.023 / 0.055` | `0.000 / 0.008 / 0.008 / 0.008` | `0.000` | `0.0859` |
| stabilized (`1.5 -> 0.5`, entropy `0.01`) | `2` | `0.000 / 0.023 / 0.023 / 0.055` | `0.000 / 0.008 / 0.008 / 0.008` | `0.000` | `0.0234` |

Both stochastic branches reached `NaN` losses after step `225`. This is a
`stochastic_gumbel_negative`, not a target-identifiability negative: oracle
wiring remained `1.000`, and full-enum true-best remained `1.000`.

### Stage 4 Upstream-Open Relaxed Bridge Stress

The deterministic relaxed bridge tolerated carefully opened upstream parameters
for effective seed `2`.

Stage 1 upstream-open training:

| Metric | Value |
| --- | ---: |
| First/best gate step | `225` |
| Best fast normal/operand/pair/calc | `0.961 / 0.961 / 0.961 / 0.961` |
| Final fast normal/operand/pair/calc | `0.664 / 0.664 / 0.664 / 0.664` |
| Final eval | `0.689` |
| Input-proj delta to selected step | `24.6044` L2 / `2.5674` max |
| Upstream delta to selected step | `0.0400` L2 / `0.0030` max |
| Upstream tensors changed | `14 / 29` |
| Semantic decoder delta | `0.0` |

Selected upstream-open Stage 1 diagnostics at step `225`:

```text
canonical operand/pair/calc = 0.973 / 0.973 / 0.973
private operand/pair/calc = 0.9825 / 0.9825 / 0.9825
full-enum learned-minus-true/best gap = 0.0455 / 0.0455
learned-best = 0.9844
```

Relaxation-off retention from the upstream-open selected checkpoint:

| Retention condition | Final fast normal/operand/pair/calc | Canonical operand/pair/calc | Private operand/pair/calc | Full-enum gaps | Learned-best |
| --- | ---: | ---: | ---: | ---: | ---: |
| upstream frozen | `1.000 / 1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000` | `0.0 / 0.0` | `1.000` |
| upstream open | `1.000 / 1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000` | `0.0 / 0.0` | `1.000` |

Interpretation: `upstream_open_positive`. Upstream parameters moved measurably
during relaxed training, semantic decoder movement stayed exactly `0.0`, and
the selected hard protocol survived relaxation-off retention with upstream
frozen and with upstream still open.

### Decision

Keep these labels:

```text
deterministic_concrete_positive
stochastic_gumbel_negative
upstream_open_positive
```

The strongest updated claim is:

```text
In the strict semantic_decoder_only setup, deterministic hard-forward /
soft-backward Concrete answer-loss training replicates across effective seeds
2, 4, and 5 as a fast-gate learner and retains/completes to exact hard
calculator protocols after the relaxation is turned off. Literal stochastic
Gumbel sampling remains unstable and negative under the tested settings. The
deterministic bridge can also tolerate modest upstream movement while keeping
the semantic decoder fixed.
```

## 2026-05-12 Natural Sum-Only Relaxed Bridge Stage 0 Blocker

Task:

```text
aiAgentProjectTasks/2026-05-12-phase-6-eighth-task-Natural-sum-only-relaxed-bridge.md
```

Runner:

```text
scripts/run_phase6_natural_sum_only_relaxed_bridge.py
```

Run root:

```text
runs/2026-05-12_phase6_natural_sum_only_relaxed_bridge
```

Code changes:

- Added the dedicated natural sum-only runner with the fixed `answer_format=sum`
  / `calculator_output_format=sum` setup and result-accuracy checkpoint
  selection logic.
- Extended `scripts/run_full_enum_action_loss_diagnostic.py` with result-aware
  grouping metrics for underidentified sum-only action spaces:
  `learned_result_best_fraction`,
  `mean_learned_result_minus_best_result_gap`,
  `best_result_group_matches_true_sum_fraction`, effective result counts, and
  same-sum near-best pair counts.

Stage 0 using the existing strict sum-only oracle semantic decoder checkpoint:

```text
runs/2026-04-30_175805_513968_model-c-oracle-op0-19-answer_decoder/model-c-2digit-seed2/final_weights.pt
```

| Metric | Value |
| --- | ---: |
| Oracle-at-eval exact | `0.9375` |
| Injection-zero exact | `0.0000` |
| Forced-random exact | `0.0547` |
| Initial hard learned answer exact | `0.0078` |
| Initial hard learned calculator-result accuracy | `0.0078` |
| Full-enum best result group matches true sum | `0.90625` |
| Full-enum true-pair best fraction | `0.078125` |
| Mean same-true-sum near-best pair count | `13.9297` |
| Mean effective action pairs | `30.1275` |
| Mean effective result count | `2.4661` |
| Semantic decoder delta | `0.0` |

This failed the required Stage 0 gate:

```text
oracle-at-eval exact >= 0.98
full-enum best calculator-result matches true sum on nearly all samples
```

Fresh oracle wiring attempts were run under the same sum-only / operand-span /
answer-decoder shape. They also failed to clear the `>=0.98` oracle gate:

| Branch | Run | Eval exact | Diagnostic/oracle exact | Injection-zero | Forced-random |
| --- | --- | ---: | ---: | ---: | ---: |
| 1000 steps, lr `0.003`, batch `64` | `runs/2026-05-12_phase6_natural_sum_only_relaxed_bridge/stage0_fresh_oracle/2026-05-12_084905_549252_model-c-oracle-op0-19-answer_decoder/model-c-2digit-seed2` | `0.9238` | `0.9453 / 0.9531` | `0.0078` | `0.0156` |
| 3000 steps, lr `0.001`, batch `64` | `runs/2026-05-12_phase6_natural_sum_only_relaxed_bridge/stage0_fresh_oracle_lr1e3/2026-05-12_085502_082246_model-c-oracle-op0-19-answer_decoder/model-c-2digit-seed3` | `0.9395` | `0.9297 / 0.9063` | `0.0000` | `0.0156` |
| 2000 steps, lr `0.003`, batch `400` | `runs/2026-05-12_phase6_natural_sum_only_relaxed_bridge/stage0_fresh_oracle_batch400/2026-05-12_090304_977325_model-c-oracle-op0-19-answer_decoder/model-c-2digit-seed4` | `0.9395` | `0.9141 / 0.9375` | `0.0078` | `0.0313` |

Decision:

- Stage 1 deterministic Concrete natural sum-only training was not run because
  the Stage 0 wiring/landscape gate failed.
- This is a useful negative/blocker, not evidence against the relaxed bridge
  itself: the strict natural sum-only semantic decoder did not provide a
  healthy enough oracle-at-eval and full-enum result landscape for interpreting
  learned-interface training.
- Interpretation label: `natural_sum_only_negative` with mechanism
  `sum_only_semantic_decoder_wiring_blocker`.

Recommendation:

Before rerunning the natural sum-only relaxed bridge, strengthen or revise the
sum-only strict answer decoder wiring so oracle-at-eval and full-enum
best-result-match both clear the Stage 0 gate. Do not proceed to `0..99` or
upstream-open natural stress until this smaller wiring gate passes.

## 2026-05-12 Sum-Only Semantic Decoder Gate

Task:

```text
aiAgentProjectTasks/2026-05-12-phase-6-ninth-task-Sum-only-semantic-decoder-gate-and-natural-bridge-readiness.md
```

Runner:

```text
scripts/run_phase6_sum_only_semantic_decoder_gate.py
```

Run root:

```text
runs/2026-05-12_phase6_sum_only_semantic_decoder_gate
```

Code changes:

- Added the dedicated Stage 0 gate runner.
- Added exhaustive all-pair support to the full-enum diagnostic through
  `--exhaustive-grid`.
- Added a test covering exhaustive grid construction.

### Stage 0A Existing Decoder Diagnosis

All-400 diagnosis of the existing April sum-only oracle checkpoint showed a
strict decoder/readout failure:

| Metric | Value |
| --- | ---: |
| Normal learned exact | `0.0300` |
| Oracle-at-eval exact | `0.9300` |
| Forced true result exact | `0.9300` |
| Injection-zero exact | `0.0050` |
| Forced-zero exact | `0.0025` |
| Forced-random exact | `0.0325` |
| Full-enum best-result group matches true sum | `0.9075` |
| Mean same-true-sum near-best pair count | `13.35` |
| Mean effective result count | `2.4820` |
| Semantic decoder delta | `0.0` |

The oracle and forced-true-result rows agreed exactly, so the miss mechanism is
not operand injection shape. Misses were concentrated on specific result
classes: oracle-at-eval missed all prompts with sums `12`, `31`, and `32`;
full-enum best-result matching missed all prompts with sums `23`, `12`, and
`31`. The checkpoint metadata matched the intended natural sum-only setup:
`answer_format=sum`, `calculator_output_format=sum`,
`calculator_read_position=operand_spans`, `calculator_read_span_width=2`, and
`calculator_bottleneck_mode=answer_decoder`.

### Stage 0B Candidate Ladder

No candidate passed the strict natural gate:

```text
oracle-at-eval exact >= 0.98
full-enum best-result group matches true sum >= 0.98
injection-zero and forced-random near chance
semantic decoder delta == 0.0
```

Best checkpoint per branch:

| Branch | Best checkpoint | Oracle-at-eval | Best result=true | Injection-zero | Forced-random |
| --- | --- | ---: | ---: | ---: | ---: |
| tiny `operand_spans` dense | `step_00500_weights.pt` | `0.9325` | `0.9300` | `0.0050` | `0.0200` |
| tiny `operands` dense | `step_00500_weights.pt` | `0.9150` | `0.9300` | `0.0075` | `0.0175` |
| `n_embd=32`, `n_head=2`, `n_layer=2` | `step_01000_weights.pt` | `0.9275` | `0.9150` | `0.0050` | `0.0200` |
| `n_embd=32`, `n_head=2`, `n_layer=3` | `step_01500_weights.pt` | `0.9350` | `0.9250` | `0.0100` | `0.0125` |

Decision:

- Stage 1 natural deterministic relaxed bridge training was not run.
- This is not a natural bridge negative. The semantic decoder/wiring gate is
  still below the threshold needed to interpret answer-only learned calculator
  use.
- Interpretation label: `sum_only_decoder_capacity_blocker`.

Recommendation:

Do not proceed to `operand_max=99`. The next axis should be a sum-only
decoder/readout redesign or a more direct result-class decoder health fix, not
more bridge training from a sub-0.98 natural gate.

## 2026-05-12 Sum-Only Interaction Decoder Gate And Natural Bridge

Task:

```text
aiAgentProjectTasks/2026-05-12-phase-6-tenth-task-Sum-only-answer-decoder-interaction-and-natural-bridge.md
```

Runner:

```text
scripts/run_phase6_sum_only_semantic_decoder_gate.py
```

Run root:

```text
runs/2026-05-12_phase6_sum_only_interaction_decoder_gate
```

Code changes:

- Added `answer_decoder_interaction=none|product` to `GPTConfig`, checkpoint
  configs, run metrics, diagnostics, and CLI training.
- Default sum-only behavior remains additive (`none`).
- New sum-only product mode uses:

```text
decoder_h = selected_signal + offset_h + selected_signal * offset_h
```

- Existing and new `sum_left_operand` behavior remains product-compatible.
- Added regression tests for additive default, product interaction, invalid
  option handling, and loading old checkpoints without the new config field.

### Stage 0 Product Decoder Gate

The previous additive-only existing decoder remained blocked:

| Metric | Value |
| --- | ---: |
| Oracle-at-eval exact | `0.9300` |
| Full-enum best-result group matches true sum | `0.9075` |
| Injection-zero exact | `0.0050` |
| Forced-random exact | `0.0325` |
| Semantic decoder delta | `0.0` |

The fresh tiny product decoder passed the all-400 natural gate without needing
the `n_embd=32,n_head=2` fallback.

Selected checkpoint:

```text
runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/stage0_candidates/tiny_operand_spans_dense/oracle_train/2026-05-12_113346_842566_model-c-oracle-op0-19-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00500_weights.pt
```

| Metric | Value |
| --- | ---: |
| Oracle-at-eval exact | `1.0000` |
| Forced true result / full-enum best-result group true sum | `1.0000` |
| Injection-zero exact | `0.0425` |
| Forced-zero exact | `0.0050` |
| Forced-random exact | `0.0175` |
| Semantic decoder delta | `0.0` |

Interpretation label:

```text
sum_only_interaction_gate_positive
```

This is a readout health positive only, not learned calculator use.

### Stage 1 Natural Deterministic Concrete Bridge

The deterministic Concrete bridge was run for effective seed `2` only, because
the first seed failed far below the result-level replication gate.

Setup highlights:

```text
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
freeze_semantic_decoder=true
freeze_upstream_encoder=true
oracle_train=false
oracle_warmup_steps=0
calculator_estimator=gumbel_concrete_interface
relaxed_calculator_mode=deterministic
relaxed_calculator_hard_forward=true
relaxed_calculator_temperature=2.0
relaxed_calculator_final_temperature=0.5
relaxed_calculator_temperature_decay_steps=300
answer_loss_weight=1.0
aux_operand_loss_weight=0.0
adaptive_interface_loss_weight=0.0
local_target_loss_weight=0.0
expected_answer_loss_weight=0.0
input_proj_anchor_weight=0.0
answer_decoder_interaction=product
```

Best selected Stage 1 snapshot:

| Metric | Value |
| --- | ---: |
| Step | `225` |
| Fast normal answer exact | `0.1350` |
| Fast learned calculator-result accuracy | `0.1350` |
| Final eval exact | `0.126953` |
| Canonical normal / result accuracy | `0.1175 / 0.1175` |
| Full-enum learned-result best fraction | `0.1100` |
| Mean learned-result minus best-result NLL gap | `5.5657` |
| Full-enum best-result group true sum | `1.0000` |
| Oracle-at-eval / injection-zero / forced-random | `1.0000 / 0.0550 / 0.0225` |
| Private result accuracy | `0.1100` |
| Semantic decoder delta | `0.0` |

Final objective weights were clean for the bridge claim:

```text
answer_loss_weight=1.0
final_aux_operand_loss_weight=0.0
final_adaptive_interface_loss_weight=0.0
final_local_target_loss_weight=0.0
final_expected_answer_loss_weight=0.0
final_relaxed_calculator_entropy_weight=0.0
final_input_proj_anchor_weight=0.0
```

Decision:

- Stage 2/3 retention was not run because the seed `2` bridge checkpoint did
  not near-pass the result-level gate.
- Seeds `4` and `5` were not replicated because the task called for replication
  only if seed `2` passed result-level gates.

Interpretation label:

```text
natural_deterministic_concrete_result_negative
```

Recommendation:

The product interaction removes the decoder/readout blocker for natural
sum-only `0..19`, but deterministic Concrete answer-loss training did not
discover a correct calculator-result protocol in the natural underidentified
task. Next work should compare whether the identifiable success relied on the
sharper `sum_left_operand` action landscape or whether natural sum-only needs a
different optimizer/objective for result-level discovery.

## 2026-05-12 Phase 6 Closure Landscape Diagnostic

Task:

```text
aiAgentProjectTasks/2026-05-12-phase-6-eleventh-task-Phase-6-closure-landscape-diagnostic-and-next-phase-decision.md
```

Runner and run root:

```text
scripts/run_phase6_closure_landscape_diagnostic.py
runs/2026-05-12_phase6_closure_landscape_diagnostic
```

No completed training branch was rerun. The closure runner compacted existing
Phase 6 evidence, reran cheap all-400 landscape probes, and ran paired one-step
deterministic hard-forward / soft-backward Concrete gradient checks from strict
`semantic_decoder_only` initializations.

### Existing Evidence Preserved

Identifiable deterministic Concrete:

- Stage 1 selected effective seeds `2`, `4`, and `5` reached fast protocol
  gates of `1.000`, `0.961`, and `0.977` respectively.
- Relaxation-off Stage 2 retention completed all three to final fast
  normal/operand/pair/calculator metrics `1.000 / 1.000 / 1.000 / 1.000`.
- Final retained objective weights were clean:
  `answer_loss_weight=1.0`, aux/adaptive/local/expected/relaxed-entropy/anchor
  weights all `0.0`.
- Semantic decoder delta stayed `0.0`.
- Upstream-open stress selected at `0.961` fast protocol metrics and retained
  to exact protocol metrics with upstream frozen and open.

Natural product-decoder bridge:

- Product decoder gate passed: oracle-at-eval `1.000`, full-enum best-result
  group true sum `1.000`, injection-zero `0.0425`, forced-random `0.0175`,
  semantic decoder delta `0.0`.
- Natural deterministic Concrete selected step `225` reached only `0.135`
  fast result accuracy; canonical result accuracy was `0.1175`.
- Full-enum learned-result best fraction was `0.1100`; learned-result minus
  best-result gap was `5.5657`.
- Final objective weights were clean: no aux operand CE, no hard-best local CE,
  no expected-loss objective, no anchor, and no oracle operands during bridge
  training.

### Paired All-400 Full-Enum Landscape

The closure diagnostic used the Phase 6 local-target temperature `0.25`.

| Setting | Best pair=true | Tie-aware true best | Best result=true | Mean true pair rank | Effective pairs | Effective results | Same-true-sum near-best pairs | True pair soft prob | True result soft prob | Top1/top3/top5 mass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Identifiable `sum_left_operand` | `1.0000` | `1.0000` | `1.0000` | `1.0000` | `1.0839` | `1.0462` | `1.0000` | `0.9879` | `0.9931` | `0.9879 / 0.9962 / 0.9984` |
| Natural sum-only product | `0.0975` | `1.0000` | `1.0000` | `1.0000` | `13.3573` | `1.0011` | `13.3500` | `0.0975` | `0.9999` | `0.0975 / 0.2775 / 0.4375` |

Interpretation:

- The identifiable setting is genuinely pair-identifiable: nearly all soft
  target mass is on the true pair.
- The natural setting is result-identifiable but pair-underidentified: the true
  result group gets essentially all mass, but it is spread across a same-sum
  diagonal with about `13.35` near-best pairs. Independent operand heads then
  have no unique pair target to discover.

### Paired One-Step Relaxed-Gradient Diagnostic

Three strict random-upstream seeds were run for each setting
(`6201`, `6202`, `6203`) with:

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
```

Mean one-step deltas:

| Setting | True pair prob delta | True result prob delta | Best result prob delta | Hard pair exact delta | Hard calc/result delta | Grad cosine vs pair CE | Grad cosine vs result group | Input/upstream/semantic delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Identifiable `sum_left_operand` | `+8.54e-06` | `+2.88e-05` | `+2.88e-05` | `+0.0033` | `+0.0033` | `0.0513` | `0.0567` | `1.0893 / 0.0 / 0.0` |
| Natural sum-only product | `+1.31e-05` | `+7.72e-06` | `+7.72e-06` | `+0.0075` | `+0.0142` | `0.1317` | `0.2188` | `1.0892 / 0.0 / 0.0` |

The one-step natural result-group signal is positive but tiny from the strict
random-upstream initialization. Combined with the many-pair result landscape
and the existing failed 300-step natural bridge, this points to an
underidentification/action-parameterization boundary rather than a decoder
health failure or a broken relaxation implementation.

### Phase 6 Closure Decision

Supported:

- Answer-derived local targets can teach strict calculator protocols in the
  identifiable task.
- Deterministic hard-forward / soft-backward Concrete answer-loss training can
  discover and retain the identifiable hard protocol without direct operand
  labels, hard-best CE, oracle operands during bridge training, or semantic
  decoder movement.
- The deterministic Concrete positive replicates across effective seeds `2`,
  `4`, and `5`, and tolerates modest upstream movement.

Not supported:

- Natural sum-only result-level discovery with the current independent operand
  heads and deterministic Concrete schedule.
- Literal stochastic Gumbel training under tested settings.
- Direct expected answer-loss optimization over independent heads.
- Scaling to larger natural arithmetic before resolving the natural
  underidentification/action-parameterization boundary.

Branches that are now controls rather than live directions:

- Oracle-only decoder/readout gates after they pass.
- Constant-weight hard-best local-target teaching in `sum_left_operand`.
- Simple linear local-target decay/handoff schedules.
- Independent-head exact expected answer-loss sweeps.
- Natural deterministic Concrete repeats with only small schedule tweaks before
  changing the objective or action parameterization.

Final Phase 6 conclusion:

```text
Phase 6 established a real answer-derived interface-discovery positive in the
identifiable `sum_left_operand` setting, culminating in deterministic Concrete
answer-loss discovery and relaxation-off retention. The natural sum-only
failure after the product decoder gate is best explained by result-level
underidentification and independent-head action parameterization, not by oracle
decoder health or broken relaxation wiring.
```

Recommended Phase 7 first task:

```text
Start Phase 7 with a natural `0..19` result-space interface parameterization
or structured joint-pair objective that can place mass on a result group or
canonical query representation, then map to valid calculator calls. Do not
start with `operand_max=99` scaling.
```
