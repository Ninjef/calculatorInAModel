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
