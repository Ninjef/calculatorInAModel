# Matched Local-Target Teaching And Retention Gate

## Claim Tested

If the Phase 6 hard-best local target is as sharp as measured, the Phase
4-matched frozen-upstream teaching recipe should teach the true calculator-query
protocol without direct true-operand labels, and answer-only continuation should
retain it after the local target is exactly off.

## Code Changes

- Added `scripts/run_phase6_matched_local_target_teaching.py`.
- Added subcommands:
  - `compare-local-target-to-aux`
  - `run-stage1`
  - `run-retention`
  - `diagnostics`
  - `summarize`
- Added configurable runner flags for answer/local-target weights, input-proj
  LR, steps, snapshot/checkpoint cadence, target mode, upstream freezing, and
  checkpoint selection.
- Added a parity diagnostic that compares hard-best local CE to direct aux CE on
  the same logits, verifies hard-best targets match true operands, and checks
  semantic decoder grad/delta remains exactly zero after one local-objective
  step.
- Fixed the runner's compact canonical/private diagnostic summary parsing after
  diagnostics were generated.

## Exact Commands

Full subprocess command records are in:

```text
runs/2026-05-10_phase6_matched_local_target_teaching/commands.jsonl
```

Primary commands:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/run_phase6_matched_local_target_teaching.py scripts/overfit_one_batch.py scripts/run_full_enum_action_loss_diagnostic.py src/model.py
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_phase6_matched_local_target_teaching.py compare-local-target-to-aux --samples 128 --temperature 0.25
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/run_phase6_matched_local_target_teaching.py run-stage1 --label branch_a_frozen_upstream_inlr0.03 --answer-loss-weight 0.0 --local-target-loss-weight 1.0 --input-proj-lr 0.03 --upstream-lr 0.003 --steps 300 --snapshot-every 25 --checkpoint-every 25 --target-mode hard_best_pair --freeze-upstream-encoder
python3 scripts/run_phase6_matched_local_target_teaching.py run-retention --threshold 0.90 --answer-loss-weight 1.0 --input-proj-lr 0.0003 --upstream-lr 0.0003 --steps 1000 --snapshot-every 50 --checkpoint-every 50
python3 scripts/run_phase6_matched_local_target_teaching.py diagnostics
python3 scripts/run_phase6_matched_local_target_teaching.py summarize
```

The Stage 1 and diagnostics commands needed to run outside the sandbox because
OpenMP failed to allocate shared memory inside the sandbox (`OMP: Error #179`).

## Run Paths

Run root:

```text
runs/2026-05-10_phase6_matched_local_target_teaching
```

Stage 0B checkpoint:

```text
/Users/jarnold/Documents/Codex/2026-05-06/please-work-in-this-repo-users-9/runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt
```

Stage 1 Branch A:

```text
runs/2026-05-10_phase6_matched_local_target_teaching/stage1/branch_a_frozen_upstream_inlr0.03/2026-05-10_174427_959957_model-c-op0-19-identifiable_full_enum_local_target-inlr0.03-uplr0.003-fullt0.25-fullchunk64-hard_best_pair-answer_decoder-sum_left_operand/model-c-2digit-seed2
```

Stage 2 retention from first gated Stage 1 snapshot:

```text
runs/2026-05-10_phase6_matched_local_target_teaching/stage2/frozen_upstream_retention/branch_a_frozen_upstream_inlr0.03_first_gate_step00075/2026-05-10_175846_210962_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed2
```

Stage 2 retention from first exact/best Stage 1 snapshot:

```text
runs/2026-05-10_phase6_matched_local_target_teaching/stage2/frozen_upstream_retention/branch_a_frozen_upstream_inlr0.03_best_gate_step00125/2026-05-10_181812_039965_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed2
```

## Parity Gate

| Metric | Value |
| --- | ---: |
| hard-best pair equals true pair | `1.000` |
| hard-best A target equals true A | `1.000` |
| hard-best B target equals true B | `1.000` |
| hard-best local CE | `2.995222` |
| direct aux CE on same logits | `2.995222` |
| local-minus-aux CE | `0.0` |
| target entropy | `0.0727` |
| effective pairs | `1.078` |
| true-pair probability | `0.988` |
| semantic decoder grad L2 | `0.0` |
| semantic decoder delta L2 | `0.0` |

The target was constructed from full-enum answer NLL over all `20 x 20` action
pairs. True operands were not used to select the local target; they were used
only for parity reporting and the aux-CE comparison.

## Stage 1 Results

Branch A recipe:

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

Branch B was not run because Branch A exceeded the retention gate.

| Checkpoint | Fast-gate normal/operand/pair/calc | Canonical operand/pair/calc | Private answer/operand/pair/calc | Full-enum learned-true/best gap | Learned-best |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage 0B baseline | not applicable | `0.0156 / 0.0156 / 0.0352` | `0.0325 / 0.0125 / 0.0125 / 0.0325` | `9.379 / 9.379` | `0.0078` |
| Branch A first gate step `75` | `0.977 / 0.977 / 0.977 / 0.977` | not run | not run | not run | not run |
| Branch A first exact step `125` | `1.000 / 1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000 / 1.000` | `0.0 / 0.0` | `1.000` |
| Branch A final step `300` | `1.000 / 1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000 / 1.000` | `0.0 / 0.0` | `1.000` |

Counterfactuals on Stage 1 selected diagnostics:

| Checkpoint | Normal | Injection-zero | Forced-zero | Forced-random | Oracle-at-eval |
| --- | ---: | ---: | ---: | ---: | ---: |
| step `125` | `1.000` | `0.000` | `0.0039` | `0.0313` | `1.000` |
| final | `1.000` | `0.000` | `0.0039` | `0.0313` | `1.000` |

Final objective weights:

```text
answer=0.0
local_target=1.0
adaptive_interface=1.0
aux=0.0
input_proj_anchor=0.0
```

Parameter movement versus Stage 1 step `0`:

| Checkpoint | input-proj L2 / max | upstream L2 / max | semantic decoder L2 / max |
| --- | ---: | ---: | ---: |
| step `125` | `91.983 / 3.732` | `0.0 / 0.0` | `0.0 / 0.0` |
| final | `209.383 / 8.844` | `0.0 / 0.0` | `0.0 / 0.0` |

## Stage 2 Retention Results

Both retention branches used:

```text
calculator_estimator=adaptive_interface
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

| Start checkpoint | Final fast-gate normal/operand/pair/calc | Canonical operand/pair/calc | Private answer/operand/pair/calc | Full-enum learned-true/best gap | Learned-best |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage 1 step `75` | `1.000 / 1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000 / 1.000` | `0.0 / 0.0` | `1.000` |
| Stage 1 step `125` | `1.000 / 1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000 / 1.000` | `0.0 / 0.0` | `1.000` |

Counterfactuals on Stage 2 final checkpoints:

| Start checkpoint | Normal | Injection-zero | Forced-zero | Forced-random | Oracle-at-eval |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage 1 step `75` final | `1.000` | `0.000` | `0.0039` | `0.0313` | `1.000` |
| Stage 1 step `125` final | `1.000` | `0.000` | `0.0039` | `0.0313` | `1.000` |

The first-gate retention branch had an intermediate nuance: its selected step
`150` checkpoint was canonical-exact, but private operand/pair/calc was
`0.9975 / 0.9975 / 0.9975` and full-enum learned-best was `0.984` with
learned-minus-true/best gap `0.1065`. The final checkpoint closed the gap to
`0.0` and learned-best `1.000`.

Final objective weights for both retention branches:

```text
answer=1.0
local_target=0.0
adaptive_interface=0.0
aux=0.0
input_proj_anchor=0.0
```

Parameter movement versus each Stage 2 source checkpoint:

| Start checkpoint | input-proj L2 / max | upstream L2 / max | semantic decoder L2 / max |
| --- | ---: | ---: | ---: |
| Stage 1 step `75` | `2.694 / 0.177` | `0.0 / 0.0` | `0.0 / 0.0` |
| Stage 1 step `125` | `3.277 / 0.220` | `0.0 / 0.0` | `0.0 / 0.0` |

## Comparison To Phase 4 Direct-Supervision Teaching

Phase 4 showed that direct aux operand supervision with:

```text
answer_loss_weight=0.0
aux_operand_loss_weight=1.0
freeze_upstream_encoder=true
input_proj_lr=0.03
```

could teach and retain the true protocol. This run matched that optimizer shape
but replaced true operand labels with the answer-derived hard-best local target.
It reached exact Stage 1 protocol metrics and retained exact final protocol
metrics after the local target was set to exactly `0.0`.

## Comparison To Phase 6 First Smoke

Phase 6 first smoke had best selected protocol metrics:

```text
frozen upstream canonical operand/pair/calc = 0.566
upstream-open canonical operand/pair/calc = 0.734
```

This matched recipe reached:

```text
Stage 1 canonical operand/pair/calc = 1.000
Stage 2 final canonical operand/pair/calc = 1.000
Stage 2 final full-enum learned-minus-true/best gap = 0.0 / 0.0
```

The first smoke failure was therefore an optimization mismatch, not a target
identifiability failure.

## Decision And Recommendation

Strong positive for the full-model Phase 6 branch:

```text
An answer-derived full-enum hard-best local interface target can replace direct
true-operand supervision for Stage 1 teaching, and answer-only continuation can
retain the learned calculator-query protocol after the local target is exactly
0.0.
```

This is not strict random-upstream discovery. The successful runs used the
Stage 0B full-model load, froze upstream, and trained only
`calculator_hook.input_proj`.

Recommended next step:

- Run the optional upstream-open retention branch from the retained Stage 2
  checkpoint, or move to the stricter `semantic_decoder_only` branch using this
  same parity gate and matched teaching recipe.
- Do not rerun oracle-only controls.
