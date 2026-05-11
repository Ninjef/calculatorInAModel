# Strict Local-Target Decay Boundary

## Claim Tested

How little answer-derived local teaching is needed in the strict
`semantic_decoder_only` branch before answer loss can retain or complete the
true calculator-query protocol with the local target exactly `0.0`?

This follows the strict random-upstream positive because Phase 6 third already
showed the hard-best local target can teach and retain the protocol without
loading the Stage 0B upstream representation. The remaining crutch tested here
was full-strength, fixed-duration local teaching.

## Code Changes

- Added `scripts/run_phase6_strict_local_target_decay_boundary.py` as a narrow
  successor to the strict random-upstream runner.
- Added `run-decay-ladder` and `run-minimum-handoff` subcommands.
- Threaded `--local-target-decay-steps` to
  `--adaptive-interface-loss-decay-steps`.
- Threaded `--local-target-floor` to `--adaptive-interface-loss-floor`, with
  default `0.0`.
- Included initial local weight, decay steps, floor, and final local weight in
  labels and summaries.
- Kept new Stage 1/decay runs on
  `semantic_decoder_checkpoint_load_scope=semantic_decoder_only`.
- Kept retention/minimum-handoff continuations loading learned checkpoints with
  `semantic_decoder_checkpoint_load_scope=full_model`.
- Added `test_phase6_decay_runner_threads_scope_and_decay_flags`.

## Verification

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/run_phase6_strict_local_target_decay_boundary.py scripts/run_phase6_strict_random_upstream_local_target.py scripts/overfit_one_batch.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
66 passed
```

The first sandboxed runner command hit the known PyTorch/OpenMP shared-memory
failure:

```text
OMP: Error #179: Function Can't open SHM failed
```

Experimental runner commands were therefore run outside the sandbox with
`OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`.

## Run Root

```text
runs/2026-05-11_phase6_strict_local_target_decay_boundary
```

Full command records:

```text
runs/2026-05-11_phase6_strict_local_target_decay_boundary/commands.jsonl
```

Summary artifacts:

```text
runs/2026-05-11_phase6_strict_local_target_decay_boundary/summary.md
runs/2026-05-11_phase6_strict_local_target_decay_boundary/summary.json
```

## Gates

The gates were rerun once under the new run root because the runner and command
construction changed.

| Gate | Result |
| --- | --- |
| Oracle wiring | pass; oracle-at-eval `1.000`, injection-zero `0.000`, forced-random `0.000`, semantic decoder delta `0.0` |
| Local-target parity | pass; hard-best=true `1.000`, local CE `2.995489`, aux CE `2.995489`, local-minus-aux `0.0`, semantic grad/delta `0.0` |

These are wiring/parity gates, not research progress by themselves.

## Decay Ladder

Command shape:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/run_phase6_strict_local_target_decay_boundary.py run-decay-ladder --decay-steps <N> --answer-loss-weight 1.0 --local-target-loss-weight 1.0 --local-target-floor 0.0 --input-proj-lr 0.03 --upstream-lr 0.003 --steps 300 --snapshot-every 25 --checkpoint-every 25 --target-mode hard_best_pair --freeze-upstream-encoder
```

Shared setup:

```text
calculator_estimator=identifiable_full_enum_local_target
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
freeze_semantic_decoder=true
freeze_upstream_encoder=true
answer_loss_weight=1.0
initial_local_target_loss_weight=1.0
local_target_loss_floor=0.0
aux_operand_loss_weight=0.0
input_proj_anchor_weight=0.0
trainable=calculator_hook.input_proj only
```

| Decay steps | Final local weight | Final eval | Best fast normal/operand/pair/calc | Canonical operand/pair/calc final | Private operand/pair/calc final | Full-enum learned-true gap | Learned-best |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `50` | `0.0` | `0.234` | `0.414 / 0.414 / 0.414 / 0.422` | `0.234 / 0.234 / 0.242` | `0.2225 / 0.2225 / 0.2325` | `3.349` | `0.180` |
| `75` | `0.0` | `0.271` | `0.492 / 0.492 / 0.492 / 0.500` | `0.309 / 0.309 / 0.320` | `0.2675 / 0.2675 / 0.2725` | `3.419` | `0.289` |
| `100` | `0.0` | `0.215` | `0.461 / 0.461 / 0.461 / 0.469` | `0.238 / 0.238 / 0.258` | `0.2225 / 0.2225 / 0.2375` | `3.790` | `0.234` |
| `150` | `0.0` | `0.592` | `0.594 / 0.594 / 0.594 / 0.594` | `0.543 / 0.543 / 0.543` | `0.585 / 0.585 / 0.585` | `2.814` | `0.516` |

Parameter movement versus each Stage 1 step `0` checkpoint:

| Decay steps | input-proj L2 | upstream L2 | semantic decoder L2 |
| ---: | ---: | ---: | ---: |
| `50` | `72.128` | `0.0` | `0.0` |
| `75` | `67.461` | `0.0` | `0.0` |
| `100` | `73.023` | `0.0` | `0.0` |
| `150` | `64.050` | `0.0` | `0.0` |

Interpretation:

- All single-stage decay branches failed retained-protocol quality after the
  local target reached exactly `0.0`.
- The conservative `150` branch was the best partial result, but still had a
  large full-enum gap and learned-best only `0.516`.
- Because the parity gate remained exact, this is best interpreted as a
  handoff/schedule-dynamics negative, not a target-identifiability negative.

## Minimum Handoff

New commands:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/run_phase6_strict_local_target_decay_boundary.py run-minimum-handoff --steps-from 25 --answer-loss-weight 1.0 --input-proj-lr 0.0003 --upstream-lr 0.00003 --continuation-steps 1000 --snapshot-every 50 --checkpoint-every 50 --freeze-upstream-encoder
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/run_phase6_strict_local_target_decay_boundary.py run-minimum-handoff --steps-from 50 --answer-loss-weight 1.0 --input-proj-lr 0.0003 --upstream-lr 0.00003 --continuation-steps 1000 --snapshot-every 50 --checkpoint-every 50 --freeze-upstream-encoder
```

Shared continuation setup:

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
trainable=calculator_hook.input_proj only
```

| Prior strict Stage 1 start | Source fast operand/pair/calc | Final eval | Best fast normal/operand/pair/calc | Canonical operand/pair/calc selected | Private operand/pair/calc selected | Full-enum learned-true gap | Learned-best |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| step `25` | `0.289 / 0.289 / 0.289` | `0.809` | `0.875 / 0.875 / 0.875 / 0.883` | `0.812 / 0.812 / 0.812` | `0.805 / 0.805 / 0.805` | `1.151` | `0.727` |
| step `50` | `0.398 / 0.398 / 0.398` | `0.848` | `0.922 / 0.922 / 0.922 / 0.922` | `0.863 / 0.863 / 0.863` | `0.845 / 0.845 / 0.845` | `0.702` | `0.844` |
| step `75` | `0.977 / 0.977 / 0.977` | prior exact pass | prior exact pass | `1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000` | `0.0` | `1.000` |

The step `75` row comes from the prior strict task's already-diagnosed
retention branch:

```text
runs/2026-05-11_phase6_strict_random_upstream_local_target/stage2/frozen_upstream_retention/semantic_decoder_only_branch_a_frozen_upstream_inlr0.03_first_gate_step00075
```

Parameter movement for new handoffs:

| Start | input-proj L2 | upstream L2 | semantic decoder L2 |
| ---: | ---: | ---: | ---: |
| step `25` | `4.222` | `0.0` | `0.0` |
| step `50` | `4.227` | `0.0` | `0.0` |

## Decision

Single-stage linear decay is a negative under this recipe: even `150` decay
steps did not yield retained-protocol quality with the final local target
exactly `0.0`.

The two-stage boundary is more informative. Answer-only continuation can
substantially improve from prior step `25` and step `50`, but it does not
complete them to exact protocol metrics. The shortest reliable handoff remains
prior Stage 1 step `75`, the first fast-gate checkpoint from the full-strength
strict local-target run.

## Recommendation

Do not rerun oracle-only controls. The next useful task should redesign the
handoff schedule: hold the local target until a protocol gate is near `0.9`,
use gate-triggered removal instead of fixed linear decay, or test a smoother
relaxation while keeping the strict semantic-decoder-only setup and dense
diagnostics.
