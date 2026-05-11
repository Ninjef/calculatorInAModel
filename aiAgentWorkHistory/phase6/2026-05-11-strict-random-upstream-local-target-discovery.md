# Strict Random-Upstream Local-Target Discovery

## Claim Tested

Can the Phase 6 answer-derived hard-best local target teach and retain the true
calculator-query protocol when only the frozen semantic decoder is loaded from
Stage 0B, leaving upstream and the calculator input projection at the new
strict-branch initialization?

## Code Changes

- Added `scripts/run_phase6_strict_random_upstream_local_target.py`.
- Added subcommands:
  - `oracle-wiring-gate`
  - `compare-local-target-to-aux`
  - `run-stage1`
  - `run-retention`
  - `diagnostics`
  - `summarize`
- Added `--semantic-decoder-checkpoint-load-scope full_model | semantic_decoder_only`.
- Defaulted the strict runner to `semantic_decoder_only` for gates and Stage 1.
- Made the parity gate build the model through the same strict load path used
  by training.
- Kept retention loading selected learned checkpoints with
  `semantic_decoder_checkpoint_load_scope=full_model`.
- Added a focused test that the strict runner command builder threads
  `semantic_decoder_only` into the `overfit_one_batch.py` command.

## Exact Commands

Full command records are in:

```text
runs/2026-05-11_phase6_strict_random_upstream_local_target/commands.jsonl
```

Primary commands:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/run_phase6_strict_random_upstream_local_target.py scripts/overfit_one_batch.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/run_phase6_strict_random_upstream_local_target.py oracle-wiring-gate
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/run_phase6_strict_random_upstream_local_target.py compare-local-target-to-aux --samples 128 --temperature 0.25
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/run_phase6_strict_random_upstream_local_target.py run-stage1 --label semantic_decoder_only_branch_a_frozen_upstream_inlr0.03 --answer-loss-weight 0.0 --local-target-loss-weight 1.0 --input-proj-lr 0.03 --upstream-lr 0.003 --steps 300 --snapshot-every 25 --checkpoint-every 25 --target-mode hard_best_pair --freeze-upstream-encoder
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/run_phase6_strict_random_upstream_local_target.py run-retention --threshold 0.90 --answer-loss-weight 1.0 --input-proj-lr 0.0003 --upstream-lr 0.00003 --steps 1000 --snapshot-every 50 --checkpoint-every 50
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/run_phase6_strict_random_upstream_local_target.py diagnostics
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_phase6_strict_random_upstream_local_target.py summarize
```

The first sandboxed `oracle-wiring-gate` attempt hit the known PyTorch/OpenMP
shared-memory failure (`OMP: Error #179`), so the experimental runner commands
were run outside the sandbox with the same command lines.

## Run Paths

Run root:

```text
runs/2026-05-11_phase6_strict_random_upstream_local_target
```

Stage 0B source checkpoint:

```text
/Users/jarnold/Documents/Codex/2026-05-06/please-work-in-this-repo-users-9/runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt
```

Semantic-decoder-only baseline / oracle wiring gate:

```text
runs/2026-05-11_phase6_strict_random_upstream_local_target/stage0/oracle_wiring_gate_semantic_decoder_only/2026-05-11_082911_737447_model-c-op0-19-adaptive_interface-inlr0.03-uplr0.003-answer_decoder-sum_left_operand/model-c-2digit-seed2
```

Stage 1 Branch A:

```text
runs/2026-05-11_phase6_strict_random_upstream_local_target/stage1/semantic_decoder_only_branch_a_frozen_upstream_inlr0.03/2026-05-11_083140_546417_model-c-op0-19-identifiable_full_enum_local_target-inlr0.03-uplr0.003-fullt0.25-fullchunk64-hard_best_pair-answer_decoder-sum_left_operand/model-c-2digit-seed2
```

Stage 2 retention from first gate step `75`:

```text
runs/2026-05-11_phase6_strict_random_upstream_local_target/stage2/frozen_upstream_retention/semantic_decoder_only_branch_a_frozen_upstream_inlr0.03_first_gate_step00075/2026-05-11_084344_091321_model-c-op0-19-adaptive_interface-inlr0.0003-uplr3e-05-answer_decoder-sum_left_operand/model-c-2digit-seed2
```

Stage 2 retention from best gate step `125`:

```text
runs/2026-05-11_phase6_strict_random_upstream_local_target/stage2/frozen_upstream_retention/semantic_decoder_only_branch_a_frozen_upstream_inlr0.03_best_gate_step00125/2026-05-11_090207_738809_model-c-op0-19-adaptive_interface-inlr0.0003-uplr3e-05-answer_decoder-sum_left_operand/model-c-2digit-seed2
```

## Gate A: Oracle Wiring

| Metric | Value |
| --- | ---: |
| load scope | `semantic_decoder_only` |
| built-in eval exact | `0.000` |
| oracle-at-eval exact | `1.000` |
| injection-zero exact | `0.000` |
| forced-zero exact | `0.000` gate / `0.0078` canonical |
| forced-random exact | `0.000` gate / `0.0039` canonical |
| semantic decoder delta L2 / max | `0.0 / 0.0` |

This is a wiring pass only: learned baseline actions were not useful
(`operand/pair/calc = 0.000 / 0.000 / 0.0234` canonical), while oracle actions
fully recovered the answer path.

## Gate B: Local-Target Parity

| Metric | Value |
| --- | ---: |
| load scope | `semantic_decoder_only` |
| hard-best pair equals true pair | `1.000` |
| hard-best A/B target equals true A/B | `1.000 / 1.000` |
| hard-best local CE | `2.995489` |
| direct aux CE on same logits | `2.995489` |
| local-minus-aux CE | `0.0` |
| effective pairs | `1.078` |
| true-pair probability | `0.988` |
| semantic decoder grad/delta | `0.0 / 0.0` |
| one-step input-proj/upstream delta L2 | `0.000058 / 0.0` |

The hard-best local target was constructed from full-enum answer NLL. True
operands were used only for parity reporting and the aux-CE comparison.

## Stage 1 Strict Teaching

Branch A recipe:

```text
calculator_estimator=identifiable_full_enum_local_target
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
freeze_upstream_encoder=true
trainable=calculator_hook.input_proj only
answer_loss_weight=0.0
local_target_loss_weight=1.0
aux_operand_loss_weight=0.0
input_proj_anchor_weight=0.0
input_proj_lr=0.03
steps=300
target_mode=hard_best_pair
```

Branch B was not run because Branch A passed the Stage 1 protocol gate.

| Checkpoint | Fast-gate normal/operand/pair/calc | Canonical operand/pair/calc | Private answer/operand/pair/calc | Full-enum learned-true/best gap | Learned-best |
| --- | ---: | ---: | ---: | ---: | ---: |
| semantic-only baseline | not applicable | `0.000 / 0.000 / 0.0234` | `0.000 / 0.000 / 0.000 / 0.0125` | `8.325 / 8.325` | `0.000` |
| first gate step `75` | `0.977 / 0.977 / 0.977 / 0.977` | not run | not run | not run | not run |
| first exact/best step `125` | `1.000 / 1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000 / 1.000` | `0.0 / 0.0` | `1.000` |
| final step `300` | `1.000 / 1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000 / 1.000` | `0.0 / 0.0` | `1.000` |

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

Stage 1 parameter movement versus step `0`:

| Checkpoint | input-proj L2 / max | upstream L2 / max | semantic decoder L2 / max |
| --- | ---: | ---: | ---: |
| final | `209.383 / 8.844` | `0.0 / 0.0` | `0.0 / 0.0` |

## Stage 2 Local-Target-Off Retention

Both retention branches used:

```text
calculator_estimator=adaptive_interface
semantic_decoder_checkpoint_load_scope=full_model
freeze_semantic_decoder=true
freeze_upstream_encoder=true
answer_loss_weight=1.0
local_target_loss_weight=0.0
adaptive_interface_loss_weight=0.0
aux_operand_loss_weight=0.0
input_proj_anchor_weight=0.0
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

The first-gate retention branch dipped early (`0.930` fast-gate at step `50`)
and recovered to exact by step `150`. The selected step `150` snapshot was
canonical-exact, but private was `0.9975` and full-enum learned-best was
`0.984` with learned-minus-true/best gap `0.1065`. The final checkpoint closed
those gaps to exact.

Final objective weights for both retention branches:

```text
answer=1.0
local_target=0.0
adaptive_interface=0.0
aux=0.0
input_proj_anchor=0.0
```

Retention parameter movement versus source checkpoint:

| Start checkpoint | input-proj L2 / max | upstream L2 / max | semantic decoder L2 / max |
| --- | ---: | ---: | ---: |
| Stage 1 step `75` | `2.694 / 0.177` | `0.0 / 0.0` | `0.0 / 0.0` |
| Stage 1 step `125` | `3.277 / 0.220` | `0.0 / 0.0` | `0.0 / 0.0` |

## Comparison To Phase 6 Full-Model Positive

The previous Phase 6 matched full-model branch showed that the local target
could replace direct true-operand labels when the Stage 0B full model was
loaded and upstream was frozen.

This run removes the largest interpretive crutch: Stage 1 used
`semantic_decoder_checkpoint_load_scope=semantic_decoder_only`, so the
successful input projection was trained against the semantic decoder without
loading the full Stage 0B upstream representation.

Both branches used the same matched recipe:

```text
freeze_upstream_encoder=true
trainable=calculator_hook.input_proj only
input_proj_lr=0.03
target_mode=hard_best_pair
```

Both reached exact Stage 1 protocol metrics and exact local-target-off retention.

## Decision And Recommendation

Strong strict-branch positive:

```text
With only the frozen semantic decoder loaded, the answer-derived hard-best
local target can teach the true calculator-query protocol from the strict
random/new interface initialization, and answer-only continuation retains that
protocol after the local target is exactly 0.0.
```

This is still local-target-assisted discovery, not pure answer-only discovery.

Recommended next step:

- Do not rerun oracle-only controls for this branch.
- If the next question is whether upstream movement helps or hurts, run a
  narrow upstream-open strict variant.
- If the next question is reducing local teaching, test local-target decay or a
  shorter handoff from the semantic-decoder-only branch.
