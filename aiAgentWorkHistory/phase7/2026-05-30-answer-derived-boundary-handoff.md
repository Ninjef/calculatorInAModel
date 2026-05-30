# 2026-05-30 Answer-Derived Boundary Handoff

## Question

Can the older full-grid result-boundary target source, which derives its target
from answer-loss scoring rather than explicit true-result forced-margin
pressure, transfer into the trusted additive non-bottleneck frozen-policy
handoff gate?

This is a less-prescriptive bridge test, not a scalability claim. The source
still scores all forced result classes, but it does not train the source with a
hard-coded true-result margin objective.

## Source Checkpoint

```text
runs/2026-05-13_phase7_full_grid_upstream_open_result_boundary_retention/stage1_primary_full_grid/2026-05-13_153947_011891_model-c-op0-19-fullgrid-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-hard_best_result-rbtt0.25-rbtchunk64-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00800_weights.pt
```

The original source run used `result_boundary_target_loss_weight=1`,
`result_boundary_target_mode=hard_best_result`, a frozen semantic decoder, and
upstream-open source training. Its original summary reported final exact match
`0.9525`, diagnostic calculator-result accuracy `0.9844`, injection-zero
`0.0703`, and forced-random `0.0156`.

## Handoff Run

```text
runs/2026-05-30_phase7_answer_derived_boundary_handoff/stage1_step800_handoff600_cpu
```

Configuration:

- loaded the source step `800` checkpoint with `compatible_model` scope
- additive non-bottleneck mode
- frozen calculator policy
- frozen semantic decoder
- answer loss weight `1`
- `600` CPU steps, full-grid `operand_max=19`

Command:

```text
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --steps 600 --batch-size 64 --eval-samples 400 --lr 0.003 --answer-loss-weight 1 --operand-max 19 --exhaustive-grid-batch --calculator-operand-vocab-size 20 --calculator-estimator ste --calculator-action-head result_space --calculator-read-position operand_spans --calculator-read-span-width 2 --calculator-injection-mode add --calculator-bottleneck-mode none --calculator-output-format sum --answer-decoder-interaction product --semantic-decoder-checkpoint runs/2026-05-13_phase7_full_grid_upstream_open_result_boundary_retention/stage1_primary_full_grid/2026-05-13_153947_011891_model-c-op0-19-fullgrid-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-hard_best_result-rbtt0.25-rbtchunk64-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00800_weights.pt --semantic-decoder-checkpoint-load-scope compatible_model --freeze-semantic-decoder --freeze-calculator-policy --snapshot-every 100 --snapshot-samples 400 --checkpoint-every 100 --log-every 100 --seed 2 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --run-root runs/2026-05-30_phase7_answer_derived_boundary_handoff/stage1_step800_handoff600_cpu --device cpu
```

## Results

Trusted 600-step frozen-policy handoff:

| Handoff step | Normal | Injection-zero | Oracle |
| ---: | ---: | ---: | ---: |
| `100` | `0.0000` | `0.0000` | `0.0000` |
| `200` | `0.2350` | `0.0000` | `0.2400` |
| `300` | `0.5075` | `0.0000` | `0.5100` |
| `400` | `0.7150` | `0.0000` | `0.7150` |
| `500` | `0.8525` | `0.0000` | `0.8600` |
| `600` | `0.8425` | `0.0000` | `0.8550` |

Final eval was `0.8825`. Diagnostic learned calculator accuracy was `0.9922`,
injection-zero `0.0000`, forced-random `0.0391`, and oracle-at-eval `0.8594`.

## Decision

```text
answer_derived_result_boundary_source_transfers_but_is_not_scalable
```

Interpretation:

- The result-boundary target source policy transfers causally into the additive
  non-bottleneck frozen-policy gate: normal accuracy is high and zero/random
  controls stay low.
- This is evidence that true-result forced-margin pressure is not strictly
  required for staged transfer. An answer-derived best-result target can learn
  a transferable result-level calculator policy.
- It is weaker than automated one-negative forced-margin recovery (`0.9875`
  final / `0.9800` step-600 normal), so it is not the new best staged recipe.
- It is still not the final goal because the source target comes from full
  forced-result enumeration and the non-bottleneck run still freezes a
  pre-trained policy.

Next useful work should use this as a bridge toward less-prescriptive target
construction or estimator work, not rerun the same old Stage 1 result-boundary
checkpoint handoff as novelty.
