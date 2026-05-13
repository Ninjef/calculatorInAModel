# Phase 7 Joint-Pair Result-Group Bridge Gate

## Task

```text
aiAgentProjectTasks/completed/phase7/2026-05-12-phase-7-first-task-Natural-joint-pair-result-group-bridge-gate.md
```

## Claim

Natural sum-only answer loss identifies a calculator result group, not a unique
operand pair. The smallest Phase 7 implementation gate is therefore a joint
pair policy that uses a hard calculator pair in the forward pass and routes
soft backward mass through same-result groups.

## Code Changes

- `src/model.py`
  - Allowed `calculator_read_position=operand_spans` with
    `calculator_action_head=joint_pair`.
  - Changed joint `pair_proj` input width to
    `2 * calculator_read_span_width * n_embd` for span reads.
  - Added joint-pair deterministic/Gumbel Concrete hard-forward /
    soft-backward result-group signal.
  - Preserved hard argmax trace fields: `pair_pred`, `a_pred`, `b_pred`,
    `result_pred`, pair confidence, and pair entropy.
- `scripts/overfit_one_batch.py`
  - Allowed `gumbel_concrete_interface` with `joint_pair`.
  - Updated joint pair logit reads and aux loss to support operand spans.
  - Added joint-pair relaxed policy metrics, including result entropy and
    effective result count.
- `tests/test_model.py`
  - Added direct joint-pair result-group soft-backward gradient coverage.
  - Added full-model `operand_spans + joint_pair` projection-shape and frozen
    non-interface gradient coverage.

## Commands

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests -q
```

One-step CLI smoke:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --steps 1 --batch-size 128 --eval-samples 128 --operand-max 19 --calculator-operand-vocab-size 20 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --answer-format sum --calculator-output-format sum --calculator-bottleneck-mode answer_decoder --answer-decoder-interaction product --calculator-estimator gumbel_concrete_interface --calculator-action-head joint_pair --calculator-read-position operand_spans --calculator-read-span-width 2 --semantic-decoder-checkpoint runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/stage0_candidates/tiny_operand_spans_dense/oracle_train/2026-05-12_113346_842566_model-c-oracle-op0-19-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00500_weights.pt --semantic-decoder-checkpoint-load-scope semantic_decoder_only --freeze-semantic-decoder --freeze-upstream-encoder --answer-loss-weight 1.0 --aux-operand-loss-weight 0.0 --adaptive-interface-loss-weight 0.0 --expected-answer-loss-weight 0.0 --relaxed-calculator-temperature 2.0 --relaxed-calculator-final-temperature 2.0 --relaxed-calculator-temperature-decay-steps 0 --relaxed-calculator-mode deterministic --relaxed-calculator-hard-forward --relaxed-calculator-entropy-weight 0.0 --input-proj-anchor-weight 0.0 --input-proj-lr 0.03 --upstream-lr 0.0003 --snapshot-every 1 --snapshot-samples 128 --checkpoint-every 1 --log-every 1 --run-root runs/2026-05-12_phase7_joint_pair_result_group_bridge_gate/stage0_cli_smoke
```

## Run Path

```text
runs/2026-05-12_phase7_joint_pair_result_group_bridge_gate/stage0_cli_smoke/2026-05-12_184116_723657_model-c-op0-19-gumbel_concrete_interface-joint_pair-inlr0.03-uplr0.0003-rtemp2-rfinal2-answer_decoder-adec-product/model-c-2digit-seed2
```

## Results

Validation:

```text
tests/test_model.py: 74 passed
tests/: 83 passed
```

One-step smoke:

| Step | Answer loss | Hard pair exact | Hard result accuracy | Pair entropy | Effective pairs | Result entropy | Effective results |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `8.3140` | `0.0000` | `0.0078` | `5.9915` | `399.9993` | `3.4932` | `32.8908` |
| `1` | `7.2661` | `0.0000` | `0.0234` | `5.9912` | `399.8765` | `3.4913` | `32.8277` |

Gradient gate:

| Metric | Value |
| --- | ---: |
| answer loss | `7.6510` |
| pair-proj gradient L2 | `0.04198` |
| pair-proj one-step delta L2 | `4.8305` |
| input-proj gradient L2 | `0.0` |
| semantic decoder gradient L2 | `0.0` |
| upstream gradient L2 | `0.0` |
| semantic output-proj delta L2 | `0.0` |
| `pair_proj.weight` shape | `[400, 64]` |
| initial hard result accuracy | `0.0234` |

## Interpretation

The implementation gate passed. Answer loss reaches `calculator_hook.pair_proj`
through the result-group relaxation while the semantic decoder and upstream
stay frozen. This is not yet a natural learned-result success; it is the
minimal wiring and gradient gate needed before the Stage 1 product-decoder /
full-enum landscape regression and any seed-2 bridge training.

## Recommendation

Proceed to the natural decoder and full-enum landscape regression gate. Launch
the seed-2 strict joint-pair bridge only after that gate passes, and select
checkpoints by hard learned calculator-result metrics rather than pair exact.
