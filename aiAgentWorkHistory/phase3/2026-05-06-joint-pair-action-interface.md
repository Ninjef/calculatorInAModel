# 2026-05-06 - Joint pair-action interface smoke

Task: implement and smoke-test Track A's joint pair-action interface under the strict Phase 2 answer-decoder bottleneck.

## Code changes

- Added `calculator_action_head=independent_operands|joint_pair` to `GPTConfig`.
- Added a direct `joint_pair` calculator head: concatenate final A/B read residuals and project to `operand_vocab_size * operand_vocab_size` logits.
- Kept the independent A/B path unchanged for existing estimators and checkpoints.
- Added `calculator_estimator=action_loss_full_enum_joint_interface`.
- Added pair-level full-enum target training:
  - enumerate all `20 x 20` action pairs;
  - score each pair with frozen answer-decoder NLL;
  - convert losses to `softmax(-nll / temperature)`;
  - train joint pair logits with pair-level CE/KL, without A/B marginalization.
- Added trace and diagnostic fields for `pair_pred`, `pair_confidence`, `pair_entropy`, `pair_logp`, pair exact, result-equivalent pair accuracy, true-pair probability/rank, learned/true/best pair NLL, and tie-aware learned-best fractions.
- Extended `scripts/diagnose_calculator_protocol.py`, `scripts/run_action_loss_diagnostic.py`, and `scripts/run_full_enum_action_loss_diagnostic.py` to report joint-head metrics.
- Added tests for joint-head trace behavior and pair-projection-only gradients.

## Validation

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/run_full_enum_action_loss_diagnostic.py scripts/run_phase1_track4_action_loss_diagnostic.py scripts/diagnose_private_protocol.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. pytest -q tests/test_model.py -k "joint or full_enum"
PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. pytest -q
```

Result: `57 passed`.

## CLI compatibility smoke

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --operand-max 19 --calculator-operand-vocab-size 20 --steps 1 --batch-size 4 --eval-samples 4 --seed 90 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --calculator-read-position operands --calculator-bottleneck-mode answer_decoder --calculator-estimator action_loss_full_enum_joint_interface --calculator-action-head joint_pair --semantic-decoder-checkpoint runs/2026-05-03_114747_070474_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed4/checkpoint_snapshots/step_00550_weights.pt --freeze-semantic-decoder --freeze-upstream-encoder --answer-loss-weight 1.0 --aux-operand-loss-weight 0.0 --input-proj-anchor-weight 0.0 --input-proj-lr 0.0003 --upstream-lr 0.0003 --action-loss-full-enum-temperature 1.0 --action-loss-full-enum-chunk-size 64 --snapshot-every 1 --checkpoint-every 1 --snapshot-samples 4 --log-every 1 --run-root /private/tmp/calculator_joint_cli_smoke
```

Result: completed and saved `/private/tmp/calculator_joint_cli_smoke/2026-05-06_082027_195985_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.0003-uplr0.0003-fullt1-fullchunk64-answer_decoder/model-c-2digit-seed92`.

## Joint smoke run

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --operand-max 19 --calculator-operand-vocab-size 20 --steps 200 --batch-size 64 --eval-samples 512 --seed 101 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --calculator-read-position operands --calculator-bottleneck-mode answer_decoder --calculator-estimator action_loss_full_enum_joint_interface --calculator-action-head joint_pair --semantic-decoder-checkpoint runs/2026-05-03_114747_070474_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed4/checkpoint_snapshots/step_00550_weights.pt --freeze-semantic-decoder --freeze-upstream-encoder --answer-loss-weight 1.0 --aux-operand-loss-weight 0.0 --input-proj-anchor-weight 0.0 --input-proj-lr 0.0003 --upstream-lr 0.0003 --action-loss-full-enum-temperature 1.0 --action-loss-full-enum-chunk-size 64 --snapshot-every 50 --checkpoint-every 50 --snapshot-samples 128 --log-every 50
```

Run path:

```text
runs/2026-05-06_082124_888126_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.0003-uplr0.0003-fullt1-fullchunk64-answer_decoder/model-c-2digit-seed103
```

Start checkpoint:

```text
runs/2026-05-03_114747_070474_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed4/checkpoint_snapshots/step_00550_weights.pt
```

## Smoke snapshots

| Step | Normal exact | Injection-zero | Oracle-at-eval | Pair exact | Calc result acc | Mean pair entropy | Learned results |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.03125 | 0.00000 | 0.89063 | 0.00000 | 0.03125 | 5.99146 | varied |
| 50 | 0.03125 | 0.00000 | 0.96875 | 0.00000 | 0.02344 | 5.99142 | mostly 12/21 |
| 100 | 0.04688 | 0.00000 | 0.92969 | 0.00000 | 0.01563 | 5.99130 | mostly 12/21 |
| 150 | 0.03906 | 0.00000 | 0.91406 | 0.00000 | 0.02344 | 5.99112 | mostly 12/21 |
| 200 | 0.06250 | 0.00000 | 0.92188 | 0.00000 | 0.05469 | 5.99086 | mostly 12/21 |

Final built-in eval exact: `25/512 = 0.04883`.

Proof constraints:

- `final_aux_operand_loss_weight=0.0`.
- `final_input_proj_anchor_weight=0.0`.
- `freeze_upstream_encoder=true`.
- Trainable parameter groups limited to `calculator_hook.pair_proj` (`13,200` params).

## Post-smoke diagnostics

Commands:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. python3 scripts/run_causal_calculator_protocol_diagnostics.py --checkpoint runs/2026-05-06_082124_888126_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.0003-uplr0.0003-fullt1-fullchunk64-answer_decoder/model-c-2digit-seed103/final_weights.pt --digits 2 --operand-max 19 --samples 64 --forced-result-sweep --forced-result-batch-size 64 --leakage-control-exact-match 0.004 --output-dir runs/2026-05-06_082124_888126_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.0003-uplr0.0003-fullt1-fullchunk64-answer_decoder/model-c-2digit-seed103/canonical_causal_diagnostics
PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. python3 scripts/run_action_loss_diagnostic.py --checkpoint runs/2026-05-06_082124_888126_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.0003-uplr0.0003-fullt1-fullchunk64-answer_decoder/model-c-2digit-seed103/final_weights.pt --samples 64 --random-actions 16 --digits 2 --operand-max 19 --no-work-history
PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. python3 scripts/run_full_enum_action_loss_diagnostic.py --checkpoint runs/2026-05-06_082124_888126_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.0003-uplr0.0003-fullt1-fullchunk64-answer_decoder/model-c-2digit-seed103/final_weights.pt --samples 64 --batch-size 32 --digits 2 --operand-max 19 --chunk-size 64
PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. python3 scripts/diagnose_private_protocol.py --checkpoint runs/2026-05-06_082124_888126_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.0003-uplr0.0003-fullt1-fullchunk64-answer_decoder/model-c-2digit-seed103/final_weights.pt --digits 2 --operand-max 19 --output-dir runs/2026-05-06_082124_888126_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.0003-uplr0.0003-fullt1-fullchunk64-answer_decoder/model-c-2digit-seed103/private_protocol_diagnostics
```

Canonical causal:

| Metric | Value |
| --- | ---: |
| Normal exact | 0.04688 |
| Injection-zero exact | 0.00000 |
| Forced-random exact | 0.06250 |
| Oracle-at-eval exact | 0.90625 |
| Pair exact | 0.00000 |
| Result-equivalent pair accuracy | 0.04688 |
| Classification | `calculator_ignored_or_bypassed` |
| Bottleneck label | `strict_bottleneck_unvalidated` |

Action-loss diagnostic:

| Metric | Value |
| --- | ---: |
| Learned NLL | 6.98358 |
| True NLL | 0.11809 |
| Random NLL | 9.97635 |
| Learned-minus-true gap | 6.86549 |
| True-best fraction | 0.95313 |
| Learned-best fraction | 0.00000 |
| Operand exact | 0.00000 |
| Calculator-result accuracy | 0.01563 |

Full-enum joint diagnostic:

| Metric | Value |
| --- | ---: |
| Best NLL | 0.09967 |
| Learned NLL | 5.72066 |
| True NLL | 0.09967 |
| Learned-minus-true gap | 5.62099 |
| Learned-minus-best gap | 5.62100 |
| Learned-best fraction | 0.00000 |
| Tie-aware learned-best <= 1e-3 | 0.10938 |
| Tie-aware learned-best <= 1e-2 | 0.10938 |
| Best result matches true sum | 0.90625 |
| Learned result matches true sum | 0.03125 |
| Teacher effective pairs | 29.21159 |
| Pair-logit effective pairs | 399.75911 |
| True pair probability | 0.06690 |
| True pair rank | 2.12500 |

Private protocol all-pair:

| Metric | Value |
| --- | ---: |
| All-pair answer exact | 0.04500 |
| Operand exact | 0.00000 |
| Pair exact | 0.00000 |
| Calculator result accuracy | 0.02750 |
| Result-equivalent pair accuracy | 0.02750 |
| Learned result distribution | `{"12": 316, "21": 84}` |
| Majority-mapped operand exact | 0.02250 |
| Majority-mapped calc result accuracy | 0.05000 |

## Gate decision

No-go for the primary selected-checkpoint ladder and Stage-B comparison in this implementation.

The smoke preserved the important controls: injection-zero stayed near zero, oracle-at-eval stayed high, aux/anchor were exactly zero, upstream was frozen, and only the joint interface head trained. But the joint head did not produce a useful nontrivial action distribution. Pair logits stayed essentially uniform (`effective_pairs ~400`) while argmax actions collapsed to a couple of result classes, pair exact stayed `0.0`, result-equivalent pair accuracy stayed near chance, and canonical causal diagnostics labeled the learned calculator ignored/bypassed.

Interpretation: the full-enum teacher is informative at the answer-decoder level, but this direct joint head did not optimize from the selected start under the current temperature/LR/smoke regime. The failure mode is optimization/collapse of the learned joint head, not absence of teacher signal.

## Recommendation

Do not run the 3x500 selected-checkpoint ladder from this smoke result. Create the next Phase 3 task for Track B: test an identifiability curriculum where operand identity is rewarded by task structure rather than addition-only answer loss.
