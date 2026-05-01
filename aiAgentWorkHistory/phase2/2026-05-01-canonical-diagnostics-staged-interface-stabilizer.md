# 2026-05-01 - Canonical diagnostics and staged interface stabilizer

Task: `aiAgentProjectTasks/2026-05-01-phase-2-third-task-Canonical-diagnostics-and-staged-interface-stabilizer.md`

## Diagnostic Contract

- Added `docs/canonical_diagnostics.md`.
- Added `scripts/run_causal_calculator_protocol_diagnostics.py` as a phase-neutral alias for the checkpoint-first causal protocol diagnostic engine.
- Added `scripts/run_action_loss_diagnostic.py` as the phase-neutral action-loss entrypoint.
- Kept backward compatibility for `scripts/run_phase1_track3_causal_diagnostics.py`, `scripts/run_phase1_track4_action_loss_diagnostic.py`, and `scripts/run_track4_action_loss_diagnostic.py`.
- Updated phase-2 fact-sheet language to prefer canonical causal protocol and canonical action-loss diagnostic names while still noting legacy Track 3/Track 4 labels.

Canonical meanings:

- Causal protocol diagnostic: causal dependence and bottleneck classification, plus forced-result sweep when requested.
- Action-loss diagnostic: learned-vs-true-vs-random/shuffled action NLL comparison.
- These are not unit tests, not phase-1-only task files, and not proof of success unless paired with calculator-dependent answer accuracy and learned-action improvement.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/diagnose_calculator_protocol.py scripts/run_causal_calculator_protocol_diagnostics.py scripts/run_action_loss_diagnostic.py scripts/run_track4_action_loss_diagnostic.py scripts/run_phase1_track3_causal_diagnostics.py scripts/run_phase1_track4_action_loss_diagnostic.py scripts/overfit_one_batch.py
```

## Shared Stabilizer Regime

Semantic decoder checkpoint:

```text
runs/2026-04-30_175805_513968_model-c-oracle-op0-19-answer_decoder/model-c-2digit-seed2/final_weights.pt
```

All stabilizer runs used:

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
calculator_estimator=adaptive_interface
freeze_semantic_decoder=true
input_proj_lr=3e-4
upstream_lr=3e-4
adaptive_interface_target_mode=hard_pair
steps=1000
batch_size=64
eval_samples=512
seed=0
```

Base command:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --operand-max 19 --calculator-operand-vocab-size 20 --steps 1000 --batch-size 64 --eval-samples 512 --seed 0 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --calculator-read-position operands --calculator-bottleneck-mode answer_decoder --calculator-estimator adaptive_interface --semantic-decoder-checkpoint runs/2026-04-30_175805_513968_model-c-oracle-op0-19-answer_decoder/model-c-2digit-seed2/final_weights.pt --freeze-semantic-decoder --input-proj-lr 3e-4 --upstream-lr 3e-4 --adaptive-interface-target-mode hard_pair --snapshot-every 200 --snapshot-samples 64 --log-every 50
```

Run-specific modifiers and outputs:

| Variant | Modifiers | Output |
| --- | --- | --- |
| Aux `0.01`, decay `250` | `--aux-operand-loss-weight 0.01 --aux-operand-loss-decay-steps 250 --aux-operand-loss-floor 0.0` | `runs/2026-05-01_083538_751519_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-aux0.01-auxdecay250/model-c-2digit-seed2` |
| Aux `0.01`, decay `500` | `--aux-operand-loss-weight 0.01 --aux-operand-loss-decay-steps 500 --aux-operand-loss-floor 0.0` | `runs/2026-05-01_084030_058179_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-aux0.01-auxdecay500/model-c-2digit-seed2` |
| Aux `0.03`, decay `250` | `--aux-operand-loss-weight 0.03 --aux-operand-loss-decay-steps 250 --aux-operand-loss-floor 0.0` | `runs/2026-05-01_084523_733373_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-aux0.03-auxdecay250/model-c-2digit-seed2` |
| Aux `0.03`, decay `500` | `--aux-operand-loss-weight 0.03 --aux-operand-loss-decay-steps 500 --aux-operand-loss-floor 0.0` | `runs/2026-05-01_085013_079390_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-aux0.03-auxdecay500/model-c-2digit-seed2` |

All final evaluations happened after the aux weight decayed to `0.0`.

Canonical causal diagnostic command template:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. python3 scripts/run_causal_calculator_protocol_diagnostics.py --checkpoint <run>/final_weights.pt --digits 2 --operand-max 19 --samples 64 --forced-result-sweep --forced-result-batch-size 64 --leakage-control-exact-match 0.004 --output-dir <run>/track3_diagnostics
```

Canonical action-loss command:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. python3 scripts/run_action_loss_diagnostic.py --checkpoint runs/2026-05-01_083538_751519_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-aux0.01-auxdecay250/model-c-2digit-seed2/final_weights.pt runs/2026-05-01_084030_058179_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-aux0.01-auxdecay500/model-c-2digit-seed2/final_weights.pt runs/2026-05-01_084523_733373_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-aux0.03-auxdecay250/model-c-2digit-seed2/final_weights.pt runs/2026-05-01_085013_079390_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-aux0.03-auxdecay500/model-c-2digit-seed2/final_weights.pt --samples 64 --random-actions 16 --digits 2 --operand-max 19 --no-work-history
```

## Results

Comparison against the failed adaptive baseline and lower-LR stabilizer:

| Run | Eval exact | Diag exact | Operand exact | Calc result acc | Target acc | Learned-target agreement | Final CE | Final aux loss | Aux weight | Causal diagnostic | Action learned-true gap | Learned best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Failed baseline | `0.0098` | `0.0156` | `0.0000` | `0.0156` | `0.9375` | `0.0156` | `193.0297` | `0.0000` | `0.0000` | `calculator_ignored_or_bypassed` | `14.3265` | `0.0000` |
| Lower LR | `0.0664` | `0.0469` | `0.0156` | `0.0469` | `0.9375` | `0.0703` | `2.2887` | `0.0000` | `0.0000` | `causally_useful_opaque_private_code` | `10.4115` | `0.0000` |
| Aux `0.01`, decay `250` | `0.0078` | `0.0156` | `0.0078` | `0.0156` | `0.9375` | `0.0000` | `2.3514` | `5.7272` | `0.0000` | `calculator_ignored_or_bypassed` | `8.7436` | `0.0000` |
| Aux `0.01`, decay `500` | `0.0176` | `0.0312` | `0.0000` | `0.0312` | `0.9375` | `0.0312` | `2.3530` | `5.7951` | `0.0000` | `calculator_ignored_or_bypassed` | `10.3902` | `0.0000` |
| Aux `0.03`, decay `250` | `0.0156` | `0.0156` | `0.0078` | `0.0156` | `0.9375` | `0.0000` | `2.3501` | `6.0152` | `0.0000` | `calculator_harmful` | `10.5551` | `0.0000` |
| Aux `0.03`, decay `500` | `0.0391` | `0.0469` | `0.0156` | `0.0469` | `0.9375` | `0.0547` | `2.2918` | `5.7369` | `0.0000` | `calculator_ignored_or_bypassed` | `9.8871` | `0.0000` |

Counterfactual table for aux runs:

| Run | Injection-zero | Forced-zero | Forced-random | Oracle-at-eval |
| --- | ---: | ---: | ---: | ---: |
| Aux `0.01`, decay `250` | `0.0000` | `0.0156` | `0.0625` | `0.9062` |
| Aux `0.01`, decay `500` | `0.0000` | `0.0156` | `0.0625` | `0.9062` |
| Aux `0.03`, decay `250` | `0.0000` | `0.0156` | `0.0625` | `0.9062` |
| Aux `0.03`, decay `500` | `0.0000` | `0.0156` | `0.0625` | `0.9062` |

Causal diagnostic summary:

| Run | Classification | Bottleneck label | Learned best forced class | True-sum best forced class |
| --- | --- | --- | ---: | ---: |
| Aux `0.01`, decay `250` | `calculator_ignored_or_bypassed` | `strict_bottleneck_unvalidated` | `0.0156` | `0.9219` |
| Aux `0.01`, decay `500` | `calculator_ignored_or_bypassed` | `strict_bottleneck_unvalidated` | `0.0469` | `0.9219` |
| Aux `0.03`, decay `250` | `calculator_harmful` | `strict_bottleneck_unvalidated` | `0.0000` | `0.9219` |
| Aux `0.03`, decay `500` | `calculator_ignored_or_bypassed` | `strict_bottleneck_unvalidated` | `0.0469` | `0.9219` |

Action-loss summary:

| Run | True NLL | Random NLL | Shuffled NLL | Learned NLL | Learned-true gap | Learned best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Aux `0.01`, decay `250` | `0.1181` | `9.9763` | `10.4317` | `8.8617` | `8.7436` | `0.0000` |
| Aux `0.01`, decay `500` | `0.1181` | `9.9763` | `10.4317` | `10.5083` | `10.3902` | `0.0000` |
| Aux `0.03`, decay `250` | `0.1181` | `9.9763` | `10.4317` | `10.6731` | `10.5551` | `0.0000` |
| Aux `0.03`, decay `500` | `0.1181` | `9.9763` | `10.4317` | `10.0052` | `9.8871` | `0.0000` |

Parameter movement versus the semantic decoder checkpoint:

| Run | `input_proj.weight` L2/max | `input_proj.bias` L2/max | Frozen semantic decoder deltas |
| --- | --- | --- | --- |
| Aux `0.01`, decay `250` | `1.5889 / 0.1235` | `0.3999 / 0.1112` | `0.0000 / 0.0000` |
| Aux `0.01`, decay `500` | `1.5881 / 0.1146` | `0.4011 / 0.1035` | `0.0000 / 0.0000` |
| Aux `0.03`, decay `250` | `1.7365 / 0.1255` | `0.4272 / 0.1137` | `0.0000 / 0.0000` |
| Aux `0.03`, decay `500` | `1.6498 / 0.1026` | `0.4096 / 0.0916` | `0.0000 / 0.0000` |

## Interpretation

- Aux `0.01` decay `250` improved the action-loss learned-minus-true gap versus the lower-LR baseline (`8.7436` vs `10.4115`), but it did not improve answer accuracy, calculator result accuracy, learned-target agreement, or learned-best fraction.
- Aux `0.03` decay `500` matched the lower-LR baseline on diagnostic exact and calculator result accuracy (`0.0469`) but did not beat it, and learned-best remained `0.0`.
- All aux runs preserved the downstream positive controls: oracle-at-eval recovered `0.9062`, and the forced true-sum class was best on `0.9219` of prompts.
- Direct operand bootstrapping did not survive removal of the aux loss as a stable learned calculator interface.

Go/no-go recommendation: no-go on adaptive-interface success. This is a useful negative result: small decaying true-operand supervision can slightly change collapse shape and sometimes improve one action-loss gap, but the strict-bottleneck learned actions still do not become correct or competitive after supervision is gone.
