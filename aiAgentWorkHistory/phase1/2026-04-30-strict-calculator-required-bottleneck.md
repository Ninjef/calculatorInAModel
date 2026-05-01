# 2026-04-30 - Strict calculator-required bottleneck

Task: implement and validate a stricter calculator-required setting where answer-token success must flow through calculator output rather than ordinary autoregressive residual context.

## Implementation

- Added `GPTConfig.calculator_bottleneck_mode` with `none` and `answer_decoder`.
- `answer_decoder` replaces logits at and after the first `=` with a small decoder that sees only the selected calculator injection plus answer-offset metadata.
- Preserved existing `calculator_injection_mode=add|replace`; strict bottlenecking is controlled by the separate `--calculator-bottleneck-mode answer_decoder` flag.
- Threaded the flag through training configs, checkpoint configs, metrics, diagnostics, and Track 4 explicit-checkpoint runs.
- Updated Track 3 classification so validated strict controls can receive `calculator_required_bottleneck`.

## Validation Config

All strict runs used:

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
seed=0
run seed=2
steps=1000
eval_samples=512
```

## Runs

| Run | Exact | Operand | Calc result | Key path |
| --- | ---: | ---: | ---: | --- |
| Model A strict/off | 0.004 | n/a | n/a | `runs/2026-04-30_175605_833720_model-a-op0-19-answer_decoder/model-a-2digit-seed2` |
| Model B strict/off | 0.004 | 0.000 | 0.000 | `runs/2026-04-30_175704_122932_model-b-op0-19-answer_decoder/model-b-2digit-seed2` |
| Oracle Model C strict | 0.924 | 1.000 | 1.000 | `runs/2026-04-30_175805_513968_model-c-oracle-op0-19-answer_decoder/model-c-2digit-seed2` |
| Learned STE Model C strict | 0.037 | 0.000 | 0.000 | `runs/2026-04-30_180050_609896_model-c-op0-19-answer_decoder/model-c-2digit-seed2` |

Oracle Model C built-in 128-sample counterfactuals:

| Condition | Exact |
| --- | ---: |
| normal | 0.945 |
| injection-zero | 0.008 |
| forced-zero | 0.000 |
| forced-random | 0.016 |
| oracle-at-eval | 0.953 |

Learned STE built-in 128-sample counterfactuals:

| Condition | Exact |
| --- | ---: |
| normal | 0.047 |
| injection-zero | 0.031 |
| forced-zero | 0.031 |
| forced-random | 0.039 |
| oracle-at-eval | 0.031 |

## Track 3

64-sample diagnostic outputs:

| Checkpoint | Exact | Operand | Calc result | Injection-zero | Forced-zero | Forced-random | Oracle-at-eval | Force-learned-result | Category | Bottleneck |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Model B strict/off | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `calculator_ignored_or_bypassed` | `calculator_required_bottleneck` |
| Oracle Model C strict | 0.906 | 1.000 | 1.000 | 0.000 | 0.016 | 0.062 | 0.906 | 0.906 | `valid_oracle_calculator_use` | `calculator_required_bottleneck` |
| Learned STE Model C strict | 0.000 | 0.000 | 0.000 | 0.047 | 0.047 | 0.047 | 0.047 | 0.000 | `calculator_ignored_or_bypassed` | `strict_bottleneck_unvalidated` |

The learned checkpoint's bottleneck label is per-checkpoint conservative because the checkpoint itself did not demonstrate calculator use; the strict setting is validated by the Model B/off and oracle controls above.

## Track 4

64 prompts, 16 random actions per prompt:

| Checkpoint | True NLL | Learned NLL | Random NLL | Shuffled NLL | Random-true gap | Shuffled-true gap | Operand exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Model B strict/off | 1.456 | 1.456 | 1.456 | 1.456 | 0.000 | 0.000 | 0.000 |
| Oracle Model C strict | 0.118 | 0.118 | 9.976 | 10.432 | 9.858 | 10.314 | 1.000 |
| Learned STE Model C strict | 1.614 | 1.195 | 1.603 | 1.599 | -0.011 | -0.016 | 0.000 |

Oracle action-loss sensitivity is strong: true operand actions massively reduce target loss relative to random and shuffled actions. Learned STE did not discover true operands and its learned private actions are preferred over true actions by the trained decoder.

## Validation

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/run_track3_causal_diagnostics.py scripts/run_track4_action_loss_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. pytest -q
```

Result: `46 passed`.

## Recommendation

Go for estimator comparisons only after treating this strict answer-decoder bottleneck as the baseline, not the old `replace` mode. The bottleneck now passes the important off/oracle controls, and Track 4 confirms the downstream decoder exposes a large true-action loss advantage in the oracle setting.

No-go on interpreting plain learned STE as successful: in this first strict run it reached only `0.037` exact match with `0.000` operand exact and no true-action loss advantage. The next estimator work should compare methods against this strict bottleneck with oracle and Track 4 gates kept mandatory.
