# 2026-04-30 - Track 4 optimization estimators and action-loss signal

Task: start Track 4 by measuring whether calculator operand actions create enough downstream answer-loss signal to justify estimator sweeps.

## Implementation

- Added `scripts/run_track4_action_loss_diagnostic.py`.
- Reused the Track 3 six-checkpoint manifest and classification labels.
- For each prompt, the diagnostic measures target answer NLL under normal learned actions, forced learned operands, true operands, shuffled true operands, and random operand actions.
- The output includes per-action CSV rows, per-prompt summaries, and JSON aggregates for action-loss variance and true/random/shuffled/learned gaps.

## Diagnostic Run

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_track4_action_loss_diagnostic.py --samples 64 --random-actions 16
```

| Checkpoint | Track 3 class | Bottleneck label | True NLL | Random NLL | Random-true gap | Action-loss std | True best | Learned best | Operand exact | Output |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Additive answer-only learned Model C | calculator_ignored_or_bypassed | non_bottleneck_leaky_by_design | 0.2204 | 0.2145 | -0.0060 | 0.0567 | 0.0625 | 0.0000 | 0.0000 | `runs/2026-04-30_124622_062528_model-c-op0-19/model-c-2digit-seed2/track4_action_loss` |
| Additive high-answer aux 0.01 Model C | calculator_ignored_or_bypassed | non_bottleneck_leaky_by_design | 0.0858 | 0.1114 | 0.0256 | 0.0449 | 0.0938 | 0.0000 | 0.0156 | `runs/2026-04-30_124941_086322_model-c-op0-19-aux0.01/model-c-2digit-seed2/track4_action_loss` |
| Replacement Model B/off leakage control | calculator_ignored_or_bypassed | invalid_or_leaky_bottleneck | 0.1553 | 0.1553 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `runs/2026-04-30_131732_611676_model-b-op0-19-replace/model-b-2digit-seed2/track4_action_loss` |
| Replacement oracle Model C control | valid_oracle_calculator_use | invalid_or_leaky_bottleneck | 0.0000 | 9.3949 | 9.3949 | 4.4842 | 0.0000 | 0.0000 | 1.0000 | `runs/2026-04-30_131816_053136_model-c-oracle-op0-19-replace/model-c-2digit-seed2/track4_action_loss` |
| Replacement answer-only learned Model C | calculator_ignored_or_bypassed | invalid_or_leaky_bottleneck | 1.0149 | 1.0214 | 0.0065 | 0.1598 | 0.0156 | 0.0000 | 0.0000 | `runs/2026-04-30_131959_337278_model-c-op0-19-replace/model-c-2digit-seed2/track4_action_loss` |
| Replacement aux decay transient candidate | causally_useful_opaque_private_code | invalid_or_leaky_bottleneck | 1.6634 | 1.5016 | -0.1618 | 0.4002 | 0.0312 | 0.0000 | 0.0000 | `runs/2026-04-30_132810_628910_model-c-op0-19-replace-aux0.03-auxdecay1000/model-c-2digit-seed2/track4_action_loss` |

## Interpretation

- The diagnostic is intentionally checkpoint-first: it measures the action-loss landscape before changing estimators.
- Existing additive checkpoints remain bypass baselines, and existing replacement checkpoints keep Track 3's invalid/leaky bottleneck label.
- Positive random-true gaps mean true calculator actions lower target loss relative to sampled random actions; near-zero gaps mean the downstream answer loss has little operand-action signal to exploit.
- Estimator comparisons should still wait for a stricter calculator-required bottleneck, because Track 2/3 showed the current replacement mode leaks through autoregressive answer-token context.

## Validation

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/run_track4_action_loss_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. pytest -q
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_track4_action_loss_diagnostic.py --samples 4 --random-actions 2 --limit 1
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_track4_action_loss_diagnostic.py --samples 64 --random-actions 16
```

Conclusion: Track 4 now has a direct operand-action loss diagnostic. The next necessary step is a stricter bottleneck that passes Model B/off and oracle controls before estimator sweeps.
