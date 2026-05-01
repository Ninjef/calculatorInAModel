# 2026-04-30 - Track 3 causal diagnostics, codebooks, and falsification

Task: harden the checkpoint-first diagnostic suite and apply it to the six Track 2 priority checkpoints.

## Implementation

- Extended `scripts/diagnose_calculator_protocol.py` to always write trace rows, diagnostic summary JSON, result and operand codebooks, a counterfactual exact-match table, and bottleneck/protocol classification.
- Added tensor-valued forced result classes and batched forced-result sweeps for 2-digit result vocabularies.
- Added read-site interventions for swapping or corrupting only A/B calculator read vectors.
- Added `scripts/run_track3_causal_diagnostics.py` as the six-checkpoint manifest runner.

## Track 2 checkpoint classifications

All rows below use `64` diagnostic samples.

| Checkpoint | Exact | Operand | Calc result | Zero inj | Forced random | Classification | Bottleneck | Output |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| Additive answer-only learned Model C | 0.750 | 0.000 | 0.031 | 0.766 | 0.750 | calculator_ignored_or_bypassed | non_bottleneck_leaky_by_design | `runs/2026-04-30_124622_062528_model-c-op0-19/model-c-2digit-seed2/track3_diagnostics` |
| Additive high-answer aux 0.01 Model C | 0.875 | 0.016 | 0.047 | 0.859 | 0.875 | calculator_ignored_or_bypassed | non_bottleneck_leaky_by_design | `runs/2026-04-30_124941_086322_model-c-op0-19-aux0.01/model-c-2digit-seed2/track3_diagnostics` |
| Replacement Model B/off leakage control | 0.844 | 0.000 | 0.000 | 0.844 | 0.844 | calculator_ignored_or_bypassed | invalid_or_leaky_bottleneck | `runs/2026-04-30_131732_611676_model-b-op0-19-replace/model-b-2digit-seed2/track3_diagnostics` |
| Replacement oracle Model C control | 1.000 | 1.000 | 1.000 | 0.047 | 0.062 | valid_oracle_calculator_use | invalid_or_leaky_bottleneck | `runs/2026-04-30_131816_053136_model-c-oracle-op0-19-replace/model-c-2digit-seed2/track3_diagnostics` |
| Replacement answer-only learned Model C | 0.094 | 0.000 | 0.047 | 0.109 | 0.062 | calculator_ignored_or_bypassed | invalid_or_leaky_bottleneck | `runs/2026-04-30_131959_337278_model-c-op0-19-replace/model-c-2digit-seed2/track3_diagnostics` |
| Replacement aux decay transient candidate | 0.078 | 0.000 | 0.031 | 0.016 | 0.047 | causally_useful_opaque_private_code | invalid_or_leaky_bottleneck | `runs/2026-04-30_132810_628910_model-c-op0-19-replace-aux0.03-auxdecay1000/model-c-2digit-seed2/track3_diagnostics` |

## Interpretation

- Additive high answer accuracy remains bypass-compatible: operand exact match and calculator result accuracy stay near chance, and counterfactuals do not show clean dependence on the calculator.
- Replacement oracle remains the positive control: true operands and true calculator result are usable downstream, and removing/corrupting calculator output collapses accuracy.
- Current replacement learned checkpoints are not valid calculator-required bottleneck evidence. The Model B/off replacement control reaches high exact match, confirming leakage through ordinary autoregressive context.
- The learned replacement checkpoints are best treated as failed or bypass-heavy protocol-learning attempts, not as evidence against a stricter future bottleneck.

## Validation

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/diagnose_calculator_protocol.py scripts/run_track3_causal_diagnostics.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. pytest -q
python3 scripts/run_track3_causal_diagnostics.py --samples 8 --limit 1 --skip-forced-sweep
python3 scripts/run_track3_causal_diagnostics.py --samples 64
```

Conclusion: Track 3 makes answer accuracy much harder to overstate by pairing it with protocol codebooks, mutual information, counterfactuals, read-site interventions, and explicit bottleneck/leakage labels.
