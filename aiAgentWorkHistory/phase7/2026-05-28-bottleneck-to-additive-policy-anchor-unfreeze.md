# 2026-05-28 Bottleneck-to-Additive Policy-Anchor Unfreeze

## Goal

Test whether explicit result-policy retention can make full-policy unfreezing
safe after a staged non-bottleneck handoff.

## Code Changes

- Added `capture_result_policy_anchor`.
- Added `result_policy_anchor_loss`.
- Added CLI flags:
  - `--result-policy-anchor-weight`
  - `--result-policy-anchor-decay-steps`
  - `--result-policy-anchor-temperature`
  - `--result-policy-anchor-mode {kl,mse}`
- Added training-curve and metrics logging for anchor loss, agreement, KL, MSE,
  and current/anchor result accuracy.
- Added a focused test that the anchor loss is near zero at capture time,
  becomes positive after result-projection drift, and sends gradient to the
  result head.

## Runs

Run root:

```text
runs/2026-05-28_phase7_bottleneck_to_additive_transfer_policy_anchor_unfreeze
```

Configuration:

- resumed from adapted weak-source additive checkpoints;
- `--semantic-decoder-checkpoint-load-scope full_model`;
- no `--freeze-calculator-policy`;
- global LR `3e-4`;
- `--result-policy-anchor-weight 10`;
- `--result-policy-anchor-mode kl`;
- answer loss weight `1`;
- exact-grid natural `0..19`;
- 400 steps.

## Results

| Run | Frozen adapted final | Plain unfreeze final | Anchored final | Best normal | Last injection-zero | Last forced-random | Last oracle | Last learned calc | Anchor agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `src4_add2` anchor | `0.6050` | `0.5200` | `0.7475` | `0.7375` at `150` | `0.0100` | `0.0950` | `0.7875` | `0.8075` | `0.9800` |
| `src5_add5` anchor | `0.8175` | `0.8100` | `0.9525` | `0.9650` at `400` | `0.0000` | `0.0450` | `0.9375` | `0.7950` | `0.9850` |

## Conclusion

Label:

```text
bottleneck_to_additive_policy_anchor_unfreeze_partial
```

The explicit result-policy anchor prevents the learned calculator-result
collapse from the plain low-LR unfreeze negative. It allows useful full-policy
adaptation in the non-bottleneck setting while preserving calculator
dependence.

This is still not the final scalable or non-prescriptive solution because it
anchors a staged learned policy.

## Verification

Focused verification after code changes:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q -k "result_policy_anchor or semantic_decoder_checkpoint_load_scope"
git diff --check
```

Result: `2 passed, 105 deselected`.

Final verification before commit:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py src/model.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
git diff --check
```

Result: `107 passed`.
