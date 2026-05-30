# 2026-05-30 Zero-Improvement Boundary Source Gate

## Question

Can an answer-derived result-boundary target train the result-space policy
without directly selecting the true sum or the full-enum argmin?

The tested target weights result classes by how much their forced answer loss
improves over a zero-injection/no-calculator baseline:

```text
weight(result) proportional max(0, zero_injection_loss - forced_result_loss)
```

This is less prescriptive than hard-best result-boundary training because it
does not tell the model which result is correct; it asks which results make the
answer better than no calculator.

## Code

- Added `result_boundary_target_mode=zero_improvement`.
- Added zero-improvement metrics to `training_curve.csv`:
  - zero-injection baseline loss;
  - fraction of prompts with any positive-improvement result;
  - fraction where the true result receives positive mass;
  - mean positive improvement mass.
- Added a focused unit test for target weights.

Focused verification:

```text
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/codex_pycache pytest tests/test_model.py -q -k "zero_improvement or result_boundary_cli_validation or result_boundary_target_uses_lowest_nll_result or result_boundary_regret_set_targets_near_best_results"
```

Result: `4 passed, 140 deselected`.

## Runs

Full enumeration:

```text
runs/2026-05-30_phase7_zero_improvement_boundary_source_gate/full_enum_step200_cpu/2026-05-30_172500_706888_model-c-op0-19-fullgrid-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed4
```

Topk8+unique24 sparse scoring:

```text
runs/2026-05-30_phase7_zero_improvement_boundary_source_gate/topk8_unique24_step200_cpu/2026-05-30_172631_037509_model-c-op0-19-fullgrid-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed4
```

Shared setup:

- op19 full-grid source gate, 400 prompts.
- `answer_loss_weight=0`.
- frozen product semantic decoder.
- `calculator_bottleneck_mode=answer_decoder`.
- 200 training steps, snapshots every 50.

## Results

| Branch | Scored classes | Step-200 true coverage | Step-200 true target mass | Step-200 effective results | Step-200 learned-best/calc | Snapshot normal/calc | Final eval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full-enum zero-improvement | `39/39` | `1.0000` | `0.9541` | `1.2692` | `0.5475` | `0.5700` | `0.5425` |
| topk8+unique24 zero-improvement | `24/39` | `0.9725` | `0.9356` | `1.2259` | `0.4275` | `0.4525` | `0.4300` |
| topk8+unique24 hard-best sampled comparator | `24/39` | `0.9600` | n/a | n/a | `0.3425` | `0.3675` | `0.3525` |

Full-enum curve:

| Step | Learned-best/calc | Snapshot normal/calc | Zero-improvement positive prompts | True target mass |
| ---: | ---: | ---: | ---: | ---: |
| `0` | `0.0225` | `0.0250` | `1.0000` | `0.9541` |
| `50` | `0.1025` | `0.1050` | `1.0000` | `0.9541` |
| `100` | `0.2100` | `0.1900` | `1.0000` | `0.9541` |
| `150` | `0.3375` | `0.3350` | `1.0000` | `0.9541` |
| `200` | `0.5475` | `0.5700` | `1.0000` | `0.9541` |

Sparse curve:

| Step | True coverage | Learned-best | Snapshot normal/calc | Positive prompts | True target mass |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `0.6025` | `0.0275` | `0.0250` | `0.6750` | `0.5862` |
| `50` | `0.6575` | `0.1125` | `0.0775` | `0.7475` | `0.6366` |
| `100` | `0.8125` | `0.1450` | `0.1325` | `0.8600` | `0.7856` |
| `150` | `0.9125` | `0.2475` | `0.2475` | `0.9425` | `0.8795` |
| `200` | `0.9725` | `0.4275` | `0.4525` | `0.9800` | `0.9356` |

## Interpretation

Full-enum zero-improvement is a real lead. It reaches the same neighborhood as
nearby full-enum hard-best result-boundary comparators:

- soft-target gate hard-best comparator: `0.5450` learned calc / `0.5475`
  final eval;
- regret-set gate hard-best comparator: `0.4625` learned calc / `0.4225`
  final eval;
- zero-improvement: `0.5475` learned calc / `0.5425` final eval.

It is not a final scalable method. The topk8+unique24 sparse version improves
over sampled hard-best (`0.4300` vs `0.3525` final), but still trails full
enumeration despite high true-candidate coverage (`0.9725`).

## Decision

```text
zero_improvement_boundary_target_partial_positive
```

Do not call this solved, and do not run blind sample-count ladders. The useful
next tests are either:

- longer source plus trusted additive handoff for full-enum zero-improvement,
  to see whether this less-prescriptive target transfers causally; or
- a changed sparse proposal/training mechanism that closes the `0.4300 ->
  0.5425` source gap at lower scorer cost.
