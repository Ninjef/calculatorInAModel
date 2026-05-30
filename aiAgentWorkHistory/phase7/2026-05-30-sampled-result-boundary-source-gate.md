# 2026-05-30 Sampled Result-Boundary Source Gate

## Question

Can the answer-derived result-boundary source objective be made cheaper by
scoring only a policy-aware candidate subset during training?

This is deliberately not a generic count sweep. It tests whether the known
hard-assignment cost reducer, policy top-k plus unique sampled candidates, also
works for the less-prescriptive result-boundary source objective.

## Code

- Added `--result-boundary-target-sample-count`.
- Added `--result-boundary-target-unique-sampling`.
- Added `--result-boundary-target-policy-topk-count`.
- Added sampled candidate metrics for scored classes, unique candidate count,
  true-candidate coverage, and forced eval count.
- Added focused parser/loss tests in `tests/test_model.py`.

Focused verification during implementation:

```text
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/codex_pycache pytest tests/test_model.py -q -k "sampled_result_boundary or result_boundary_cli_validation or result_boundary_target_uses_lowest_nll_result"
```

The focused pytest gate passed: `3 passed, 140 deselected`.

## Run

```text
runs/2026-05-30_phase7_sampled_result_boundary_source_gate/topk8_unique24_step200_cpu/2026-05-30_171340_576865_model-c-op0-19-fullgrid-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-hard_best_result-rbtt0.25-rbtchunk64-rbts24-rbtuniq-rbttopk8-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed4
```

Key setup:

- `operand_max=19`, full-grid batch, 400 prompts.
- Bottleneck source mode: `calculator_bottleneck_mode=answer_decoder`.
- Result-boundary target mode: `hard_best_result`.
- Candidate scoring: `24/39` result classes per prompt.
- Policy-aware candidates: current policy top-8 plus unique sampled fill.
- Answer loss weight: `0`.

## Results

| Step | True coverage | Learned-best fraction | Snapshot normal/calc | Injection-zero | Oracle |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `0.6025` | `0.0275` | `0.0250` | `0.0575` | `1.0000` |
| `50` | `0.6450` | `0.1175` | `0.0375` | `0.0400` | `1.0000` |
| `100` | `0.7950` | `0.1500` | `0.1000` | `0.0400` | `1.0000` |
| `150` | `0.9075` | `0.2400` | `0.2525` | `0.0650` | `1.0000` |
| `200` | `0.9600` | `0.3425` | `0.3675` | `0.0475` | `1.0000` |

Final eval:

```text
exact_match = 141/400 = 0.3525
```

## Interpretation

The sampled candidate target did learn something, but it did not preserve the
full-enum result-boundary source signal. Candidate coverage was high by step
`200` (`0.9600`), so the miss is not mostly "the true result was never sampled."
The sparse/candidate hard-best signal itself is weaker.

Matched full-enum hard-best comparators from the same result-boundary source
cluster were stronger:

- soft-target gate hard-best comparator: step-200 learned calc / final eval
  `0.5450` / `0.5475`;
- regret-set gate hard-best comparator: step-200 learned calc / final eval
  `0.4625` / `0.4225`.

## Decision

```text
sampled_result_boundary_topk_unique24_source_negative
```

Do not continue with simple `sample_count`, top-k count, or unique-sampling
ladders around this mechanism. Future result-boundary source work needs active
proposal/training co-design, a stronger online/state-calibrated proposal, or a
different target construction. Otherwise, move to a different less-prescriptive
credit-assignment family.
