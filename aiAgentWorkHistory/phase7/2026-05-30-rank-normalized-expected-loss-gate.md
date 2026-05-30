# 2026-05-30 Rank-Normalized Expected Answer-Loss Gate

## Question

Can a rank transform of forced-result answer losses fix the local gradient
direction of exact result-space expected answer loss?

Raw expected answer loss and ordinary sampled policy gradient were previously
anti-aligned with the answer-derived boundary target. This gate tests a
materially different cost transform: replace each prompt's forced-result NLLs
with within-prompt ranks before taking the policy expectation.

This is still full-enum for the diagnostic, but if it had cleared Stage 0 it
would have suggested a sampled/rank-estimator family worth testing.

## Code

Added `expected_answer_loss_cost_normalization=rank` in:

```text
scripts/overfit_one_batch.py
```

The transform maps the lowest-loss result to `0.0` and the highest-loss result
to `1.0` per prompt. Added a focused unit test for the per-prompt ranks.

## Run

```text
runs/2026-05-30_phase7_rank_expected_answer_loss_gate/2026-05-30_170553_274497_model-c-op0-19-fullgrid-full_enum_expected_answer_loss-result_space-inlr0.01-uplr0.0003-expanspolt1-expanschunk64-rank-expansgraddiag-answer_decoder-adec-product/model-c-2digit-seed4/expected_answer_loss_gradient_diagnostic_summary.json
```

Setup:

- natural `0..19` full grid
- result-space expected answer-loss objective
- frozen product answer decoder
- semantic decoder frozen
- `cost_normalization=rank`
- gradient diagnostic only; no Stage 1 training

## Result

| Metric | Value |
| --- | ---: |
| exact result-proj grad L2 | `0.016785` |
| exact upstream grad L2 | `0.006196` |
| exact vs boundary result-proj cosine | `0.049551` |
| exact vs boundary upstream cosine | `0.002584` |
| PG vs exact result/upstream cosine | `0.723444` / `0.719965` |
| PG vs boundary result/upstream cosine | `0.027483` / `0.016271` |
| boundary hard-best=true-sum | `1.0000` |

## Interpretation

Rank normalization weakly improves the result-head sign relative to raw
expected answer loss, but the upstream cosine is essentially zero. This is
weaker than the earlier contrastive-margin decoder gate, which had a stronger
local sign flip and still failed Stage 1 discovery. The rank objective does
not justify a long training run.

## Decision

```text
rank_normalized_expected_answer_loss_stage0_negative
```

Do not run rank-normalized expected answer-loss Stage 1 training as novelty.
Future expected-loss work needs a stronger structural estimator/objective than
per-prompt rank or scale transforms of the same full-enum costs.
