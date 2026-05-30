# 2026-05-30 Many-Calculator Assignment Scaling Accounting

## Context

After replicated op29 policy-topk validation, another same-family seed would
not answer the scalability requirement. I added a small accounting script to
quantify how exact versus topk8+unique24 hard assignment scales with operand
range and independent calculator count.

## Code

- Added `scripts/analyze_assignment_scaling.py`.
- Added `tests/test_assignment_scaling.py`.

The script computes result vocabulary size, forced-candidate evaluation count,
saved evaluations, and `rhead64` result-head parameters for configurable
operand ranges and calculator counts.

## Command

```bash
python3 scripts/analyze_assignment_scaling.py --operand-maxes 19,29,39,99 --calculator-counts 1,4,16,64 --sample-count 24 --assignment-steps 630 --n-embd 32 --span-width 2 --result-head-hidden-size 64
```

## Key Results

| Range | Calculators | Exact forced evals | Topk8+unique24 forced evals | Eval savings | Result-head params |
| --- | ---: | ---: | ---: | ---: | ---: |
| op29 | `1` | `33,453,000` | `13,608,000` | `19,845,000` (`59.3%`) | `12,091` |
| op29 | `16` | `535,248,000` | `217,728,000` | `317,520,000` (`59.3%`) | `193,456` |
| op39 | `1` | `79,632,000` | `24,192,000` | `55,440,000` (`69.6%`) | `13,391` |
| op39 | `16` | `1,274,112,000` | `387,072,000` | `887,040,000` (`69.6%`) | `214,256` |
| op99 | `16` | `20,059,200,000` | `2,419,200,000` | `17,640,000,000` (`87.9%`) | `339,056` |

## Interpretation

Policy-topk sparse assignment materially reduces result-class scorer work as
range grows, but it is not a full many-calculator solution. The current model
has one calculator hook, and independent calculators would still multiply
scorer cost and result-head parameters by active calculator count. This turns
topk8+unique24 into the lower-cost hard-assignment baseline to beat rather than
the end of the scalability story.

## Verification

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache pytest tests/test_assignment_scaling.py -q
```

Result: `3 passed in 0.01s`.
