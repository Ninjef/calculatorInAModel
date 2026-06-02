# 2026-06-02 Capped Prior Many-Calculator Accounting

## Why This Review Exists

The quality-gated capped-prior recipe is now the op29 source/handoff cost lead
and has fresh-seed handoff replication. The next strategic question is whether
that result meaningfully satisfies the user's many-calculator scalability
requirement.

## What Changed

Added reproducible accounting for the current numeric-prior replay family:

```bash
python3 scripts/analyze_prior_replay_scaling.py --calculator-counts 1,4,16,64
```

The accounting uses the measured original capped op29 source costs:

- `720` prompt-memory entries per calculator;
- about `192` sparse candidate-scoring steps before memory fill;
- `294,912` sparse forced-candidate evaluations per calculator;
- `1,254,817` prior fit examples per calculator;
- `1,080,000` full-memory fit examples per calculator;
- `7,995` numeric-prior parameters per calculator.

| Calculators | Candidate evals | Prior fit examples | Full-fit examples | Candidate + prior examples |
| ---: | ---: | ---: | ---: | ---: |
| `1` | `294,912` | `1,254,817` | `1,080,000` | `1,549,729` |
| `4` | `1,179,648` | `5,019,268` | `4,320,000` | `6,198,916` |
| `16` | `4,718,592` | `20,077,072` | `17,280,000` | `24,795,664` |
| `64` | `18,874,368` | `80,308,288` | `69,120,000` | `99,182,656` |

The fresh-seed cap run is nearly identical in cost. Using its `1,260,852` fit
examples gives `24,892,224` candidate + prior examples at 16 calculators and
`99,568,896` at 64.

## Interpretation

This is a real improvement over repeated top-k hard assignment. The previous
op29 topk8+unique24 hard-assignment accounting cost `217,728,000` forced-result
evaluations at 16 calculators over the 630-step source window. The capped-prior
recipe instead bounds candidate scoring by memory fill, and its h128 numeric
prior has only `7,995` parameters per calculator.

But it does not clear the scalability bar. The cost simply moves from repeated
forced-result scoring into per-calculator prompt memory and per-calculator
prior fitting. Both remain linear in independent active calculator count, and
prior fitting dominates the current accounting.

## What Should Stop

- Do not run cap-value, proportional-fraction, refresh-window, or same-recipe
  seed ladders as scalability work.
- Do not claim the capped-prior recipe is scalable merely because candidate
  scoring is much cheaper than repeated top-k assignment.
- Do not treat small numeric-prior parameter count as the main scaling answer;
  optimizer examples and target discovery dominate.

## What Deserves Compute

- A shared or global prior that amortizes target learning across calculators.
- A mechanism that removes per-calculator prompt-memory target tables.
- A less-prescriptive credit-assignment signal that avoids answer-derived
  candidate scoring altogether.

## Are We Closer?

Yes on cost accounting and bounded candidate scoring; no on the full
many-calculator requirement. Family 14 is now a strong benchmark, but the next
algorithmic step must break the per-calculator target/prior scaling rather than
polish the current cap.
