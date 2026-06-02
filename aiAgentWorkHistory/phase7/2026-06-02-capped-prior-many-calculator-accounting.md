# 2026-06-02 Capped Prior Many-Calculator Accounting

## Question

Does the current op29 quality-gated capped-prior recipe satisfy the
many-calculator scalability requirement, or does it merely move the cost from
forced-result scoring into prior fitting?

## Code

- Added `scripts/analyze_prior_replay_scaling.py`.
- Added `tests/test_prior_replay_scaling.py`.

The script scales the measured capped-prior op29 source costs across independent
active calculators:

- sparse candidate scoring before prompt memory fills;
- amortized-prior fit examples;
- full-memory prior fit examples;
- prompt-memory entries;
- numeric-prior MLP parameters.

## Command

```bash
python3 scripts/analyze_prior_replay_scaling.py --calculator-counts 1,4,16,64
```

Defaults match the original capped op29 source:

- operand range `0..29`;
- 20% heldout split, so `720` prompt-memory entries per calculator;
- about `192` sparse-scoring steps before memory fill;
- batch size `64`;
- topk/unique candidate count `24`;
- `1,254,817` total prior fit examples;
- `1,080,000` full-memory fit examples;
- numeric h128 prior.

## Results

| Calculators | Prompt memory | Candidate evals | Prior fit examples | Full-fit examples | Prior params | Candidate + prior examples |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | `720` | `294,912` | `1,254,817` | `1,080,000` | `7,995` | `1,549,729` |
| `4` | `2,880` | `1,179,648` | `5,019,268` | `4,320,000` | `31,980` | `6,198,916` |
| `16` | `11,520` | `4,718,592` | `20,077,072` | `17,280,000` | `127,920` | `24,795,664` |
| `64` | `46,080` | `18,874,368` | `80,308,288` | `69,120,000` | `511,680` | `99,182,656` |

For comparison, the older op29 topk8+unique24 hard-assignment accounting cost
`217,728,000` forced-result evals at 16 calculators over the 630-step source
window. The capped-prior recipe is much cheaper on candidate scoring, but it
still requires per-calculator prompt memory and per-calculator prior fitting.

The fresh capped seed is essentially the same cost: replacing the original
`1,254,817` prior fit examples with `1,260,852` gives `24,892,224` candidate +
prior examples at 16 calculators and `99,568,896` at 64.

## Interpretation

Result:

```text
capped_numeric_prior_replay_improves_candidate_cost_but_still_scales_linearly
```

The capped-prior family materially improves the old top-k assignment cost
profile. Candidate scoring is bounded by prompt-memory fill instead of repeated
throughout the source run, and the numeric prior is small (`7,995` parameters
per calculator).

But this is not enough for the user's scalability requirement. If independent
active calculators each need their own prompt memory and amortized prior, total
training work remains linear in calculator count. At 64 active calculators on
the tiny op29 task, the cost is already about `99M` candidate/prior examples,
mostly prior fitting. Larger tasks or calculators with broader query spaces
would worsen this.

Next work should not tune the cap. It should either:

- share/amortize target discovery and prior fitting across calculators;
- remove answer-derived candidate scoring;
- or test a credit mechanism that does not require per-calculator prompt-memory
  target tables.

## Verification

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache pytest tests/test_prior_replay_scaling.py tests/test_assignment_scaling.py -q
```

Result: `7 passed in 0.02s`.
