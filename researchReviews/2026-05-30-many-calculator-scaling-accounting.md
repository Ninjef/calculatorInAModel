# 2026-05-30 Many-Calculator Scaling Accounting

## Why This Review Exists

The policy-topk sparse assignment branch has cleared two op19 staged seeds and
two op29 range seeds. That is enough to stop asking whether this exact
single-calculator recipe can replicate locally. The next question is whether it
actually addresses the user's scalability requirement: many calculators
scattered through a model without seriously handicapping training.

## What Changed

Added reproducible accounting:

```bash
python3 scripts/analyze_assignment_scaling.py --operand-maxes 19,29,39,99 --calculator-counts 1,4,16,64 --sample-count 24 --assignment-steps 630 --n-embd 32 --span-width 2 --result-head-hidden-size 64
```

The current code has one calculator hook. This review therefore treats many
independent calculators as a cost model: scorer work and independent result-head
parameters multiply by the number of active calculators.

| Range | Calculators | Exact forced evals | Topk8+unique24 forced evals | Eval savings | Result-head params |
| --- | ---: | ---: | ---: | ---: | ---: |
| op29 | `1` | `33,453,000` | `13,608,000` | `19,845,000` (`59.3%`) | `12,091` |
| op29 | `16` | `535,248,000` | `217,728,000` | `317,520,000` (`59.3%`) | `193,456` |
| op39 | `1` | `79,632,000` | `24,192,000` | `55,440,000` (`69.6%`) | `13,391` |
| op39 | `16` | `1,274,112,000` | `387,072,000` | `887,040,000` (`69.6%`) | `214,256` |
| op99 | `16` | `20,059,200,000` | `2,419,200,000` | `17,640,000,000` (`87.9%`) | `339,056` |

## Interpretation

Policy-topk sparse assignment changes the per-calculator candidate-scoring
slope from `O(result_vocab)` to `O(24)` for the tested setup. This is the first
cost-reduction branch with both source/handoff evidence and a meaningful range
scaling argument.

But it does not solve many-calculator scaling. Even when independent
calculators are instantiated, scorer cost and head parameters remain linear in
active calculator count. The method is also still prescriptive: hard assignment
scores forced calculator results and trains the policy toward the best scored
result.

## Implementation Follow-Up

The repo now has a same-layer multi-hook prerequisite path:
`GPTConfig.calculator_hook_count` instantiates independent calculator hooks,
their injections are summed at the hook layer, diagnostics report
`calculator_active_hook_count` plus per-hook injections, and
`scripts/overfit_one_batch.py` accepts `--calculator-hook-count`.

A zero-step smoke with `--calculator-hook-count 3` wrote `calculator_hook_count=3`
in both `config.json` and `metrics.json`, and tests verify that extra hook
policy heads are grouped/frozen with the primary calculator head. This removes
the immediate single-hook code blocker, but it is not yet a routed or scattered
many-calculator training result.

A first routed variant now exists too. `calculator_hook_routing='left_operand_mod'`
activates one hook per fixed-width prompt by final left-operand digit modulo
hook count, reports route IDs/counts, zeroes non-routed applied injections, and
is exposed through `--calculator-hook-routing left_operand_mod`. A zero-step
smoke with three hooks wrote matching routing/count fields in config and
metrics. This enables a task-partitioned diagnostic, but still does not prove
per-hook specialization or scalable training.

## What Should Stop

- More op19 or op29 topk8+unique24 seed replications as scalability evidence.
- Treating a lower result-class count as proof of many-calculator feasibility
  without a routed/multi-hook diagnostic.
- Claiming topk solves the thesis; it is now the lower-cost hard-assignment
  baseline to beat.

## What Deserves Compute

- A true multi-calculator/routing diagnostic that reports active calculator
  count, scorer calls, and whether independent policies can train without
  shared-target leakage.
- An op39 sparse-vs-exact test only with an explicit compute hypothesis and a
  clear stopping rule.
- Less-prescriptive target construction that reduces or removes forced-result
  scoring, using topk8+unique24 as the staged baseline.

## Are We Closer?

Closer on the candidate-scoring slope, not yet on the full many-calculator
requirement. The topk branch is now strong enough to be a baseline; future
work should either instantiate many calculators/routing or move beyond hard
assignment.
