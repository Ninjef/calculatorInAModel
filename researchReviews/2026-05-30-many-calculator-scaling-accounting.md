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

Diagnostic snapshots now also expose routed per-hook quality: routed rows read
the active hook trace, and `diagnostic_snapshots.csv` includes route
distribution plus per-hook route counts, operand accuracy, calculator-result
accuracy, and sampled log-probability. This closes an observability gap for the
next routed training diagnostic; it is not by itself evidence that hooks train
or specialize.

The first routed training gate found a real prerequisite. With two routed hooks
and frozen semantic decoder/output interface, extra hooks must not keep random
frozen output projections. Uncloned exact/topk source200 runs mostly trained
hook 0 and left hook 1 near chance. A 50-step exact route diagnostic showed
hook 1 was assigned targets, but they were semantically wrong (`0.0839` target
accuracy) because forced-result scoring through its random output projection
did not mean the same thing as the primary calculator output.

Adding `--clone-primary-calculator-output-proj` made the routed gate fair. The
cloned exact source50 route target accuracies were `0.8831/0.9333`, and the
cloned topk8+unique24 source200 reached `0.9250` step-200 normal with hook calc
`0.9315/0.9171` while scoring `24/39` results (`9,600` forced evals per
full-grid step versus `15,600` exact). This is closer to many-calculator source
training, but it is not yet a non-bottleneck proof: injection-zero was `0.4325`,
and no trusted handoff or fresh seed has been run.

The follow-up handoff/source controls make the next bottleneck sharper. A
strict frozen-policy additive handoff from the routed source200 reached high
normal accuracy but high injection-zero (`0.4925`). Re-running the routed
source with the matched `embd32` product decoder parity checkpoint and 630-step
recipe trained both hooks almost perfectly (`1.0000/0.9944` hook calc), but
source injection-zero was still high (`0.4600`), unlike the single-hook
`embd32` source630 (`0.0275`). Freezing upstream reduced leakage at source200
(`0.1875`) but undertrained (`0.4150` normal). A longer frozen-upstream
source630 recovered learning (`0.9475` final, `0.9750` step-630 normal,
`0.9955/0.9494` hook calc), but injection-zero returned (`0.4400` snapshot,
`0.5000` final counterfactual). The active many-calculator issue is therefore
no longer only scorer count or hook observability; routed source training needs
source-time anti-leak pressure or a stricter routed architecture before handoff
results can be trusted.

Correction after the review: those routed injection-zero controls were invalid
because the temporary injection-scale helper zeroed only the primary hook. The
fixed helper scales every routed hook. Corrected evidence is much stronger:
source200 rerun reached `0.9225` normal / `0.0200` injection-zero, source630
reload reached `0.9950` normal / `0.0250` injection-zero, and strict handoff600
reload reached `0.9250` normal / `0.0000` injection-zero. The many-calculator
bottleneck is no longer anti-leak routed acquisition; it is validating routed
training across fresh seeds/more hooks and removing cloned per-hook output
projection parameter growth.

The next corrected-control gate did validate a stronger routed handoff:
the `embd32` source630 checkpoint reached `1.0000` final and `1.0000`
step-600 normal in trusted additive handoff, with `0.0550` injection-zero,
`0.0300` forced-random, and hook calc `1.0000/0.9955`. This upgrades routed
sparse assignment from source-only evidence to a real two-hook non-bottleneck
positive, while leaving fresh-seed/more-hook/shared-output validation as the
remaining scalability work.

The first more-hook validation also cleared: four routed hooks reached
`0.9950` source final with all hooks trained and then `1.0000` trusted handoff
final with `0.0400` injection-zero, `0.0200` forced-random, and all four hooks
at `1.0000` calculator-result accuracy on the final snapshot. This strengthens
the many-calculator trainability claim. The remaining scaling gap is now
implementation/economics: current routed execution still calls every hook
before masking and still clones per-hook output projections.

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
