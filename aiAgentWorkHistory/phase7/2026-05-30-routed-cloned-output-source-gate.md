# Routed Cloned-Output Source Gate

## Question

Can the routed two-hook path train independent calculator result policies under
the lower-cost topk8+unique24 assignment proposal, and do the per-route
diagnostics identify why an extra hook fails?

## Implementation

- Result-policy reads are now route-aware: under `left_operand_mod`, each
  example trains against the result logits from its active hook.
- Training curves now report:
  - `result_policy_improvement_assignment_forced_eval_count`
  - `result_policy_active_hook_count`
  - `result_policy_route_distribution`
  - `result_policy_hook_{i}_route_count`
  - `result_policy_hook_{i}_assignment_fraction`
  - `result_policy_hook_{i}_assignment_target_accuracy`
- Final metrics now include `diagnostic_routed_summary` and write
  `diagnostic_routed_summary.json`.
- Added `--clone-primary-calculator-output-proj`, which copies the primary hook
  output projection into extra hooks after checkpoint loading and before
  freezing. This gives routed hooks the same result-to-residual semantic
  interface while leaving their result policies independent.

## Runs

Uncloned exact source200:

```text
runs/2026-05-30_phase7_routed_multi_hook_training/op19_rhead64_exact_hooks2_source200_cpu/2026-05-30_133201_337437_model-c-op0-19-fullgrid-direct_feedback_alignment-hooks2-routeleft_operand_mod-answer_decoder-adec-product/model-c-2digit-seed43
```

Uncloned topk8+unique24 source200:

```text
runs/2026-05-30_phase7_routed_multi_hook_training/op19_rhead64_topk8_unique24_hooks2_source200_cpu/2026-05-30_133439_344170_model-c-op0-19-fullgrid-direct_feedback_alignment-hooks2-routeleft_operand_mod-answer_decoder-adec-product/model-c-2digit-seed43
```

Uncloned exact source50 with route assignment metrics:

```text
runs/2026-05-30_phase7_routed_multi_hook_training/op19_rhead64_exact_hooks2_source50_route_metrics_cpu/2026-05-30_133849_897108_model-c-op0-19-fullgrid-direct_feedback_alignment-hooks2-routeleft_operand_mod-answer_decoder-adec-product/model-c-2digit-seed43
```

Cloned exact source50:

```text
runs/2026-05-30_phase7_routed_multi_hook_training/op19_rhead64_exact_hooks2_cloneout_source50_route_metrics_cpu/2026-05-30_134233_280608_model-c-op0-19-fullgrid-direct_feedback_alignment-hooks2-routeleft_operand_mod-answer_decoder-adec-product/model-c-2digit-seed43
```

Cloned topk8+unique24 source200:

```text
runs/2026-05-30_phase7_routed_multi_hook_training/op19_rhead64_topk8_unique24_hooks2_cloneout_source200_cpu/2026-05-30_134343_313820_model-c-op0-19-fullgrid-direct_feedback_alignment-hooks2-routeleft_operand_mod-answer_decoder-adec-product/model-c-2digit-seed43
```

## Results

| Run | Scored results | Forced evals / full-grid step | Final eval | Step normal | Oracle | Hook calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Uncloned exact200 | `39/39` | `15,600` | `0.4825` | `0.5075` | `0.5675` | `0.8767/0.0387` |
| Uncloned topk200 | `24/39` | `9,600` | `0.5250` | `0.5525` | `0.5675` | `0.9315/0.0110` |
| Cloned topk200 | `24/39` | `9,600` | `0.9025` | `0.9250` | `1.0000` | `0.9315/0.9171` |

Route assignment diagnostics:

| Run | Step | Hook 0 target acc | Hook 1 target acc |
| --- | ---: | ---: | ---: |
| Uncloned exact50 | 50 | `1.0000` | `0.0839` |
| Uncloned topk50 | 50 | `0.7929` | `0.0833` |
| Cloned exact50 | 50 | `0.8831` | `0.9333` |
| Cloned topk200 | 200 | `1.0000` | `1.0000` |

The cloned topk200 run also had injection-zero `0.4325` and forced-random
`0.0325` at step 200.

## Interpretation

The routed hook-1 collapse was not caused by route imbalance or lack of
assignment pressure. It was caused by an unaligned extra-hook output projection:
forced-result scoring through hook 1 did not have the same semantics as scoring
through the primary hook. Cloning the primary output projection fixes that
interface and lets the sparse assignment train both routed hooks.

This is a real many-hook source-training step, but it is not the project goal.
The run is still prescriptive hard assignment, source-only, one seed, and has a
high injection-zero control. The next meaningful gate is a trusted additive
handoff from the cloned routed topk checkpoint, followed by fresh-seed and a
true shared/tied output-projection implementation if the handoff is causal.
