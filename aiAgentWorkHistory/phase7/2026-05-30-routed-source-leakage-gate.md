# Routed Source Leakage Gate

## Question

Does the cloned two-hook routed topk source transfer causally into additive
non-bottleneck mode, and if not, is the problem source weakness, semantic
freezing, or leakage through the upstream residual?

## Implementation Correction

`--freeze-semantic-decoder` previously did not freeze calculator output
projections for `--calculator-estimator ste` handoff runs. That made the first
handoff less strict than intended. The flag now always calls
`freeze_semantic_decoder_parameters`, and the CLI test checks that
`calculator_hook.output_proj.weight` is absent from trainable parameters when
semantic freezing is requested.

## Runs

Strict frozen handoff from the 200-step cloned routed source:

```text
runs/2026-05-30_phase7_routed_multi_hook_training/op19_rhead64_topk8_unique24_hooks2_cloneout_handoff600_strictfreeze_cpu/2026-05-30_135520_391326_model-c-op0-19-fullgrid-hooks2-routeleft_operand_mod-adec-product/model-c-2digit-seed43
```

Matched `embd32` routed source630:

```text
runs/2026-05-30_phase7_routed_multi_hook_training/op19_rhead64_topk8_unique24_hooks2_cloneout_embd32_source630_cpu/2026-05-30_140522_171215_model-c-op0-19-fullgrid-direct_feedback_alignment-hooks2-routeleft_operand_mod-answer_decoder-adec-product/model-c-2digit-seed43
```

Matched `embd32` routed source200 with frozen upstream encoder:

```text
runs/2026-05-30_phase7_routed_multi_hook_training/op19_rhead64_topk8_unique24_hooks2_cloneout_embd32_freezeup_source200_cpu/2026-05-30_141321_270677_model-c-op0-19-fullgrid-direct_feedback_alignment-hooks2-routeleft_operand_mod-answer_decoder-adec-product/model-c-2digit-seed43
```

## Results

| Run | Final eval | Snapshot | Injection-zero | Oracle | Forced-random | Hook calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Strict handoff from source200 | `0.9075` | step-600 normal `0.9175` | `0.4925` | `0.9925` | `0.0175` | `0.9438/0.8784` |
| Matched open-upstream source630 | `1.0000` | step-630 normal `0.9975` | `0.4600` | `1.0000` | `0.0200` | `1.0000/0.9944` |
| Frozen-upstream source200 | `0.3925` | step-200 normal `0.4150` | `0.1875` | `1.0000` | `0.0325` | `0.4384/0.3867` |

Reference single-hook comparator:

```text
runs/2026-05-30_phase7_assignment_cost_reduction/op19_rhead64_topk8_unique24_source630_cpu/.../model-c-2digit-seed43
```

That source reached step-630 normal `1.0000` with injection-zero `0.0275`.

## Interpretation

The routed sparse-assignment machinery can train both hooks. The matched
`embd32` source630 has nearly perfect per-hook calculator-result accuracy, so
the failure is not target starvation or hook-1 collapse.

The new failure mode is leakage through the open upstream residual. In the
routed source, normal accuracy remains high when calculator injection is
zeroed. This invalidates additive handoff claims from that source because the
source is not itself cleanly calculator-causal. Freezing the upstream encoder
pushes injection-zero down but slows result-policy learning at 200 steps.

The next useful work is not another open-upstream routed handoff. It is an
anti-leak routed source recipe: longer frozen-upstream training, staged
unfreezing/trust region, explicit source causal-gap pressure, or tied/shared
output projections plus a source control gate before any handoff.
