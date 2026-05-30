# 2026-05-30 - Multi-Hook Injection-Zero Fix

## Question

Were the high routed multi-hook injection-zero controls measuring residual
leakage, or were they failing to ablate every active calculator hook?

## Bug

`temporary_calculator_injection_scale` only changed
`model.calculator_hook.injection_scale`. In routed multi-hook runs, examples
assigned to `extra_calculator_hooks` still received calculator injections during
`injection_scale=0.0` evaluation, causal-gap training, and other forced-scale
contexts.

That explains the previous apparent `0.44-0.53` injection-zero values: in a
two-hook left-operand route, roughly half the examples still had an active
calculator path.

## Fix

The helper now iterates over `model.calculator_hook_modules()`, sets every
hook's `injection_scale`, and restores all original values afterward.

Added regression test:

```text
tests/test_model.py::test_temporary_injection_scale_applies_to_all_calculator_hooks
```

## Corrected Evidence

Source200 rerun with the matched `embd32` routed recipe:

```text
runs/2026-05-30_phase7_routed_multi_hook_training/op19_rhead64_topk8_unique24_hooks2_cloneout_embd32_source200_fixed_zero_cpu/2026-05-30_143634_710555_model-c-op0-19-fullgrid-direct_feedback_alignment-hooks2-routeleft_operand_mod-answer_decoder-adec-product/model-c-2digit-seed43
```

- Final eval: `376/400 = 0.9400`
- Step-200 normal: `0.9225`
- Step-200 injection-zero: `0.0200`
- Step-200 forced-random: `0.0325`
- Hook calc: `0.9406/0.9006`

Source630 checkpoint reload with corrected controls:

```text
runs/2026-05-30_phase7_routed_multi_hook_training/op19_rhead64_topk8_unique24_hooks2_cloneout_embd32_source630_fixed_zero_eval_cpu/2026-05-30_143917_927313_model-c-op0-19-fullgrid-direct_feedback_alignment-hooks2-routeleft_operand_mod-answer_decoder-adec-product/model-c-2digit-seed43
```

- Final eval: `400/400 = 1.0000`
- Reload snapshot normal: `0.9950`
- Reload snapshot injection-zero: `0.0250`
- Reload snapshot forced-random: `0.0325`
- Hook calc: `1.0000/0.9893`

Strict handoff600 checkpoint reload with corrected controls:

```text
runs/2026-05-30_phase7_routed_multi_hook_training/op19_rhead64_topk8_unique24_hooks2_cloneout_handoff600_strictfreeze_fixed_zero_eval_cpu/2026-05-30_143954_084896_model-c-op0-19-fullgrid-hooks2-routeleft_operand_mod-adec-product/model-c-2digit-seed43
```

- Final eval: `363/400 = 0.9075`
- Reload snapshot normal: `0.9250`
- Reload snapshot injection-zero: `0.0000`
- Reload snapshot forced-random: `0.0300`
- Hook calc: `0.9108/0.9198`

## Interpretation

The routed multi-hook source and strict handoff were calculator-causal under
corrected controls. The previous anti-leak branch was based on a measurement
bug, not a model failure.

Do not cite pre-fix routed multi-hook injection-zero values as evidence of
residual leakage. Future multi-hook counterfactuals and causal-gap objectives
must verify that every hook is ablated.

Next work should validate corrected-control routed training on a fresh seed,
more hooks, the stronger `embd32` source630-to-additive handoff, or a
shared/tied output projection that avoids per-hook semantic-output growth.
