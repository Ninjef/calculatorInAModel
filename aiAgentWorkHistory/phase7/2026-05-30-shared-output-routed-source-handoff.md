# 2026-05-30 - Shared-output routed source and handoff

Task: empirically validate whether shared output projection is a drop-in
replacement for cloned per-hook output projections in the four-hook routed
source/handoff recipe.

## Question

Can the known four-hook routed source and trusted additive handoff still work
when `--share-calculator-output-proj` replaces
`--clone-primary-calculator-output-proj`?

## Source

Path:

```text
runs/2026-05-30_phase7_routed_multi_hook_training/op19_rhead64_topk8_unique24_hooks4_shareout_embd32_source630_cpu/2026-05-30_152821_990258_model-c-op0-19-fullgrid-direct_feedback_alignment-hooks4-routeleft_operand_mod-answer_decoder-adec-product/model-c-2digit-seed43
```

This matched the known cloned-output four-hook source recipe except for the
output interface:

- `--share-calculator-output-proj`
- `--calculator-hook-count 4`
- `--calculator-hook-routing left_operand_mod`
- `--calculator-result-head-hidden-size 64`
- `topk8+unique24`
- product decoder parity checkpoint
- late recovery from step `600`

Results:

- Final eval: `400/400 = 1.0000`.
- Step-630 normal/source-calc: `1.0000`.
- Step-630 injection-zero: `0.0275`.
- Step-630 forced-random: `0.0300`.
- Step-630 oracle: `1.0000`.
- Step-630 hook calculator-result accuracy:
  `1.0000/1.0000/1.0000/1.0000`.
- Final 128-sample counterfactuals: `0.0391` injection-zero, `0.0234`
  forced-random.

## Trusted Handoff

Path:

```text
runs/2026-05-30_phase7_routed_multi_hook_training/op19_rhead64_topk8_unique24_hooks4_shareout_embd32_handoff600_from_source630_cpu/2026-05-30_153417_398814_model-c-op0-19-fullgrid-hooks4-routeleft_operand_mod-adec-product/model-c-2digit-seed43
```

Settings:

- Trusted 600-step frozen-policy additive handoff.
- `--semantic-decoder-checkpoint-load-scope compatible_model`
- `--freeze-semantic-decoder`
- `--freeze-calculator-policy`
- `--share-calculator-output-proj`

Results:

- Final eval: `305/400 = 0.7625`.
- Step-600 normal: `0.7800`.
- Step-600 injection-zero: `0.0875`.
- Step-600 forced-random: `0.0725`.
- Step-600 oracle: `0.7800`.
- Step-600 calculator-result accuracy: `0.9950`.
- Step-600 hook normal: `0.7255/0.8095/0.8289/0.7604`.
- Final 128-sample counterfactuals: `0.1016` injection-zero, `0.0781`
  forced-random, `0.7422` oracle-at-eval.

## Continuation Diagnostic

Path:

```text
runs/2026-05-30_phase7_routed_multi_hook_training/op19_rhead64_topk8_unique24_hooks4_shareout_embd32_handoff_continue600_from_handoff600_cpu/2026-05-30_153624_227984_model-c-op0-19-fullgrid-hooks4-routeleft_operand_mod-adec-product/model-c-2digit-seed43
```

Continued from the handoff600 final checkpoint for another `600` steps.

- Final eval: `317/400 = 0.7925`.
- Step-600 continuation normal: `0.8050`.
- Step-600 continuation injection-zero: `0.0725`.
- Step-600 continuation forced-random: `0.0800`.
- Step-600 continuation calculator-result accuracy: `0.9950`.

## Interpretation

Shared output projection preserves four-hook routed source training and removes
the cloned-output parameter slope, but it does not preserve the known cloned
non-bottleneck handoff result under the trusted 600-step gate. The handoff
failure is not a calculator-policy failure: learned calculator-result accuracy
stays near perfect. It is a downstream/readout geometry issue caused by the
shared-output source representation.

Do not treat tied output projection as a complete many-calculator solution
until a handoff-aware source or readout mechanism clears the trusted additive
gate.
