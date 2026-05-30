# 2026-05-30 - Shared-output confound audit

## Question

Was the shared-output routed handoff miss caused by shared output projection
itself, or by a mismatched source-training schedule?

This is a high-leverage audit rather than a knob sweep: the prior shared-output
result directly affects the many-calculator parameter-scaling claim.

## Finding

The first cloned-vs-shared A/B was not perfectly matched:

- cloned-output positive: `additive_forced_margin_start_step=50`
- shared-output miss: `additive_forced_margin_start_step=0`

All other scalar settings checked were either the intended output-interface
change or matched.

## Matched Source

Path:

```text
runs/2026-05-30_phase7_routed_multi_hook_training/op19_rhead64_topk8_unique24_hooks4_shareout_embd32_source630_start50_cpu/2026-05-30_154953_427794_model-c-op0-19-fullgrid-direct_feedback_alignment-hooks4-routeleft_operand_mod-answer_decoder-adec-product/model-c-2digit-seed45
```

Result:

- Final eval: `399/400 = 0.9975`
- Step-630 normal: `0.9950`
- Step-630 injection-zero: `0.0425`
- Step-630 forced-random: `0.0275`
- Final 128-sample injection-zero: `0.0547`
- Diagnostic calculator-result accuracy: `0.9922`
- Hook calculator-result accuracy:
  `0.9697/1.0000/1.0000/1.0000`

## Matched Handoff

Path:

```text
runs/2026-05-30_phase7_routed_multi_hook_training/op19_rhead64_topk8_unique24_hooks4_shareout_embd32_handoff600_start50_from_source630_matchedhead_cpu/2026-05-30_155644_455733_model-c-op0-19-fullgrid-hooks4-routeleft_operand_mod-adec-product/model-c-2digit-seed45
```

Result:

- Final eval: `299/400 = 0.7475`
- Step-600 normal: `0.7225`
- Step-600 injection-zero: `0.0875`
- Step-600 forced-random: `0.0725`
- Step-600 oracle: `0.7325`
- Step-600 calculator-result accuracy: `0.9900`
- Step-600 hook normal:
  `0.6339/0.7744/0.6892/0.7901`
- Final 128-sample injection-zero: `0.0938`
- Final 128-sample forced-random: `0.0703`

An earlier attempted handoff from the matched source omitted
`--calculator-result-head-hidden-size 64`. That made the compatible load skip
the source policy head; it is invalid and should not be used as evidence.

## Interpretation

The schedule mismatch was real, but it does not explain the shared-output
handoff failure. With the cloned positive's delayed margin restored, tied
output projection still trains a strong routed source and still misses the
trusted non-bottleneck handoff while retaining near-perfect calculator-result
accuracy.

Do not rerun this same matched source/handoff path as novelty. Continuing the
shared-output branch needs a new transfer-geometry mechanism, not another
same-recipe comparison.
