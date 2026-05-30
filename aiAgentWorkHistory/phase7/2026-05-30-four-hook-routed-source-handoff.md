# 2026-05-30 - Four-Hook Routed Source and Handoff

## Question

Does the corrected-control routed recipe still train and transfer when the
task is partitioned across four calculator hooks instead of two?

This is a many-calculator scaling gate, not a seed tweak. Each hook receives
only the examples routed by the final left-operand digit modulo `4`.

## Source Run

Path:

```text
runs/2026-05-30_phase7_routed_multi_hook_training/op19_rhead64_topk8_unique24_hooks4_cloneout_embd32_source630_fixedzero_cpu/2026-05-30_145203_849432_model-c-op0-19-fullgrid-direct_feedback_alignment-hooks4-routeleft_operand_mod-answer_decoder-adec-product/model-c-2digit-seed43
```

Key settings:

- `--calculator-hook-count 4`
- `--calculator-hook-routing left_operand_mod`
- `--clone-primary-calculator-output-proj`
- `--calculator-result-head-hidden-size 64`
- `--result-policy-improvement-assignment-policy-topk-count 8`
- `--result-policy-improvement-assignment-sample-count 24`
- `--result-policy-improvement-assignment-unique-sampling`
- `--late-source-recovery-start-step 600`

Source results:

- Final eval: `398/400 = 0.9950`
- Step-630 normal: `0.9975`
- Step-630 injection-zero: `0.0275`
- Step-630 forced-random: `0.0225`
- Step-630 oracle: `1.0000`
- Step-630 hook calc: `0.9928/1.0000/1.0000/1.0000`
- Step-630 route distribution: `{"0": 138, "1": 109, "2": 84, "3": 69}`
- Final 128-sample counterfactuals: `0.0391` injection-zero, `0.0156`
  forced-random.

The route split is intentionally not perfectly balanced because final decimal
digits modulo `4` map digits `0,4,8` and `1,5,9` to larger buckets than
`2,6` and `3,7` over operand range `0..19`.

## Handoff Run

Path:

```text
runs/2026-05-30_phase7_routed_multi_hook_training/op19_rhead64_topk8_unique24_hooks4_cloneout_embd32_handoff600_from_source630_fixedzero_cpu/2026-05-30_150246_867985_model-c-op0-19-fullgrid-hooks4-routeleft_operand_mod-adec-product/model-c-2digit-seed43
```

Key settings:

- Trusted `600`-step frozen-policy additive handoff.
- `--calculator-estimator ste`
- `--calculator-bottleneck-mode none`
- `--answer-loss-weight 1`
- `--semantic-decoder-checkpoint-load-scope compatible_model`
- `--freeze-semantic-decoder`
- `--freeze-calculator-policy`
- `--calculator-hook-count 4`
- `--calculator-hook-routing left_operand_mod`

Handoff results:

- Final eval: `400/400 = 1.0000`
- Step-600 normal: `1.0000`
- Step-600 injection-zero: `0.0400`
- Step-600 forced-random: `0.0200`
- Step-600 forced-zero: `0.0025`
- Step-600 oracle: `1.0000`
- Step-600 calculator-result accuracy: `1.0000`
- Step-600 hook calc: `1.0000/1.0000/1.0000/1.0000`
- Final 128-sample counterfactuals: `0.0625` injection-zero, `0.0156`
  forced-random.

## Interpretation

This is the first more-than-two-hook routed non-bottleneck positive. The result
directly advances the many-calculator axis: four independently parameterized
routed hooks can train from the sparse topk assignment source and survive the
trusted additive handoff with corrected all-hook controls.

It does not yet prove efficient scaling. The current implementation still calls
every hook before route masking, and the semantic output projection is cloned
per hook. The next high-leverage work should implement active-only routed hook
execution and/or shared/tied output projection, then repeat this four-hook gate
with explicit compute and parameter accounting.
