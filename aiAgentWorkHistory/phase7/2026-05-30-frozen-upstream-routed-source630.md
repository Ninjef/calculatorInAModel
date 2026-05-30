# 2026-05-30 - Frozen-Upstream Routed Source630 Gate

## Question

Was the low injection-zero in the frozen-upstream routed source200 run a real
anti-leak effect, or just a side effect of undertraining?

## Run

Path:

```text
runs/2026-05-30_phase7_routed_multi_hook_training/op19_rhead64_topk8_unique24_hooks2_cloneout_embd32_freezeup_source630_cpu/2026-05-30_142020_017308_model-c-op0-19-fullgrid-direct_feedback_alignment-hooks2-routeleft_operand_mod-answer_decoder-adec-product/model-c-2digit-seed43
```

Key settings:

- `--calculator-hook-count 2`
- `--calculator-hook-routing left_operand_mod`
- `--clone-primary-calculator-output-proj`
- `--freeze-semantic-decoder`
- `--freeze-upstream-encoder`
- `--result-policy-improvement-assignment-policy-topk-count 8`
- `--result-policy-improvement-assignment-sample-count 24`
- `--result-policy-improvement-assignment-unique-sampling`
- `--late-source-recovery-start-step 600`

## Results

Final eval:

- Exact match: `379/400 = 0.9475`
- Final loss: `14.1456`

Last 400-sample snapshot at step `630`:

- Normal exact match: `0.9750`
- Injection-zero exact match: `0.4400`
- Oracle exact match: `1.0000`
- Forced-zero exact match: `0.0025`
- Forced-random exact match: `0.0200`
- Hook 0 calculator-result accuracy: `0.9955`
- Hook 1 calculator-result accuracy: `0.9494`
- Route distribution: `{"0": 222, "1": 178}`

Final 128-sample counterfactuals:

- Injection-zero exact match: `0.5000`
- Oracle-at-eval exact match: `1.0000`
- Forced-zero exact match: `0.0078`
- Forced-random exact match: `0.0000`

Final 128-sample routed summary:

- Hook 0 calculator-result accuracy: `0.9286`
- Hook 1 calculator-result accuracy: `0.9444`
- Route distribution: `{"0": 56, "1": 72}`

## Interpretation

Longer frozen-upstream training recovered routed source learning, including both
active hooks. However, injection-zero rose with source learning and ended close
to the open-upstream routed leak level. The source200 frozen-upstream run was
low-leak mainly because it was undertrained, not because this recipe solved
causal calculator acquisition.

Do not repeat the same frozen-upstream source630 recipe as the anti-leak fix.
The next routed work should add source-time anti-leak pressure, source ablation
controls, a stricter bottlenecked route, or a shared/tied output-projection
design that is validated by low injection-zero before handoff.
