# 2026-05-30 - Routed embd32 Source630 Trusted Handoff

## Question

After fixing multi-hook injection-zero controls, does the stronger routed
`embd32` source630 checkpoint clear the trusted 600-step frozen-policy additive
handoff gate?

## Run

Path:

```text
runs/2026-05-30_phase7_routed_multi_hook_training/op19_rhead64_topk8_unique24_hooks2_cloneout_embd32_handoff600_from_source630_fixedzero_cpu/2026-05-30_144532_185971_model-c-op0-19-fullgrid-hooks2-routeleft_operand_mod-adec-product/model-c-2digit-seed43
```

Source checkpoint:

```text
runs/2026-05-30_phase7_routed_multi_hook_training/op19_rhead64_topk8_unique24_hooks2_cloneout_embd32_source630_cpu/2026-05-30_140522_171215_model-c-op0-19-fullgrid-direct_feedback_alignment-hooks2-routeleft_operand_mod-answer_decoder-adec-product/model-c-2digit-seed43/checkpoint_snapshots/step_00630_weights.pt
```

Key settings:

- `--calculator-estimator ste`
- `--calculator-bottleneck-mode none`
- `--answer-loss-weight 1`
- `--semantic-decoder-checkpoint-load-scope compatible_model`
- `--freeze-semantic-decoder`
- `--freeze-calculator-policy`
- `--calculator-hook-count 2`
- `--calculator-hook-routing left_operand_mod`
- `--calculator-result-head-hidden-size 64`
- `600` handoff steps

## Results

Final eval:

- Exact match: `400/400 = 1.0000`
- Final loss: `9.58e-08`

Step-600 snapshot:

- Normal exact match: `1.0000`
- Injection-zero exact match: `0.0550`
- Forced-random exact match: `0.0300`
- Forced-zero exact match: `0.0100`
- Oracle exact match: `1.0000`
- Calculator-result accuracy: `0.9975`
- Hook 0 calculator-result accuracy: `1.0000`
- Hook 1 calculator-result accuracy: `0.9955`
- Route distribution: `{"0": 178, "1": 222}`

Final 128-sample counterfactuals:

- Injection-zero exact match: `0.078125`
- Forced-random exact match: `0.0234375`
- Forced-zero exact match: `0.015625`
- Oracle-at-eval exact match: `1.0000`

## Interpretation

This is the first corrected-control two-hook routed non-bottleneck staged
transfer positive. The routed sparse-assignment source is no longer source-only:
after transfer, both active hooks remain accurate and the additive model fails
when all calculator hooks are ablated.

The result is still not the final thesis answer. It uses hard improvement
assignment, frozen transfer, cloned per-hook output projections, a pretrained
product decoder, and one op19 seed. The next meaningful scaling axes are a
fresh routed seed, more routes/hooks with active-hook cost accounting, or a
shared/tied output projection that removes per-hook semantic-output growth.
