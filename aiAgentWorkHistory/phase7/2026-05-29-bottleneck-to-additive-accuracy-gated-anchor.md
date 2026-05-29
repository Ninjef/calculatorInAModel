# 2026-05-29 Bottleneck-to-Additive Accuracy-Gated Anchor

## Aim

Test whether behavior-gated anchoring improves when the gate watches current
calculator-result accuracy instead of source-policy agreement.

## Runs

Run root:

```text
runs/2026-05-29_phase7_bottleneck_to_additive_transfer_policy_anchor_accuracy_gated_unfreeze
```

Shared configuration:

- `model-c`, 2-digit exact-grid natural `0..19`.
- `calculator_action_head=result_space`.
- `calculator_bottleneck_mode=none`.
- Continued from adapted weak-source frozen-policy handoff checkpoints.
- Full model checkpoint load, no frozen policy.
- LR `3e-4`, 400 steps, snapshots every 50.
- Base result-policy KL anchor `0.01`.
- Gate metric `current_argmax_accuracy`.
- Gate weight `0.1`.

Specific cells:

| Cell | Threshold | Run directory |
| --- | ---: | --- |
| `src4_add2` | `0.80` | `runs/2026-05-29_phase7_bottleneck_to_additive_transfer_policy_anchor_accuracy_gated_unfreeze/source_seed4_additive_seed2_adapted_unfreeze_lr3e-4_anchor0.01_calcacc_gate0.8_to0.1_steps400/2026-05-28_185433_780653_model-c-op0-19-fullgrid/model-c-2digit-seed4` |
| `src5_add5` | `0.80` | `runs/2026-05-29_phase7_bottleneck_to_additive_transfer_policy_anchor_accuracy_gated_unfreeze/source_seed5_additive_seed5_adapted_unfreeze_lr3e-4_anchor0.01_calcacc_gate0.8_to0.1_steps400/2026-05-28_185433_780697_model-c-op0-19-fullgrid/model-c-2digit-seed7` |
| `src4_add2` | `0.82` | `runs/2026-05-29_phase7_bottleneck_to_additive_transfer_policy_anchor_accuracy_gated_unfreeze/source_seed4_additive_seed2_adapted_unfreeze_lr3e-4_anchor0.01_calcacc_gate0.82_to0.1_steps400/2026-05-28_185727_077810_model-c-op0-19-fullgrid/model-c-2digit-seed4` |
| `src5_add5` | `0.82` | `runs/2026-05-29_phase7_bottleneck_to_additive_transfer_policy_anchor_accuracy_gated_unfreeze/source_seed5_additive_seed5_adapted_unfreeze_lr3e-4_anchor0.01_calcacc_gate0.82_to0.1_steps400/2026-05-28_185727_077759_model-c-op0-19-fullgrid/model-c-2digit-seed7` |

## Results

| Run | Final eval | Best normal | Last normal | Last inj-zero | Last forced-random | Last oracle | Last calc | Last anchor weight | Final agreement | Final anchor acc | Gate active rows | Mean effective weight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `src4_accgate0.80` | `0.7725` | `0.7975` at `400` | `0.7975` | `0.0500` | `0.1150` | `0.8200` | `0.8100` | `0.0100` | `0.8525` | `0.8200` | `0/9` | `0.0100` |
| `src5_accgate0.80` | `0.9825` | `0.9625` at `400` | `0.9625` | `0.0000` | `0.0550` | `0.9175` | `0.7900` | `0.1000` | `0.9125` | `0.7875` | `8/9` | `0.0900` |
| `src4_accgate0.82` | `0.7900` | `0.8075` at `150` | `0.7975` | `0.0775` | `0.1075` | `0.8500` | `0.8000` | `0.0100` | `0.9025` | `0.8250` | `4/9` | `0.0500` |
| `src5_accgate0.82` | `0.9825` | `0.9675` at `400` | `0.9675` | `0.0000` | `0.0525` | `0.9175` | `0.7900` | `0.1000` | `0.9200` | `0.7925` | `8/9` | `0.0900` |

## Conclusion

The accuracy gate is mechanically useful, but simple discrete thresholds
`0.80` and `0.82` are not a better recipe than fixed anchor `0.1`. The `src5`
cell slightly improved final answer accuracy, but `src4` stayed below the
fixed-anchor result even when the gate activated intermittently.

Label:

```text
bottleneck_to_additive_accuracy_gated_anchor_mixed_no_go
```

## Anti-Rerun Note

Do not repeat base anchor `0.01`, gate weight `0.1`, thresholds `0.80` or
`0.82`, metric `current_argmax_accuracy`, LR `3e-4`, 400-step unfreeze from
the same adapted `src4_add2/src5_add5` checkpoints as novelty.
