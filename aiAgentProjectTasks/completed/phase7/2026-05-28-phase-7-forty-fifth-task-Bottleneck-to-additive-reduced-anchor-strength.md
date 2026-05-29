# Phase 7 Forty-Fifth Task: Bottleneck-to-Additive Reduced Anchor Strength

## Status

Completed on 2026-05-28.

## Question

Does full-policy non-bottleneck unfreezing require the large KL anchor weight
`10`, or can much weaker constant anchors preserve calculator use?

## Setup

- Start from the adapted weak-source additive checkpoints.
- Load full model weights with `--semantic-decoder-checkpoint-load-scope full_model`.
- Remove `--freeze-calculator-policy`.
- Use LR `3e-4`, answer loss weight `1`, exact-grid natural `0..19`.
- Use `--result-policy-anchor-mode kl`.
- Test constant anchor weights `1.0` and `0.1`.
- Train for `400` steps.

## Runs

```text
runs/2026-05-28_phase7_bottleneck_to_additive_transfer_policy_anchor_strength_unfreeze/source_seed4_additive_seed2_adapted_unfreeze_lr3e-4_anchor1_steps400
runs/2026-05-28_phase7_bottleneck_to_additive_transfer_policy_anchor_strength_unfreeze/source_seed5_additive_seed5_adapted_unfreeze_lr3e-4_anchor1_steps400
runs/2026-05-28_phase7_bottleneck_to_additive_transfer_policy_anchor_strength_unfreeze/source_seed4_additive_seed2_adapted_unfreeze_lr3e-4_anchor0.1_steps400
runs/2026-05-28_phase7_bottleneck_to_additive_transfer_policy_anchor_strength_unfreeze/source_seed5_additive_seed5_adapted_unfreeze_lr3e-4_anchor0.1_steps400
```

## Result

| Run | Final eval | Best normal | Final calc | Final injection-zero | Final anchor agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| `src4_add2` anchor `1.0` | `0.7775` | `0.7550` at `400` | `0.8050` | `0.0225` | `0.9625` |
| `src5_add5` anchor `1.0` | `0.9925` | `0.9825` at `400` | `0.7925` | `0.0075` | `0.9625` |
| `src4_add2` anchor `0.1` | `0.8325` | `0.8275` at `400` | `0.8075` | `0.0250` | `0.9225` |
| `src5_add5` anchor `0.1` | `0.9750` | `0.9700` at `400` | `0.7725` | `0.0000` | `0.9075` |

## Conclusion

Label:

```text
bottleneck_to_additive_reduced_anchor_strength_partial
```

A large anchor is not required for this two-cell handoff gate. Even anchor
weight `0.1` preserved useful calculator policies and low injection-zero,
while improving final eval over the original anchor-10 runs.

This is still a staged and actively anchored method. Do not claim from-scratch
or anchor-free success from it. Next work can test even weaker/floored/gated
anchors, selective unfreezing, or source-policy acquisition that needs less
retention scaffolding.
