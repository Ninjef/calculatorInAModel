# Phase 7 Forty-Eighth Task: Bottleneck-to-Additive Freeze Action Head

## Status

Completed on 2026-05-28.

## Question

Can selective unfreezing preserve the transferred non-bottleneck calculator
policy by freezing only the calculator action head?

## Code Change

Added `--freeze-calculator-action-head` to `scripts/overfit_one_batch.py`.
For result-space policies, this freezes `calculator_hook.result_proj` while
leaving surrounding model parameters trainable.

## Setup

- Start from the adapted weak-source additive checkpoints.
- Load full model weights with `--semantic-decoder-checkpoint-load-scope full_model`.
- Remove `--freeze-calculator-policy`.
- Add `--freeze-calculator-action-head`.
- Use no result-policy anchor.
- Use LR `3e-4`, answer loss weight `1`, exact-grid natural `0..19`.
- Train for `400` steps.

## Runs

```text
runs/2026-05-28_phase7_bottleneck_to_additive_selective_unfreeze/source_seed4_additive_seed2_adapted_unfreeze_lr3e-4_freeze_action_head_steps400
runs/2026-05-28_phase7_bottleneck_to_additive_selective_unfreeze/source_seed5_additive_seed5_adapted_unfreeze_lr3e-4_freeze_action_head_steps400
```

## Result

| Run | Final eval | Best normal | Final calc | Final injection-zero | Trainable groups |
| --- | ---: | ---: | ---: | ---: | --- |
| `src4_add2` freeze action head | `0.5200` | `0.6500` at `0` | `0.3000` | `0.0225` | `upstream` |
| `src5_add5` freeze action head | `0.8100` | `0.8325` at `0` | `0.2525` | `0.0225` | `upstream` |

## Conclusion

Label:

```text
bottleneck_to_additive_freeze_action_head_unfreeze_negative
```

Freezing only the result-space action head does not preserve the transferred
policy. Because only the upstream group was trainable, the failure shows that
upstream representation drift alone can collapse calculator-result accuracy.

Do not repeat this exact selective-freeze gate as novelty. Future selective
unfreezing should either protect the full policy path, use behavior-level
anchoring/gating, or target downstream-only adaptation.
