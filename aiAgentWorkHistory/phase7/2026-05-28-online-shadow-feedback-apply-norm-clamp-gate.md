# 2026-05-28 Online Shadow Feedback Apply-Norm Clamp Gate

## Question

Did fixed online MLP shadow feedback fail Stage 1 mainly because the predicted
feedback norm exploded?

## Implementation

- Added `--shadow-feedback-apply-max-norm`.
- Added max-norm scaling inside `online_shadow_feedback_fixed_module_loss`.
- Stage 1 metrics now include:
  - `shadow_feedback_apply_max_norm`;
  - `shadow_feedback_apply_norm_scale`;
  - `shadow_feedback_unclamped_predicted_l2`;
  - applied `shadow_feedback_predicted_l2`.
- Added CLI parsing coverage.

## Runs

Run root:

```text
runs/2026-05-28_phase7_online_shadow_feedback_apply_norm_clamp_gate
```

Common configuration:

- model-c, natural `0..19`, exact-grid batch.
- frozen product semantic decoder.
- fixed online MLP shadow module calibrated with validation-gradient loss
  weight `0.5` and norm weight `0.1`.
- `shadow_feedback_weight=1.0`.
- `answer_loss_weight=0`.
- 200 steps, snapshots every `25`.

Results:

| Apply max norm | Final exact match | Best snapshot exact | Final applied norm | Final unclamped norm |
| ---: | ---: | ---: | ---: | ---: |
| `3.5` | `0.075` | `0.0525` | `3.5` | `77802.8` |
| `10` | `0.075` | `0.0525` | `10.0` | `79123.9` |

## Conclusion

```text
online_mlp_shadow_feedback_apply_norm_clamp_stage1_negative
```

Simple L2 output clamping is not enough. It prevents the applied feedback norm
from exploding, but Stage 1 remains below the `0.16` output-projection
feedback baseline.

## Anti-Regression Note

Do not repeat fixed-module h32 validation-gradient Stage 1 with plain apply
L2 clamps `3.5` or `10` as novelty. The next branch should refresh the module
on-policy or gate updates using refreshed model-gradient agreement.
