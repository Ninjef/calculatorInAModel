# 2026-05-28 Online Shadow Feedback On-Policy Refresh Gate

## Question

Can periodic on-policy refresh turn the Stage 0B-passing online MLP shadow
module into useful Stage 1 learning?

## Implementation

- Added `--shadow-feedback-refresh-every`.
- Refactored online shadow calibration into a reusable Stage 1 helper.
- During Stage 1, if refresh cadence is enabled, the shadow module is refit
  against the current model before the configured training step.
- Saved `online_shadow_feedback_refresh_history.json`.
- Added CLI parsing coverage.

## Runs

Run root:

```text
runs/2026-05-28_phase7_online_shadow_feedback_on_policy_refresh_gate
```

Common configuration:

- model-c, natural `0..19`, exact-grid batch.
- frozen product semantic decoder.
- h32 validation-gradient online shadow module.
- validation-gradient loss weight `0.5`, norm weight `0.1`.
- `shadow_feedback_weight=1.0`.
- refresh every `50` steps.
- no apply norm clamp.
- 200 steps, snapshots every `25`.

Refresh history:

| Step | Heldout result/upstream cosine | Train-heldout gap |
| ---: | ---: | ---: |
| `0` | `0.8068 / 0.8083` | `0.1227 / 0.1343` |
| `50` | `0.9820 / 1.0000` | `0.0034 / 0.0000` |
| `100` | `0.9971 / 0.9999` | `0.0029 / 0.0001` |
| `150` | `0.9978 / 0.9991` | `0.0013 / 0.0008` |
| `200` | `0.9716 / 0.9997` | `0.0017 / 0.0001` |

Stage 1 result:

| Metric | Value |
| --- | ---: |
| final exact match | `0.025` |
| best snapshot exact match | `0.0475` |
| final shadow feedback norm | `27.627` |

## Conclusion

```text
online_mlp_shadow_feedback_on_policy_refresh_alignment_pass_stage1_negative
```

Refresh solves the stale-gradient-agreement problem but not calculator
discovery. Stage 1 remains in a single-result collapse regime.

## Anti-Regression Note

Do not repeat h32 validation-gradient on-policy refresh every `50` steps with
`shadow_feedback_weight=1.0`, no apply clamp, and 200-step budget as novelty.
The next branch should constrain training dynamics directly.
