# Phase 7 Twenty-Eighth Task: Online MLP Shadow Feedback On-Policy Refresh Gate

## Purpose

Test whether periodically refreshing the online MLP shadow module against the
current model rescues the Stage 1 failure caused by stale fixed feedback.

## Setup

- Base task: natural `0..19` exact-grid, model-c, seed `2` CLI / effective
  seed `4`.
- Decoder: frozen product semantic decoder from the Phase 6 sum-only oracle
  checkpoint.
- Shadow mode: `online_mlp`.
- Calibrated module: h32 direct validation-gradient module.
- Shadow weight: `1.0`.
- Refresh cadence: every `50` training steps.
- Apply max norm: disabled.
- Training: 200-step early-lift smoke, snapshots every `25`.

## Refresh Results

| Step | Heldout result/upstream cosine | Train-heldout gap |
| ---: | ---: | ---: |
| `0` | `0.8068 / 0.8083` | `0.1227 / 0.1343` |
| `50` | `0.9820 / 1.0000` | `0.0034 / 0.0000` |
| `100` | `0.9971 / 0.9999` | `0.0029 / 0.0001` |
| `150` | `0.9978 / 0.9991` | `0.0013 / 0.0008` |
| `200` | `0.9716 / 0.9997` | `0.0017 / 0.0001` |

## Stage 1 Results

| Metric | Value |
| --- | ---: |
| final exact match | `0.025` |
| best snapshot exact match | `0.0475` |
| final shadow feedback norm | `27.627` |

## Conclusion

```text
online_mlp_shadow_feedback_on_policy_refresh_alignment_pass_stage1_negative
```

Periodic refresh restores excellent current-model gradient agreement, but
still does not produce calculator-result discovery. The model remains in a
single-result collapse regime.

## Next

Do not rerun refresh every `50` with this same h32 module and weight `1.0` as
novelty. Next work should add training-dynamics constraints: step-level trust
region, entropy/diversity stabilization, or a target/state that avoids
single-result collapse.
