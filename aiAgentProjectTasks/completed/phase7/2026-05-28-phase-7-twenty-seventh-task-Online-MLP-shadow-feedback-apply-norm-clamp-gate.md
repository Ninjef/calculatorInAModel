# Phase 7 Twenty-Seventh Task: Online MLP Shadow Feedback Apply-Norm Clamp Gate

## Purpose

Test whether the fixed online MLP shadow-feedback Stage 1 failure is mainly a
feedback norm explosion problem.

## Setup

- Base task: natural `0..19` exact-grid, model-c, seed `2` CLI / effective
  seed `4`.
- Decoder: frozen product semantic decoder from the Phase 6 sum-only oracle
  checkpoint.
- Shadow mode: `online_mlp`.
- Calibrated module: h32 direct validation-gradient module from the Stage 0B
  pass.
- Shadow weight: `1.0`.
- Apply max norms: `3.5`, `10`.
- Training: 200-step early-lift smoke, snapshots every `25`.

## Results

| Apply max norm | Final exact match | Best snapshot exact | Final applied norm | Final unclamped norm |
| ---: | ---: | ---: | ---: | ---: |
| `3.5` | `0.075` | `0.0525` | `3.5` | `77802.8` |
| `10` | `0.075` | `0.0525` | `10.0` | `79123.9` |

## Conclusion

```text
online_mlp_shadow_feedback_apply_norm_clamp_stage1_negative
```

The clamp prevents applied feedback norm blow-up, but it does not improve
Stage 1 learning. The fixed module's direction still goes stale as model
features move.

## Next

Do not rerun simple apply L2 clamps `3.5` or `10` as novelty. Next work should
refresh the shadow module on-policy or use a trust-region criterion based on
refreshed gradient agreement.
