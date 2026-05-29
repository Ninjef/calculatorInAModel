# 2026-05-29 Reduced Readout Budget Validation

## Aim

Reduce the expensive final readout-adaptation part of the selected-source
continuation recipe.

Prior fair-continuation runs used 1600 steps of no-anchor stable-policy readout
adaptation after the extra frozen-policy continuation. Snapshot curves suggested
that much less might be enough, especially after the continued handoff was
already strong.

## Runs

Run root:

```text
runs/2026-05-29_phase7_reduced_readout_budget_validation
```

Shared configuration:

- Started from the selected-source continued frozen-policy checkpoints:
  - `src4` step-1200/add2 continued checkpoint.
  - `src5` step-1100/add5 continued checkpoint.
- Loaded full model checkpoints.
- Used `--freeze-semantic-decoder`.
- Used `--freeze-calculator-policy-backbone`.
- Used no result-policy anchor.
- LR `3e-4`, answer loss weight `1`, exact-grid natural `0..19`.
- Tested 200-step and 600-step readout budgets.

Cells:

| Cell | Run directory |
| --- | --- |
| `src4_200` | `runs/2026-05-29_phase7_reduced_readout_budget_validation/source_seed4_step1200_additive_seed2_continued_freeze_policy_backbone_steps200/2026-05-28_212953_598617_model-c-op0-19-fullgrid/model-c-2digit-seed6` |
| `src4_600` | `runs/2026-05-29_phase7_reduced_readout_budget_validation/source_seed4_step1200_additive_seed2_continued_freeze_policy_backbone_steps600/2026-05-28_213417_809826_model-c-op0-19-fullgrid/model-c-2digit-seed6` |
| `src5_200` | `runs/2026-05-29_phase7_reduced_readout_budget_validation/source_seed5_step1100_additive_seed5_continued_freeze_policy_backbone_steps200/2026-05-28_212953_605303_model-c-op0-19-fullgrid/model-c-2digit-seed9` |
| `src5_600` | `runs/2026-05-29_phase7_reduced_readout_budget_validation/source_seed5_step1100_additive_seed5_continued_freeze_policy_backbone_steps600/2026-05-28_213417_810305_model-c-op0-19-fullgrid/model-c-2digit-seed9` |

## Results

| Run | Readout steps | Final eval | Best normal | Last injection-zero | Last forced-random | Last oracle | Last calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `src4_200` | `200` | `0.8775` | `0.9025` at `200` | `0.0000` | `0.0225` | `0.8400` | `0.8050` |
| `src4_600` | `600` | `0.9025` | `0.9250` at `500` | `0.0025` | `0.0175` | `0.8625` | `0.8000` |
| `src4_1600` reference | `1600` | `0.9125` | `0.9475` at `1400` | `0.0075` | `0.0250` | `0.8825` | `0.8025` |
| `src5_200` | `200` | `0.9275` | `0.9325` at `100` | `0.0000` | `0.0075` | `0.9450` | `0.8000` |
| `src5_600` | `600` | `0.9325` | `0.9525` at `600` | `0.0000` | `0.0250` | `0.9300` | `0.8250` |
| `src5_1600` reference | `1600` | `0.9425` | `0.9625` at `800` | `0.0000` | `0.0225` | `0.9600` | `0.8275` |

## Conclusion

The 200-step readout budget is source-sensitive. It works for the easier
selected `src5` lineage (`0.9275`) but misses the `0.90` final-eval gate for
selected `src4` (`0.8775`), even though the 200-step diagnostic snapshot was
`0.9025`.

The 600-step readout budget passes both selected lineages while retaining
calculator dependence under controls:

- `src4`: `0.9025` final eval, injection-zero `0.0025`,
  forced-random `0.0175`.
- `src5`: `0.9325` final eval, injection-zero `0.0000`,
  forced-random `0.0250`.

This reduces the stable-policy readout-adaptation stage from 1600 to 600 steps
for the current selected-source continuation recipe. The reduced budget loses
some ceiling (`src4` `0.9125 -> 0.9025`, `src5` `0.9425 -> 0.9325`) but keeps
the calculator-dependent non-bottleneck handoff above `0.90`.

Label:

```text
reduced_readout_budget_600_positive_200_mixed
```

## Anti-Rerun Note

Do not repeat 200-step or 600-step no-anchor policy-backbone-frozen readout
adaptation from the same continued selected `src4` step-1200/add2 and `src5`
step-1100/add5 checkpoints as novelty.

Next useful tests should reduce the upstream 600-step handoff probe or the
extra 800-step frozen-policy continuation, or replace post-hoc selection with
source acquisition that optimizes early handoff/continuation slope directly.
