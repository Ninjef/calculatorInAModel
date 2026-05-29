# 2026-05-29 Reduced Continuation Budget Validation

## Aim

Test whether the extra frozen-policy continuation in the selected-source recipe
can be reduced from 800 steps to 600 steps.

The current cheaper validated recipe is:

1. 500-step source-selection handoff probe.
2. 800-step frozen-policy continuation from the selected handoff.
3. 600-step stable-policy readout adaptation.

This task tests whether step 2 can be cut to 600 while keeping the 600-step
readout stage.

## Runs

Run root:

```text
runs/2026-05-29_phase7_reduced_continuation_budget_validation
```

Shared setup:

- Started from the selected 800-step frozen-policy handoff checkpoints:
  - `src4` step-1200/add2.
  - `src5` step-1100/add5.
- Ran a 600-step frozen-policy continuation with LR `3e-3`.
- Then ran a 600-step no-anchor readout adaptation with
  `--freeze-calculator-policy-backbone` and LR `3e-4`.
- Used additive, non-bottleneck result-space calculator mode throughout.
- Used exact-grid natural `0..19`, answer loss weight `1`, and frozen semantic
  decoder.

Cells:

| Cell | Run directory |
| --- | --- |
| `src4_continue600` | `runs/2026-05-29_phase7_reduced_continuation_budget_validation/source_seed4_step1200_additive_seed2_continue600_freeze_policy/2026-05-28_214952_376350_model-c-op0-19-fullgrid/model-c-2digit-seed6` |
| `src4_continue600_read600` | `runs/2026-05-29_phase7_reduced_continuation_budget_validation/source_seed4_step1200_additive_seed2_continue600_freeze_policy_backbone_steps600/2026-05-28_215905_458507_model-c-op0-19-fullgrid/model-c-2digit-seed6` |
| `src5_continue600` | `runs/2026-05-29_phase7_reduced_continuation_budget_validation/source_seed5_step1100_additive_seed5_continue600_freeze_policy/2026-05-28_214952_376082_model-c-op0-19-fullgrid/model-c-2digit-seed9` |
| `src5_continue600_read600` | `runs/2026-05-29_phase7_reduced_continuation_budget_validation/source_seed5_step1100_additive_seed5_continue600_freeze_policy_backbone_steps600/2026-05-28_215905_451490_model-c-op0-19-fullgrid/model-c-2digit-seed9` |

## Results

| Run | Continuation steps | Readout steps | Continuation final | Readout final | Readout best normal | Readout inj-zero | Readout forced-random | Readout oracle | Readout calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `src4` reduced | `600` | `600` | `0.7950` | `0.8750` | `0.8975` at `500` | `0.0000` | `0.0200` | `0.8625` | `0.8000` |
| `src4` reference | `800` | `600` | `0.8150` | `0.9025` | `0.9250` at `500` | `0.0025` | `0.0175` | `0.8625` | `0.8000` |
| `src5` reduced | `600` | `600` | `0.8850` | `0.9275` | `0.9525` at `600` | `0.0000` | `0.0275` | `0.9350` | `0.8250` |
| `src5` reference | `800` | `600` | `0.8800` | `0.9325` | `0.9525` at `600` | `0.0000` | `0.0250` | `0.9300` | `0.8250` |

## Conclusion

The 600-step continuation budget is source-sensitive and is not a safe
replacement for 800 steps in the current two-source validation:

- `src5` passes with little loss: `0.9275` versus the 800-continuation reference
  `0.9325`.
- `src4` fails the `0.90` final-eval gate: `0.8750` versus the 800-continuation
  reference `0.9025`.

The controls still show calculator dependence, so this is not a collapse into
the neural bypass. The issue is that the weaker `src4` selected lineage needs
the extra continuation depth before the reduced 600-step readout stage.

Label:

```text
reduced_continuation_budget_600_source_sensitive_negative
```

## Anti-Rerun Note

Do not repeat 600-step frozen-policy continuation plus 600-step no-anchor
policy-backbone-frozen readout from the same selected `src4` step-1200/add2 and
`src5` step-1100/add5 checkpoints as novelty.

Next useful tests should either keep the 800-step continuation for weak sources,
try an intermediate 700-step continuation if fine-grained cost tuning matters,
or move source acquisition toward improving continuation slope directly.
