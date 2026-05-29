# 2026-05-29 `src2` 500-Step Selector Validation

## Aim

Validate the shortened 500-step handoff selector on a source family not used in
the `src4/src5` shorter-probe trace audit.

The useful `src2` case is a known source-accuracy counterexample: source step
`1300` had higher source normal/calculator accuracy than the final checkpoint,
but transferred worse into additive seed `4`.

## Source Traces

Run root:

```text
runs/2026-05-29_phase7_source_checkpoint_selection_replication
```

Compared candidates:

| Candidate | Source normal/calc | Final handoff |
| --- | ---: | ---: |
| step `1300` | `0.9475` | `0.8675` |
| final / step `1600` | `0.9150` | `0.9525` |

## Results

| Candidate | Normal @ 400 | Normal @ 500 | Normal @ 600 | Normal @ 800 | Final handoff |
| --- | ---: | ---: | ---: | ---: | ---: |
| step `1300` | `0.4100` | `0.5875` | `0.7200` | `0.8875` | `0.8675` |
| final / step `1600` | `0.5700` | `0.6900` | `0.8025` | `0.9325` | `0.9525` |

## Conclusion

The 500-step handoff selector validates on `src2`: it picks the known better
final checkpoint over the source-accuracy-favored step `1300`. In fact, this
case is easy enough that the 400-step probe already selects the right
checkpoint, but `src5` remains the reason not to generalize the selector all
the way down to 400 steps.

Across audited families:

- `src2`: 500-step selector picks final / step `1600`, matching the winner.
- `src4`: 500-step selector picks step `1200`, matching the 600-step selector.
- `src5`: 500-step selector picks step `1100`, matching the 600-step selector.

Label:

```text
src2_500_step_selector_validation_positive
```

## Anti-Rerun Note

Do not repeat this same trace audit over `src2` step-1300 versus final additive
seed-4 handoffs as novelty.

Next useful tests should validate 500-step selection on newly acquired source
checkpoints, or replace post-hoc selection with source acquisition optimized
for early handoff/continuation slope.
