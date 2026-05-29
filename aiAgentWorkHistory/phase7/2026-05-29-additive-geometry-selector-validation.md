# 2026-05-29 Additive Geometry Selector Validation

## Aim

Test whether the additive handoff geometry probe is good enough to select
source checkpoints within source families that already have known additive
handoff outcomes.

The prior geometry probe showed that forced-result geometry flags the
seed-10 no-decay stabilized lineage as unusually hostile, but it did not prove
the metric can replace the actual handoff selector. This task validates it
against existing `src2`, `src4`, and `src5` checkpoint comparisons without
rerunning the handoff training.

## Run

Run root:

```text
runs/2026-05-29_phase7_additive_handoff_geometry_selector_validation/src2_src4_src5_known_handoffs
```

Command shape:

```text
python3 scripts/run_additive_handoff_geometry_probe.py \
  --checkpoint <src5 1100/1400/1500/final> \
              <src4 1000/1200/final> \
              <src2 1300/final> \
  --slope-steps 0 \
  --output-root runs/2026-05-29_phase7_additive_handoff_geometry_selector_validation/src2_src4_src5_known_handoffs
```

The probe used no adaptation steps. This was a pure source-checkpoint
geometry scan.

## Result

| Source | Known handoff result | Winner? | Calc | True-best | True top-3 | True-best gap |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `src5` step `1100` | `0.7950` full | yes | `0.8200` | `0.0325` | `0.1375` | `0.0038` |
| `src5` step `1400` | `0.6400` full | no | `0.8875` | `0.0075` | `0.1350` | `0.0037` |
| `src5` step `1500` | `0.6975` full | no | `0.9050` | `0.0075` | `0.1325` | `0.0035` |
| `src5` final | `0.5550` full | no | `0.8250` | `0.0125` | `0.1350` | `0.0033` |
| `src4` step `1000` | `0.5225` full | no | `0.8100` | `0.0600` | `0.1350` | `0.0036` |
| `src4` step `1200` | `0.7800` full | yes | `0.8075` | `0.0600` | `0.1325` | `0.0035` |
| `src4` final | no full comparison here | no | `0.8450` | `0.0600` | `0.1100` | `0.0030` |
| `src2` step `1300` | `0.8675` full | no | `0.9200` | `0.0325` | `0.0350` | `0.0042` |
| `src2` final | `0.9525` full | yes | `0.9025` | `0.0325` | `0.0350` | `0.0039` |

## Decision

Label:

```text
additive_geometry_selector_validation_negative
```

The geometry probe is not a reliable source-checkpoint selector.

It partially identifies the `src5` winner through `true_best` because step
`1100` has `0.0325` while the others are `0.0075-0.0125`. But it fails or
ties on the other families:

- `src4` winner step `1200` ties step `1000` and final on `true_best`.
- `src2` final beats step `1300` in handoff, but the geometry metrics tie or
  nearly tie.
- `true_best_gap` would select `src5` final and `src4` final, not the known
  handoff winners.

## Interpretation

Forced-result geometry is useful as a warning metric for seed-10-like hostile
geometry, but it is too coarse as a selector inside normal source families. It
should not be optimized as the sole source objective yet, because the metric
can favor checkpoints that do not transfer best.

For now:

- keep actual 400/600-step additive handoff probes as selection gates;
- use geometry only as a cheap diagnostic;
- if turning geometry into a source-training auxiliary, pair it with a real
  handoff-slope validation gate.

## Anti-Rerun Note

Do not repeat this geometry scan over the same `src2`, `src4`, and `src5`
checkpoints as novelty.

Next useful tests:

- optimize source acquisition for early handoff slope directly;
- add the geometry metric as logging only during fresh source acquisition;
- design a stronger proxy that uses gradients or one/few downstream updates,
  not only forced-result loss ranking at initialization.

## Verification

The probe completed and wrote:

```text
runs/2026-05-29_phase7_additive_handoff_geometry_selector_validation/src2_src4_src5_known_handoffs/additive_handoff_geometry_summary.json
runs/2026-05-29_phase7_additive_handoff_geometry_selector_validation/src2_src4_src5_known_handoffs/additive_handoff_slope_rows.csv
```
