# 2026-05-29 Short Slope Selector Validation

## Aim

Test whether a cheaper one/few-update downstream slope proxy can replace the
400/500/600-step additive handoff selector for choosing source checkpoints.

The previous additive geometry selector validation showed that forced-result
geometry alone is too coarse. This task keeps the same known `src2`, `src4`,
and `src5` source-family comparisons, but lets the additive downstream path
adapt for 25, 50, and 100 steps.

## Run

Run root:

```text
runs/2026-05-29_phase7_additive_handoff_slope_selector_validation/src2_src4_src5_known_handoffs_steps100
```

Command shape:

```text
python3 scripts/run_additive_handoff_geometry_probe.py \
  --checkpoint <src5 1100/1400/1500/final> \
              <src4 1000/1200/final> \
              <src2 1300/final> \
  --slope-steps 0,25,50,100 \
  --output-root runs/2026-05-29_phase7_additive_handoff_slope_selector_validation/src2_src4_src5_known_handoffs_steps100
```

This reused the additive-compatible frozen-policy downstream adaptation path
from `scripts/run_additive_handoff_geometry_probe.py`.

## Result

100-step loss-slope result:

| Source | Known handoff result | Winner? | 100-step loss | 100-step loss drop |
| --- | ---: | --- | ---: | ---: |
| `src5` step `1100` | `0.7950` full | yes | `1.7956` | `0.8238` |
| `src5` step `1400` | `0.6400` full | no | `1.7752` | `0.8464` |
| `src5` step `1500` | `0.6975` full | no | `1.7686` | `0.8535` |
| `src5` final | `0.5550` full | no | `1.7684` | `0.8549` |
| `src4` step `1000` | `0.5225` full | no | `1.7128` | `0.8984` |
| `src4` step `1200` | `0.7800` full | yes | `1.8478` | `0.7672` |
| `src4` final | no full comparison here | no | `1.8721` | `0.7493` |
| `src2` step `1300` | `0.8675` full | no | `1.6639` | `0.9513` |
| `src2` final | `0.9525` full | yes | `1.6424` | `0.9778` |

The 100-step loss proxy selects:

- `src5` final or step `1500`, not the known winner step `1100`;
- `src4` step `1000`, not the known winner step `1200`;
- `src2` final, matching the known winner.

Existing handoff exact-match traces also show that very early exact accuracy
is not enough. For example:

| Family | Known winner | Earliest exact-match point that selects winner |
| --- | --- | --- |
| `src5` | step `1100` | step `500` |
| `src4` | step `1200` | step `300` |
| `src2` | final | step `200` |

Step `300` and step `400` exact-match probes still misselect `src5` by
favoring step `1500`.

## Decision

Label:

```text
short_slope_selector_validation_negative
```

A 100-step downstream loss-slope proxy cannot replace the established
400/500/600-step handoff selector. It is especially misleading on `src5`, the
family where short exact-match probes already required roughly 500 steps to
select the known winner.

## Interpretation

The cost of reliable source selection is not just measuring whether the
downstream loss can begin falling. Many source checkpoints reduce loss quickly
for the first 100 steps, including checkpoints that later transfer worse. The
selector needs enough time for exact answer behavior and calculator dependence
to separate.

For now:

- keep 500/600-step handoff probes as source-selection gates;
- do not use 25/50/100-step loss slope as a selector;
- if optimizing source acquisition directly, use actual early handoff exact or
  a stronger learned proxy validated against the 500/600-step gate.

## Anti-Rerun Note

Do not repeat the 0/25/50/100-step loss-slope probe over this same
`src2/src4/src5` checkpoint set as novelty.

Next useful tests:

- optimize source training against a small number of actual handoff steps,
  then verify with the 500/600-step gate;
- add snapshot logging that records geometry and source metrics, but keep
  handoff probes for selection;
- validate whether a learned proxy trained on accumulated handoff traces can
  predict the 500/600-step selector.

## Verification

The slope probe completed and wrote:

```text
runs/2026-05-29_phase7_additive_handoff_slope_selector_validation/src2_src4_src5_known_handoffs_steps100/additive_handoff_geometry_summary.json
runs/2026-05-29_phase7_additive_handoff_slope_selector_validation/src2_src4_src5_known_handoffs_steps100/additive_handoff_slope_rows.csv
```
