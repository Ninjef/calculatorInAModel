# 2026-05-29 Additive Handoff Geometry Probe

## Aim

Build a cheaper proxy for source-to-additive transfer that measures the
additive answer path more directly than frozen source-state linear
decodability.

The previous seed-10 checkpoint sweep showed that:

- source accuracy and learned calculator accuracy do not select transferable
  checkpoints;
- frozen-state linear sum probing is not a valid selector;
- the next proxy should measure additive learning slope or
  injection-to-answer geometry directly.

## Code

Added:

```text
scripts/run_additive_handoff_geometry_probe.py
```

The script loads a bottleneck source checkpoint in additive-compatible mode,
freezes the calculator policy, and reports:

- normal/zero/oracle full-grid answer loss;
- learned calculator-result accuracy;
- forced-result counterfactual geometry over every result class;
- short downstream-only learning slope for predeclared steps.

The main forced-result geometry metrics are:

- `forced_best_true_fraction`: fraction of prompts where the true sum class is
  the lowest-loss forced calculator result under the initial additive answer
  path.
- `forced_top3_true_fraction`: same, but true sum is in the top three.
- `forced_true_minus_best`: average loss gap between true forced result and
  the lowest-loss forced result.

## Runs

Primary seed-9/seed-10 sweep:

```text
runs/2026-05-29_phase7_additive_handoff_geometry_probe/seed9_seed10_sweep
```

Validation on older known lineages:

```text
runs/2026-05-29_phase7_additive_handoff_geometry_probe/src6_src7_validation
```

All probes used slope steps `0,50,100`.

## Result

Primary sweep:

| Source | Known handoff status | Calc | True-best | True top-3 | True-best gap | 100-step loss drop | 100-step loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| seed-9 final | positive after continuation/readout | `0.8675` | `0.0625` | `0.2125` | `0.0034` | `1.0730` | `1.5852` |
| seed-10 step `1000` | weak | `0.7550` | `0.0000` | `0.0300` | `0.0061` | `0.9837` | `1.6569` |
| seed-10 step `1300` | weak | `0.8575` | `0.0000` | `0.0300` | `0.0063` | `1.0388` | `1.5986` |
| seed-10 step `1400` | weak | `0.8475` | `0.0000` | `0.0300` | `0.0058` | `1.0417` | `1.5955` |
| seed-10 final | negative | `0.9025` | `0.0000` | `0.0450` | `0.0063` | `1.0127` | `1.6221` |

Older-lineage validation:

| Source | Known handoff status | Calc | True-best | True top-3 | True-best gap | 100-step loss drop | 100-step loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `src6` final | positive | `0.8450` | `0.0050` | `0.0550` | `0.0049` | `0.9182` | `1.7164` |
| `src7` step `1400` | boundary negative | `0.8000` | `0.0050` | `0.0950` | `0.0048` | `0.8221` | `1.8318` |

## Decision

Label:

```text
additive_handoff_geometry_probe_partial_no_selector
```

Forced-result geometry is a better warning signal than frozen-state linear
decodability for the seed-10 problem: all seed-10 checkpoints had
`forced_best_true_fraction=0.0`, low true top-3 rate, and worse
`forced_true_minus_best` than the seed-9 positive.

However, it is not a validated selector. The `src6` positive and `src7`
boundary-negative both had small nonzero true-best fractions, and 100-step
loss slope did not cleanly rank known pass/fail outcomes. This diagnostic can
reject obviously hostile source geometries, but it cannot replace the
400/600-step additive handoff probe.

## Interpretation

The useful signal is not "is the sum linearly present?" but "does the initial
additive answer path assign relatively low loss to the true forced calculator
result?" Seed-10 is unusually bad on that geometry despite high learned
calculator accuracy, which matches the later transfer failure.

The next source-acquisition improvement should optimize or monitor this
geometry during source training, but it should still be gated by actual
handoff/readout performance until the proxy is stronger.

## Anti-Rerun Note

Do not repeat this exact geometry probe over seed-9 final, seed-10
`1000/1300/1400/final`, `src6` final, or `src7` step `1400` as novelty.

Next useful tests:

- add the forced-result geometry metric as a source-training snapshot metric
  and test whether it predicts later handoff within a fresh source run;
- optimize source acquisition for `forced_true_minus_best` or true top-3 while
  preserving learned calculator accuracy;
- keep using actual 400/600-step handoff probes as the selection gate.

## Verification

`python3 -m py_compile scripts/run_additive_handoff_geometry_probe.py` passed.
Both probe runs completed and wrote `additive_handoff_geometry_summary.json`
plus `additive_handoff_slope_rows.csv`.
