# 2026-05-29 Short Handoff Probe Audit

## Goal

Find a cheaper source-quality signal after the `src2` source-selection
replication disproved source normal/calculator accuracy as a reliable
checkpoint selector.

## Periodic Review

The recent ledger entries ruled out repeating:

- frozen-policy 800-step transfer matrix cells as novelty;
- `src5` step-1500 selected-source handoff as novelty;
- `src2` step-1300 versus final selected-source replication as novelty.

The allowed direction was to measure handoff geometry directly. Existing
800-step transfer traces already contain intermediate checkpoints, so I audited
those traces instead of launching redundant short runs.

## Data

Audited diagnostic snapshots from:

- the original frozen-policy replication cells;
- the weak-source downstream continuation cells;
- the `src5` selected-source handoff;
- the `src2` selected-source and final-source control handoffs.

## Results

For the non-continued 800-step frozen-policy transfer cells:

| Cell | Normal @ 200 | Normal @ 400 | Normal @ 600 | Normal @ 800 | Final eval | Final calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `src2_final_add4` | `0.1250` | `0.5700` | `0.8025` | `0.9325` | `0.9525` | `0.9150` |
| `src2_step1300_add4` | `0.1100` | `0.4100` | `0.7200` | `0.8875` | `0.8675` | `0.9250` |
| `src4_final_add2` | `0.1375` | `0.2400` | `0.2675` | `0.3150` | `0.3025` | `0.8725` |
| `src4_final_add4` | `0.1175` | `0.2375` | `0.2825` | `0.2950` | `0.3375` | `0.8575` |
| `src5_final_add5` | `0.0300` | `0.2975` | `0.4525` | `0.5725` | `0.5550` | `0.8000` |
| `src5_step1500_add5` | `0.0450` | `0.4325` | `0.5250` | `0.6925` | `0.6975` | `0.9000` |

Pearson correlations between early normal accuracy and final eval across these
six non-continued cells:

| Probe | Correlation with final eval |
| --- | ---: |
| normal @ 200 | `-0.0959` |
| normal @ 400 | `0.9374` |
| normal @ 600 | `0.9935` |
| normal @ 800 | `0.9960` |

The step-400/600 signal also catches the `src2` selector failure: the higher
source-accuracy step-1300 checkpoint was already behind the final-source
control by step 400 (`0.4100` versus `0.5700`) and step 600 (`0.7200` versus
`0.8025`).

## Conclusion

Label:

```text
bottleneck_to_additive_short_handoff_probe_partial
```

Short additive handoff progress is a better source-quality signal than source
normal/calculator accuracy in the current data. Step 400 is already useful, and
step 600 is very predictive across the audited cells.

This is still not a full scalable selector. It requires running a partial
additive handoff, and the sample is small. But it gives a concrete probe for
handoff geometry: a source checkpoint is not just good if it asks the calculator
for the right result; it is good if the additive post-hook path can quickly
learn to use that result.

## Anti-Regression Note

Do not repeat this exact trace audit as novelty. Next useful tests are:

- use a 400- or 600-step handoff probe to select among multiple source
  checkpoints, then confirm only the selected source with a full transfer;
- build a cheaper readout/linear probe that predicts the same step-400/600
  handoff signal without running hundreds of downstream steps;
- optimize source acquisition for early additive handoff slope, not source
  action accuracy alone.

## Verification

No code changed and no new training was needed. The audit parsed existing
`diagnostic_snapshots.csv` and `metrics.json` files from the run roots listed
above.
