# 2026-05-29 Handoff Probe Selector Validation

## Goal

Validate the short additive handoff probe as a source checkpoint selector on
candidate checkpoints not already selected by source normal/calculator
accuracy.

## Periodic Review

The current ledger rules out:

- selecting source checkpoints by highest source normal/calculator accuracy;
- repeating the `src5` step-1500 selected-source handoff as novelty;
- replacing the short handoff probe with the corrected frozen-state linear
  readout probe, which failed.

The allowed next test was to use the 400/600-step handoff probe to select among
candidate source checkpoints, then confirm the selected source with a full
transfer.

## Runs

Run root:

```text
runs/2026-05-29_phase7_handoff_probe_selector_validation
```

Candidates:

| Candidate | Source normal/calc | Prior status |
| --- | ---: | --- |
| `src5` step `1100` | `0.8400` | unseen handoff candidate |
| `src5` step `1400` | `0.8900` | unseen handoff candidate |
| `src5` step `1500` | `0.9200` | previous source-accuracy-selected handoff |
| `src5` final / step `1600` | `0.8325` | previous final-source baseline |

All transfer cells used:

- additive path, `calculator_bottleneck_mode=none`;
- `calculator_estimator=ste`;
- `calculator_action_head=result_space`;
- compatible checkpoint load from the bottleneck source;
- `--freeze-calculator-policy`;
- answer loss weight `1`;
- exact-grid natural `0..19`;
- additive CLI seed `5`.

## Results

| Source checkpoint | Source normal | Normal @ 400 | Normal @ 600 | Normal @ 800 | Injection-zero @ end | Oracle @ end | Calc @ end | Final eval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| final / step `1600` | `0.8325` | `0.2975` | `0.4525` | `0.5725` | `0.0125` | `0.5600` | `0.8000` | `0.5550` |
| step `1500` | `0.9200` | `0.4325` | `0.5250` | `0.6925` | `0.0150` | `0.7025` | `0.9000` | `0.6975` |
| step `1400` probe | `0.8900` | `0.4150` | `0.6625` | n/a | `0.0000` | `0.7050` | `0.8950` | `0.6400` |
| step `1100` probe | `0.8400` | `0.3475` | `0.6825` | n/a | `0.0000` | `0.7650` | `0.8325` | `0.6850` |
| step `1100` full | `0.8400` | `0.3475` | `0.6825` | `0.7775` | `0.0000` | `0.8625` | `0.8250` | `0.7950` |

## Conclusion

Label:

```text
bottleneck_to_additive_handoff_probe_selector_positive
```

The 600-step handoff probe successfully selected a lower-source-accuracy
checkpoint (`src5` step `1100`, source normal `0.8400`) that transferred better
than the previous source-accuracy-selected checkpoint (`src5` step `1500`,
source normal `0.9200`): full transfer improved from `0.6975` to `0.7950`.

The 400-step probe was not sufficient in this case: step `1500` led at 400
steps (`0.4325` versus `0.3475`), but step `1100` led by 600 steps and won the
full confirmation. This suggests that the selector should use the 600-step
handoff score, not the earlier 400-step score, for these weak-source
candidates.

The result is still not a final scalable recipe because it requires partial
downstream training. It is, however, a better validated source-selection method
than source normal/calculator accuracy and a practical target for future
cheaper proxy development.

## Anti-Regression Note

Do not repeat the same `src5` step `1100/1400/1500/final` additive seed `5`
handoff-probe comparison as novelty. Next useful tests are:

- use the 600-step handoff probe to select among newly acquired source
  checkpoints;
- reduce the probe cost or approximate it with a validated non-leaky proxy;
- optimize source acquisition directly for 600-step handoff score.

## Verification

No code changed in this task. The probe and confirmation runs completed and
wrote metrics under the run root above.
