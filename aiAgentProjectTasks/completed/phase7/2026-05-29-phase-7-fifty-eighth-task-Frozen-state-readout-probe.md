# Phase 7 Fifty-Eighth Task - Frozen-State Readout Probe

## Task

Test a cheaper readout/linear proxy for handoff geometry after the short
handoff probe audit showed that 400/600-step downstream progress predicts final
handoff.

## Done

- Loaded known source checkpoints into additive-compatible models.
- Extracted frozen `=` residual states on the exact grid.
- Trained tiny linear sum probes with a deterministic `320/80` split.
- Recorded the result in the Phase 7 work history, fact sheet, and hypothesis
  ledger.

## Result

Decision:

```text
bottleneck_to_additive_frozen_state_readout_probe_negative
```

The original scratch result was invalid: it used the wrong token id for `=`,
selecting a wrong/leaky position. The reusable script validation corrected this
and found that safe non-answer probes do not reliably predict handoff quality.
Best safe probe correlation with known final handoff was only `0.2865`.
