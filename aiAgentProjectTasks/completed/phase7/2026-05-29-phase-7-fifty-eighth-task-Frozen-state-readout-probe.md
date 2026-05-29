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
bottleneck_to_additive_frozen_state_readout_probe_partial
```

The clean `=` residual probe correlated with known final additive handoff at
about `0.96` across five source checkpoints and correctly ranked
`src2_final` above the misleading higher-source-accuracy `src2_step1300`
checkpoint. It is promising but needs validation on unseen checkpoints.
