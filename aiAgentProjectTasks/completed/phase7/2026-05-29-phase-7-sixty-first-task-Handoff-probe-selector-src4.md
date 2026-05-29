# Phase 7 Sixty-First Task - Handoff Probe Selector on src4

## Task

Apply the 600-step handoff-probe selector to the weak `src4` source and confirm
whether it can find a better handoff checkpoint than the final source.

## Done

- Reproduced `src4` with checkpoint snapshots.
- Ran 600-step handoff probes for step `1000` and step `1200`.
- Confirmed the selected step `1200` source with a full 800-step frozen-policy
  additive transfer.
- Recorded the result in the Phase 7 work history, fact sheet, and hypothesis
  ledger.

## Result

Decision:

```text
bottleneck_to_additive_handoff_probe_selector_src4_positive
```

The selector picked step `1200`, which had lower source accuracy (`0.7550`) than
the final source (`0.8700`) but far better additive handoff: `0.7800` versus
the old final-source `0.3025`, and above the old continued final-source
baseline `0.6050`.
