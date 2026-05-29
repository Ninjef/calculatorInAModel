# Phase 7 Fifty-Sixth Task - Source Selection Metric Replication

## Task

Replicate source checkpoint selection on the strong `src2` source and test
whether source normal/calculator accuracy alone is a reliable handoff selector.

## Done

- Reproduced the `src2` bottleneck source with checkpoint snapshots.
- Selected the source step-1300 checkpoint by highest source diagnostic
  accuracy (`0.9475`).
- Transferred both the selected checkpoint and the reproduced final checkpoint
  into additive seed `4` with frozen calculator policy.
- Recorded the result in the Phase 7 work history, fact sheet, and hypothesis
  ledger.

## Result

Decision:

```text
bottleneck_to_additive_source_accuracy_selector_negative
```

The step-1300 selected checkpoint had better source accuracy (`0.9475`) than
the final checkpoint (`0.9150`) but transferred worse: `0.8675` versus
`0.9525` final additive eval. Source checkpoint selection matters, but source
accuracy alone is not a reliable selector.
