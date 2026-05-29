# Phase 7 Fifty-Ninth Task - Frozen-State Readout Probe Script

## Task

Make the frozen-state readout probe reusable and validate it against the known
source checkpoints.

## Done

- Added `scripts/run_frozen_state_readout_probe.py`.
- Added focused tests for exact-grid construction, additive-compatible loading,
  feature extraction, and collision-safe output directories.
- Re-ran the probe on the five known source checkpoints.
- Corrected the earlier scratch-probe conclusion after finding the hardcoded
  token id bug.
- Recorded the result in the Phase 7 work history, fact sheet, and hypothesis
  ledger.

## Result

Decision:

```text
bottleneck_to_additive_frozen_state_readout_probe_negative
```

The corrected safe probes did not predict handoff quality well. Best safe probe
correlation with known final handoff was only `0.2865`; `src4_final` scored
near the strong `src2` sources despite poor additive transfer. The reusable
script is useful, but simple frozen-state linear separability is not enough.
