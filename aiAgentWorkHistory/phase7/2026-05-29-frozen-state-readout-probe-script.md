# 2026-05-29 Frozen-State Readout Probe Script

## Goal

Turn the one-off frozen-state readout probe into a reusable diagnostic and
validate it against the known source checkpoints.

## Code Changes

Added:

```text
scripts/run_frozen_state_readout_probe.py
```

The script:

- loads one or more source checkpoints;
- optionally converts bottleneck checkpoints into additive-compatible models by
  loading compatible tensors only;
- runs the exact operand grid once with diagnostics;
- trains small linear sum probes on non-answer frozen features;
- writes per-checkpoint CSV/JSON summaries;
- uses hash-suffixed output directories to avoid collisions between checkpoint
  snapshots with the same parent directory name.

Focused tests cover exact-grid construction, additive-compatible loading,
operand-pair feature extraction, and collision-safe output directories.

## Validation Run

Run root:

```text
runs/2026-05-29_phase7_frozen_state_readout_probe_script_validation_v2
```

Command shape:

```text
python3 scripts/run_frozen_state_readout_probe.py \
  --checkpoint <src2_final> <src2_step1300> <src5_step1500> <src5_final> <src4_final> \
  --features read_eq read_pair layer1_pair layer2_pair \
  --output-root runs/2026-05-29_phase7_frozen_state_readout_probe_script_validation_v2
```

## Results

| Source | Known final additive handoff | Read-`=` | Read-pair | Layer-1 pair | Layer-2 pair | Best safe probe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `src2_final` | `0.9525` | `0.0250` | `0.5125` | `0.5125` | `0.5500` | `0.5500` |
| `src2_step1300` | `0.8675` | `0.0250` | `0.5000` | `0.5000` | `0.5000` | `0.5000` |
| `src5_step1500` | `0.6975` | `0.1625` | `0.3375` | `0.3375` | `0.3375` | `0.3375` |
| `src5_final` | `0.5550` | `0.1750` | `0.3250` | `0.3250` | `0.3375` | `0.3375` |
| `src4_final` | `0.3025` | `0.0125` | `0.5000` | `0.5000` | `0.5000` | `0.5000` |

Correlation with known final additive handoff:

| Probe | Correlation |
| --- | ---: |
| read-`=` | `-0.1218` |
| read-pair | `0.2118` |
| layer-1 pair | `0.2118` |
| layer-2 pair | `0.2865` |
| best safe probe | `0.2865` |

## Conclusion

Label:

```text
bottleneck_to_additive_frozen_state_readout_probe_negative
```

The reusable script invalidated the earlier scratch positive. The scratch
command had used the wrong token id for `=`, selecting a wrong/leaky position.
Correct safe features are not reliable handoff-quality predictors in this
sample.

The script is still useful infrastructure for future diagnostics, but source
selection should continue to use the 400/600-step handoff probe until a cheaper
non-leaky geometry metric is proven.

## Verification

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/run_frozen_state_readout_probe.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q -k "frozen_state_probe"
```

Result: `4 passed, 110 deselected`.
