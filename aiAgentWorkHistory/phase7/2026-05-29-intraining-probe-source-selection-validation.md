# 2026-05-29 In-Training Probe Source Selection Validation

## Aim

Run the new logging-only additive handoff probe on a real source-acquisition
lineage, then check whether the probe-selected checkpoint agrees with the
established standalone 600-step frozen-policy handoff gate.

## Setup

Source run:

```text
runs/2026-05-29_phase7_intraining_handoff_probe_source/src11_nodecay_probe500_steps800
```

The source recipe used the no-decay stabilization branch:

```text
entropy 0.05
batch diversity 0.1
hard improvement assignment weight 10
answer loss weight 0
800 source steps
additive handoff probe every 400 source steps
500 probe steps
400 probe samples
```

Source snapshots:

| Source step | Source normal/calc | Source injection-zero | Source oracle |
| ---: | ---: | ---: | ---: |
| `0` | `0.0125` | `0.0600` | `1.0000` |
| `200` | `0.3400` | `0.0225` | `1.0000` |
| `400` | `0.4250` | `0.0625` | `1.0000` |
| `600` | `0.7050` | `0.0350` | `1.0000` |
| `800` | `0.6600` | `0.0400` | `1.0000` |

Final source eval exact-match was `0.6850`.

## In-Training Probe Result

The in-training probe logged three source checkpoints:

| Source step | Probe normal @ 500 | Probe injection-zero | Probe oracle | Probe learned calc |
| ---: | ---: | ---: | ---: | ---: |
| `0` | `0.2850` | `0.0100` | `0.0700` | `0.0250` |
| `400` | `0.5625` | `0.0050` | `0.4350` | `0.4175` |
| `800` | `0.5525` | `0.0000` | `0.5350` | `0.7425` |

By 500-step normal exact-match, the probe selected source step `400` over
source step `800` by `0.0100`.

## Verification

Standalone 600-step frozen-policy additive handoffs were run from the selected
step `400` checkpoint and the runner-up step `800` checkpoint.

Run roots:

```text
runs/2026-05-29_phase7_intraining_handoff_probe_source/verify_step400_probe_selected_handoff600
runs/2026-05-29_phase7_intraining_handoff_probe_source/verify_step800_probe_runnerup_handoff600
```

Verification results:

| Source checkpoint | Standalone normal @ 500 | Standalone normal @ 600 | Final eval | Injection-zero @ 600 | Oracle @ 600 | Learned calc @ 600 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| step `400` | `0.5650` | `0.5975` | `0.6050` | `0.0100` | `0.4950` | `0.3900` |
| step `800` | `0.5975` | `0.6925` | `0.7075` | `0.0000` | `0.6500` | `0.6550` |

The verification gate rejected the 500-step in-training selector choice:
step `800` was materially better than step `400` under the standalone
600-step handoff.

## Decision

Label:

```text
intraining_500_probe_selector_mixed_negative
```

The in-training additive handoff probe is useful instrumentation and exposes
checkpoint-specific downstream learnability during source training, but the
500-step normal score is not yet a reliable selector for fresh source
lineages. Here it missed a stronger checkpoint because step `800` had better
oracle and learned-calculator transfer even though its 500-step normal score
was slightly lower in the embedded probe.

## Next Work

Use in-training probes for logging and triage, but verify candidate
checkpoints with a standalone 600-step handoff before selecting them as source
checkpoints. Next non-duplicative work should either:

- run the same in-training probe with 600 probe steps, if the compute budget is
  acceptable;
- log probe rows more frequently and select by 600-step or trend metrics;
- add a source-training term that optimizes handoff/readout geometry directly.

## Anti-Rerun Note

Do not repeat this same source step `400` versus step `800` comparison as
novelty. It already establishes that a 500-step in-training probe can be a
near-miss selector on a fresh no-decay source lineage and needs standalone
600-step confirmation.
