# 2026-05-29 Source Acquisition Stabilization Floor

## Aim

Follow up the decay-to-zero stabilization negative by keeping the same
entropy/diversity/improvement-assignment source objective active through the
whole run. This tests whether the previous collapse was caused by removing the
source objective, and whether the resulting source is useful for additive
non-bottleneck handoff.

## Source Acquisition

Run root:

```text
runs/2026-05-29_phase7_source_acquisition_stabilization_floor
```

Source cell:

```text
src9_entropy0p05_div0p1_nodecay_steps1600
```

Saved source run:

```text
runs/2026-05-29_phase7_source_acquisition_stabilization_floor/src9_entropy0p05_div0p1_nodecay_steps1600/2026-05-29_012919_931445_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed11
```

The CLI used `--seed 9`; the script stores `seed=args.seed+num_digits`, so the
saved run directory is `seed11`.

Configuration:

- current bottleneck source recipe with frozen product semantic decoder
- `result_policy_improvement_assignment_weight=10`
- `result_policy_entropy_weight=0.05`
- `result_policy_batch_diversity_weight=0.1`
- `result_policy_stabilization_decay_steps=0`
- exact-grid natural `0..19`
- 1600 source steps with 100-step snapshots

## Source Curve

| Step | Source normal | Injection-zero | Oracle | Learned calc |
| --- | ---: | ---: | ---: | ---: |
| `700` | `0.6400` | `0.0475` | `1.0000` | `0.6400` |
| `1000` | `0.7500` | `0.0475` | `1.0000` | `0.7500` |
| `1200` | `0.8250` | `0.0525` | `1.0000` | `0.8250` |
| `1400` | `0.9100` | `0.0650` | `1.0000` | `0.9100` |
| `1500` | `0.8800` | `0.0400` | `1.0000` | `0.8800` |
| `1600` | `0.8650` | `0.0400` | `1.0000` | `0.8650` |
| final eval | `0.8575` | `0.0078` | `1.0000` | `0.8750` |

Keeping the stabilization objective active prevents the prior collapse. The
source reaches a strong bottleneck snapshot (`0.9100`) and remains high at the
end (`0.8575` final eval).

## Additive Handoff Candidates

Both handoffs used additive non-bottleneck mode, compatible checkpoint load,
`--freeze-calculator-policy`, exact-grid natural `0..19`, CLI seed `9`, and
800 training steps.

| Candidate | Source normal | Normal @ 600 | Normal @ 800 | Final eval | Final injection-zero | Final forced-random | Final oracle | Final calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| step `1400` | `0.9100` | `0.4575` | `0.5200` | `0.4425` | `0.0078` | `0.0625` | `0.5234` | `0.8828` |
| final | `0.8575` | `0.5250` | `0.7025` | `0.6500` | `0.0000` | `0.0859` | `0.7422` | `0.8750` |

The final-source checkpoint transfers better than the higher-source-normal
step `1400` checkpoint. But even the better final-source handoff is weak:
`0.6500` final eval is below the recent weak `src7` selected handoff
(`0.7325`), whose continuation/readout chain still missed the `0.90` gate.

## Decision

Label:

```text
source_acquisition_entropy_diversity_nodecay_source_positive_transfer_negative
```

No downstream continuation/readout was run. The handoff signal is too weak to
justify spending the full recipe, and the more important result is already
resolved: persistent entropy/diversity source stabilization improves bottleneck
source accuracy but does not produce handoff-friendly source geometry.

## Interpretation

The decay-to-zero collapse was caused by removing the only active
result-policy source objective; keeping it on fixes that failure mode. But the
no-decay objective appears to bias the source toward a high-accuracy protocol
that the additive non-bottleneck downstream path cannot easily read out.

This strengthens the current Phase 7 direction: source acquisition should be
optimized for 600-step handoff or continuation slope, not just bottleneck
source accuracy or source-policy diversity.

## Anti-Rerun Note

Do not repeat this exact no-decay source-only recipe plus step `1400`/final
additive seed-9 handoff comparison as novelty.

Next useful tests:

- add a handoff/continuation proxy to the source-acquisition objective;
- test a source-policy anchor/floor that preserves handoff-friendly geometry,
  not merely source accuracy;
- inspect why the high-source-normal step `1400` has worse additive transfer
  than final despite slightly better bottleneck accuracy.

## Verification

The source run and both handoff candidates completed and wrote metrics under
the run root above. No code changed.
