# 2026-05-29 Source Assignment Weight-5 Transfer Probe

## Aim

Test whether the hostile seed-10 no-decay source failure came from an overly
strong hard improvement-assignment weight. The previous seed-10 source learned
a strong bottleneck calculator policy, but its final and earlier checkpoints
transferred poorly to the additive non-bottleneck path.

This run lowers `result_policy_improvement_assignment_weight` from `10` to `5`
while keeping entropy `0.05`, batch diversity `0.1`, no decay, the frozen
product semantic decoder, and exact-grid natural result-space source training.

## Source Run

Run root:

```text
runs/2026-05-29_phase7_source_acquisition_stabilization_floor_weight_sweep/src10_entropy0p05_div0p1_improve5_nodecay_steps1600
```

Saved source run:

```text
runs/2026-05-29_phase7_source_acquisition_stabilization_floor_weight_sweep/src10_entropy0p05_div0p1_improve5_nodecay_steps1600/2026-05-29_044707_001906_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed12
```

The CLI used `--seed 10`; the script stores `seed=args.seed+num_digits`, so
the saved directory is `seed12`.

## Source Result

| Step | Source normal | Injection-zero | Oracle | Learned calc |
| --- | ---: | ---: | ---: | ---: |
| `1000` | `0.5475` | `0.0325` | `1.0000` | `0.5475` |
| `1200` | `0.7800` | `0.0575` | `1.0000` | `0.7800` |
| `1300` | `0.7725` | `0.0375` | `1.0000` | `0.7725` |
| `1400` | `0.7625` | `0.0625` | `1.0000` | `0.7625` |
| `1500` | `0.7825` | `0.0300` | `1.0000` | `0.7825` |
| `1600` | `0.6900` | `0.0750` | `1.0000` | `0.6900` |
| final eval | `0.6750` | n/a | n/a | n/a |

Lowering the assignment weight made the source less stable and weaker than the
previous weight-10 seed-10 source, which reached final eval `0.9000` and
learned calc `0.8984`.

## Additive Handoff Probe

I ran 600-step frozen-policy additive handoffs from the best source snapshot
region and from the final checkpoint.

| Source checkpoint | 600-step handoff snapshot | Final eval | Injection-zero at step 600 | Oracle at step 600 | Calc at step 600 |
| --- | ---: | ---: | ---: | ---: | ---: |
| step `1200` | `0.3425` | `0.3000` | `0.0250` | `0.3575` | `0.7725` |
| final | `0.2475` | `0.2325` | `0.0325` | `0.2325` | `0.6600` |

The strongest lower-weight checkpoint did not beat the prior weight-10
seed-10 checkpoint sweep. The previous weight-10 seed-10 step `1000`
checkpoint reached `0.4475` at the 600-step handoff snapshot and `0.4450`
final eval, while the final weight-10 checkpoint reached `0.3375` at the
600-step snapshot and `0.3275` final eval.

## Decision

Label:

```text
source_improvement_weight5_transfer_negative
```

Lowering hard improvement-assignment weight from `10` to `5` does not fix the
seed-10 transfer-hostile source geometry. It weakens source acquisition and
does not improve the bounded additive handoff probe.

## Interpretation

The seed-10 failure is not explained by assignment weight `10` simply being too
forceful. A gentler weight produced a less reliable bottleneck source, and the
best available checkpoint still transferred worse than the earlier weight-10
step `1000` checkpoint. Source acquisition should not proceed by just sweeping
lower assignment weights.

The useful next direction is to optimize or select sources using actual
500/600-step handoff behavior, or to add source-training terms that directly
reward handoff/readout geometry while preserving calculator correctness.

## Anti-Rerun Note

Do not repeat this exact seed-10 no-decay entropy `0.05`, batch diversity
`0.1`, improvement weight `5`, 1600-step source run or the step-1200/final
600-step frozen-policy additive handoffs as novelty.

Next useful tests:

- optimize source acquisition against actual early additive handoff exact;
- add a handoff/readout-geometry source-training term and gate it with the
  500/600-step handoff probe;
- train a learned selector from accumulated handoff traces, but keep the
  500/600-step gate until that selector is validated.

## Verification

The source run and both 600-step handoff probes completed and wrote metrics
under the run root above.
