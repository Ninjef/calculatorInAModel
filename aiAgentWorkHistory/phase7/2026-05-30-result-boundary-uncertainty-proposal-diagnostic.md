# 2026-05-30 Result-Boundary Uncertainty Proposal Diagnostic

## Question

Can the answer-derived result-boundary critic be useful as an adaptive
candidate proposer, even though direct hidden/output critic argmin prediction
failed?

This is a different gate from the previous amortized-critic diagnostic. The
critic does not need to pick the exact best result directly; it proposes a small
candidate set, then the diagnostic asks whether actually scoring only that set
would recover the full-enumeration best result.

## Tooling

Extended:

```text
scripts/diagnose_result_boundary_amortized_critic.py
```

Added:

- `--ensemble-size`
- `--uncertainty-candidates`
- `--uncertainty-beta`
- mean-proposal metrics
- lower-confidence-bound proposal metrics, using `mean - beta * std`
- scored-subset metrics: whether the best actually scored candidate equals the
  full-enum best / true sum, plus regret

The diagnostic still computes full forced-result losses only for evaluation.

## Runs

Smoke:

```text
runs/2026-05-30_phase7_result_boundary_uncertainty_proposal/smoke.json
```

Primary three-checkpoint gate:

```text
runs/2026-05-30_phase7_result_boundary_uncertainty_proposal/uncertainty_lcb_k8_pairwise_ensemble4_step0_100_800.json
```

Settings:

- known May 13 result-boundary source lineage
- checkpoints: step `0`, step `100`, step `800`
- heldout prompts: `100`
- train prompts: `300`
- pairwise critic
- `8` sparse forced scores per train prompt per ensemble member
- hidden size `128`
- `600` epochs
- four-member ensemble
- top-8 candidate proposal
- `beta=1.0`

Budget sanity checks at step `800`:

```text
runs/2026-05-30_phase7_result_boundary_uncertainty_proposal/uncertainty_lcb_k16_pairwise_ensemble4_step800.json
runs/2026-05-30_phase7_result_boundary_uncertainty_proposal/single_pairwise_k16_step800.json
runs/2026-05-30_phase7_result_boundary_uncertainty_proposal/single_pairwise_k8_step800.json
```

## Results

The full-enum best was the true sum on all heldout prompts in every run.

Direct critic argmin remained weak:

| Setup | Checkpoint | Heldout argmin = full best | Top-5 contains full best |
| --- | --- | ---: | ---: |
| ensemble pairwise, top-8 run | step `0` | `0.0500` | `0.2900` |
| ensemble pairwise, top-8 run | step `100` | `0.0600` | `0.3500` |
| ensemble pairwise, top-8 run | step `800` | `0.2400` | `0.6700` |
| single pairwise | step `800` | `0.2000` | `0.5900` |

Proposal-plus-rescoring was much stronger:

| Setup | Proposed candidates | Train scores/prompt | Scored best = full best | Mean regret |
| --- | ---: | ---: | ---: | ---: |
| single pairwise step `800` | `8/39` | `8` | `0.7900` | `0.6707` |
| single pairwise step `800` | `16/39` | `8` | `0.9600` | `0.1162` |
| ensemble pairwise step `0` mean proposal | `8/39` | `32` | `0.4900` | `1.7071` |
| ensemble pairwise step `100` mean proposal | `8/39` | `32` | `0.4900` | `1.6299` |
| ensemble pairwise step `800` mean proposal | `8/39` | `32` | `0.8400` | `0.5032` |
| ensemble pairwise step `800` mean proposal | `16/39` | `32` | `1.0000` | `0.0000` |

LCB uncertainty did not improve over mean proposal:

| Setup | Proposed candidates | LCB scored best = full best | Mean-proposal scored best = full best |
| --- | ---: | ---: | ---: |
| ensemble step `0` | `8/39` | `0.4700` | `0.4900` |
| ensemble step `100` | `8/39` | `0.5300` | `0.4900` |
| ensemble step `800` | `8/39` | `0.7900` | `0.8400` |
| ensemble step `800` | `16/39` | `0.9800` | `1.0000` |

## Interpretation

This is a mixed result.

Positive:

- The old direct-argmin metric understated what the critic can do as a broad
  proposal mechanism.
- Scoring only the single-critic top-8 candidates recovers the full-enum target
  on `79%` of heldout prompts at the trained step-800 checkpoint.
- Top-16 proposal rescoring can get close to exact target recovery.

Negative:

- Top-16 already scores `16/39 = 41%` of the result vocabulary.
- The four-member ensemble uses `32` sparse forced scores per train prompt,
  close to full enumeration on the `39`-class result vocabulary.
- LCB uncertainty did not provide the hoped adaptive-compute advantage over the
  mean prediction.
- The gate is static/fixed-grid; it does not yet prove streaming or evolving
  checkpoint generalization.

## Decision

```text
result_boundary_proposal_rescoring_mixed_not_solved
```

Do not wire this static proposal count directly into source training as a
solved scalable target. The next useful result-boundary work needs a changed
mechanism: adaptive stopping/calibration that expands compute only where the
critic is uncertain, set/soft targets that tolerate missing the exact argmin,
or validation across evolving checkpoints before training integration.
