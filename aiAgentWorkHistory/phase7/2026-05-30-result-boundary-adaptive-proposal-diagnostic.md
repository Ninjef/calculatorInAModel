# 2026-05-30 Result-Boundary Adaptive Proposal Diagnostic

## Question

Can a result-boundary critic allocate extra forced-result scoring only to
uncertain prompts, instead of scoring a broad fixed top-16 candidate set for
every prompt?

The previous proposal-rescoring diagnostic found that fixed top-16 proposals
can nearly recover the full-enum boundary target, but that scoring `16/39`
results per prompt is still expensive. This diagnostic tests whether adaptive
expansion from top-8 to top-16 can keep most of the target recovery while
reducing the average scored candidates.

## Tooling

Extended:

```text
scripts/diagnose_result_boundary_amortized_critic.py
```

Added:

- `--adaptive-base-candidates`
- `--adaptive-expanded-candidates`
- `--adaptive-expand-fraction`
- adaptive proposal metrics for:
  - cutoff-margin expansion,
  - ensemble/std expansion,
  - LCB-intrusion expansion,
  - random expansion baseline.

Each adaptive metric ranks heldout prompts by a predicted uncertainty score,
expands only the selected fraction from top-8 to top-16, then asks whether
actually scoring that variable-size candidate set recovers the full-enum best.

## Runs

Single-critic step-800 checks:

```text
runs/2026-05-30_phase7_result_boundary_adaptive_proposal/single_pairwise_adaptive8to16_f25_step800.json
runs/2026-05-30_phase7_result_boundary_adaptive_proposal/single_pairwise_adaptive8to16_f50_step800.json
```

Ensemble step-800 checks:

```text
runs/2026-05-30_phase7_result_boundary_adaptive_proposal/ensemble4_pairwise_adaptive8to16_f25_step800.json
runs/2026-05-30_phase7_result_boundary_adaptive_proposal/ensemble4_pairwise_adaptive8to16_f50_step800.json
```

Common settings:

- May 13 result-boundary source lineage, step `800`
- heldout prompts: `100`
- train prompts: `300`
- pairwise critic
- `8` sparse forced scores per train prompt per ensemble member
- hidden size `128`
- `600` epochs
- adaptive base: top `8/39`
- adaptive expanded: top `16/39`

## Results

Baselines from the same runs:

| Setup | Fixed top-8 scored best = full best | Fixed top-16 scored best = full best |
| --- | ---: | ---: |
| single critic | `0.7900` | `0.9600` |
| four-critic ensemble | `0.8400` | `1.0000` |

Adaptive expansion:

| Setup | Expanded prompts | Mean candidates | Margin adaptive | Random adaptive |
| --- | ---: | ---: | ---: | ---: |
| single critic | `25%` | `10/39` | `0.8500` | `0.8200` |
| single critic | `50%` | `12/39` | `0.9200` | `0.8800` |
| four-critic ensemble | `25%` | `10/39` | `0.9100` | `0.8800` |
| four-critic ensemble | `50%` | `12/39` | `0.9700` | `0.9100` |

Other uncertainty scores were weaker than cutoff margin:

| Setup | Expanded prompts | LCB intrusion | Std expansion |
| --- | ---: | ---: | ---: |
| single critic | `25%` | `0.8400` | `0.8400` |
| single critic | `50%` | `0.8700` | `0.8700` |
| four-critic ensemble | `25%` | `0.8900` | `0.8900` |
| four-critic ensemble | `50%` | `0.9000` | `0.9100` |

## Interpretation

Adaptive cutoff-margin expansion contains some useful signal:

- It beats random expansion at matched average candidate count.
- In the best ensemble setting, it reaches `0.97` target recovery while scoring
  `12/39` candidates on average instead of fixed top-16's `16/39`.

But this is not a scalable bridge yet:

- The four-critic ensemble uses `32` sparse train scores per prompt, close to
  full enumeration on this `39`-class task.
- Single-critic adaptive expansion is materially worse than fixed top-16
  (`0.92` versus `0.96`) even after expanding half the prompts.
- The explicit uncertainty scores, std and LCB intrusion, are weaker than the
  simple cutoff-margin heuristic.
- This is still static fixed-grid validation, not streaming/evolving-checkpoint
  training evidence.

## Decision

```text
result_boundary_adaptive_expansion_mixed_negative
```

Do not continue with threshold/beta/expand-fraction sweeps as novelty. The
useful lesson is that margin can allocate extra scoring better than random, but
the current critic does not provide enough adaptive-compute leverage to replace
full forced-result enumeration. The next mechanism should change target
construction, validate across evolving checkpoints, or train a proposal model
whose uncertainty is calibrated without relying on broad ensemble/sparse-score
cost.
