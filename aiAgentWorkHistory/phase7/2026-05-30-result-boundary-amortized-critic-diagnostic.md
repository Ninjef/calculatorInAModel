# 2026-05-30 Result-Boundary Amortized Critic Diagnostic

## Question

Can a shared critic trained on sparse forced-result scores approximate the
full-enumeration answer-derived result-boundary target on heldout prompts?

If this worked, it would be a plausible bridge from the positive
result-boundary source transfer result toward a scalable target construction.

## Tooling

Added:

```text
scripts/diagnose_result_boundary_amortized_critic.py
```

The diagnostic:

- loads a checkpoint from the known result-boundary source lineage;
- computes full forced-result losses only for evaluation;
- trains an MLP critic from sparse `(prompt hidden state, candidate output
  vector) -> forced answer loss` examples on training prompts;
- validates whether the critic's predicted argmin recovers the full-enum
  boundary best result on heldout prompts.
- supports pointwise, pairwise ranking, and hybrid critic losses.

This is not a training method yet. It is a gate to avoid wiring a weak
approximation into the training loop.

## Runs

Primary sparse gate:

```text
runs/2026-05-30_phase7_result_boundary_amortized_critic/hidden_output_k8_step0_100_800.json
```

Configuration:

- source lineage:
  `runs/2026-05-13_phase7_full_grid_upstream_open_result_boundary_retention/stage1_primary_full_grid/...`
- checkpoints: step `0`, step `100`, step `800`
- full grid `operand_max=19`
- heldout prompts: `100`
- train prompts: `300`
- sparse forced scores per train prompt: `8`
- critic hidden size `128`, `600` epochs

Wider budget check:

```text
runs/2026-05-30_phase7_result_boundary_amortized_critic/hidden_output_k24_step0_800.json
```

Same setup, but `24` sparse forced scores per train prompt for step `0` and
step `800`.

Rank-aware checks:

```text
runs/2026-05-30_phase7_result_boundary_amortized_critic/hidden_output_pairwise_k24_step0_800.json
runs/2026-05-30_phase7_result_boundary_amortized_critic/hidden_output_hybrid_k24_step0_800.json
```

Same `k=24` setup with pairwise ranking loss and pointwise+pairwise hybrid
loss.

## Results

At all tested checkpoints, the full-enum boundary best was exactly the true sum
on heldout prompts (`1.0000`), so the target was sharp and valid.

Sparse `k=8` critic:

| Checkpoint | Forced scores used | Heldout argmin = full best | Top-3 contains best | Top-5 contains best | Mean regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| step `0` | `2400` | `0.0800` | `0.1700` | `0.3200` | `5.2390` |
| step `100` | `2400` | `0.0800` | `0.1700` | `0.2900` | `4.7240` |
| step `800` | `2400` | `0.1700` | `0.4700` | `0.6400` | `3.5766` |

Wider `k=24` critic:

| Checkpoint | Forced scores used | Heldout argmin = full best | Top-3 contains best | Top-5 contains best | Mean regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| step `0` | `7200` | `0.2600` | `0.4700` | `0.6500` | `3.6525` |
| step `800` | `7200` | `0.1900` | `0.5900` | `0.7800` | `3.3686` |

Rank-aware `k=24` critic:

| Mode | Checkpoint | Heldout argmin = full best | Top-3 contains best | Top-5 contains best | Mean regret |
| --- | --- | ---: | ---: | ---: | ---: |
| pairwise | step `0` | `0.2600` | `0.4600` | `0.6200` | `3.6179` |
| pairwise | step `800` | `0.4000` | `0.6400` | `0.8300` | `2.3874` |
| hybrid | step `0` | `0.2000` | `0.4100` | `0.6200` | `3.9556` |
| hybrid | step `800` | `0.2700` | `0.5800` | `0.7800` | `2.9605` |

## Decision

```text
hidden_output_amortized_boundary_critic_negative
```

Interpretation:

- A hidden-state plus candidate-output-vector critic is not good enough to
  replace full forced-result enumeration for boundary target selection.
- Pairwise ranking helps at the trained step-800 checkpoint (`0.4000` argmin
  recovery), but this is still far below what a source-training target needs
  and uses most of the result vocabulary as sparse supervision.
- Do not wire this critic family into source training as a scalable
  result-boundary approximation.
- The next approximation needs a stronger generalization mechanism or a
  different target construction, not another pointwise/rank loss tweak over
  the same hidden/output features.
