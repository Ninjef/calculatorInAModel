# 2026-05-30 Result-Boundary Cross-Checkpoint Critic Gate

## Question

Can a sparse result-boundary proposal critic trained on one model state propose
useful candidates for later model states, or is the previous proposal result a
static same-checkpoint diagnostic?

This tests the "evolving-checkpoint validation" path left open by the static
result-boundary reviews. It is still a diagnostic, not a source-training
integration.

## Tooling

Added:

```text
scripts/diagnose_result_boundary_cross_checkpoint_critic.py
```

The script:

- trains a sparse forced-loss critic on one checkpoint;
- evaluates direct argmin and proposal-plus-rescoring on the train checkpoint
  and one or more different eval checkpoints;
- uses the same heldout prompt split across checkpoints;
- logs standardized feature drift under the train critic normalization.

Regression coverage:

```text
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/codex_pycache pytest tests/test_model.py -q -k "cross_checkpoint or result_boundary_uncertainty_candidate_proposals or result_boundary_adaptive_candidate_expansion"
```

Result:

```text
3 passed, 138 deselected
```

## Setup

Known May 13 full-grid upstream-open result-boundary source lineage:

```text
runs/2026-05-13_phase7_full_grid_upstream_open_result_boundary_retention/stage1_primary_full_grid/2026-05-13_153947_011891_model-c-op0-19-fullgrid-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-hard_best_result-rbtt0.25-rbtchunk64-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed2
```

Settings:

- train prompts: `300`
- heldout prompts: `100`
- result classes: `39`
- sparse train scores: `8` per train prompt (`2400` total)
- critic: single pairwise MLP, hidden size `128`
- epochs: `600`
- proposal: top-8 predicted candidates, then score only those candidates
- full forced-result losses computed only for evaluation

Run files:

```text
runs/2026-05-30_phase7_result_boundary_cross_checkpoint_critic/train100_eval400_800_single_pairwise_k8.json
runs/2026-05-30_phase7_result_boundary_cross_checkpoint_critic/train400_eval800_single_pairwise_k8.json
runs/2026-05-30_phase7_result_boundary_cross_checkpoint_critic/train800_eval100_400_single_pairwise_k8.json
```

## Results

Top-8 proposal rescoring, scored-best equals full-enum best:

| Train checkpoint | Eval checkpoint | Top-8 recovery | Mean regret | Direct argmin recovery | Standardized feature abs mean |
| --- | --- | ---: | ---: | ---: | ---: |
| step `100` | step `100` | `0.48` | `1.70` | `0.06` | `0.92` |
| step `100` | step `400` | `0.11` | `5.59` | `0.00` | `2.12` |
| step `100` | step `800` | `0.12` | `5.29` | `0.00` | `3.86` |
| step `400` | step `400` | `0.74` | `0.82` | `0.22` | `0.93` |
| step `400` | step `800` | `0.23` | `3.70` | `0.03` | `1.06` |
| step `800` | step `800` | `0.79` | `0.67` | `0.20` | `0.93` |
| step `800` | step `100` | `0.42` | `2.15` | `0.02` | `1.09` |
| step `800` | step `400` | `0.58` | `1.45` | `0.13` | `1.01` |

The full-enum best was the true sum on all heldout prompts in these rows.

## Interpretation

Static sparse proposal critics are state-local.

- Same-state proposal quality improves as the source matures:
  step `100` top-8 recovery `0.48`, step `400` `0.74`, step `800` `0.79`.
- Forward transfer fails badly. A critic trained at step `100` drops to
  `0.11-0.12` on steps `400/800`; a critic trained at step `400` drops to
  `0.23` on step `800`.
- Backward transfer from step `800` is partial (`0.42` to step `100`, `0.58`
  to step `400`) but not strong enough to be a training recipe.
- Feature drift is visible for the worst forward transfer rows, but the
  step `400` to step `800` failure shows that modest feature drift can still
  hide a target-quality collapse.

## Decision

```text
result_boundary_cross_checkpoint_proposal_negative
```

Do not treat a frozen/static sparse result-boundary critic as a scalable bridge
from proposal diagnostics into training. If result-boundary proposals remain
active, they need online refresh, state calibration, or an explicit evolving
validation objective that preserves target quality as the source model changes.
