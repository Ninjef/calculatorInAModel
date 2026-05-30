# 2026-05-30 Result-Boundary Online-Calibrated Critic Gate

## Question

Can a frozen result-boundary proposal critic be rescued by a small amount of
fresh online/state calibration at the new checkpoint?

The previous cross-checkpoint diagnostic showed frozen critics are state-local.
This gate tests the natural next mechanism: warm-start from the old critic,
retarget feature/target normalization on fresh sparse scores from the current
state, and fine-tune before proposing candidates.

## Tooling

Extended:

```text
scripts/diagnose_result_boundary_cross_checkpoint_critic.py
```

Added:

- `--adapt-samples-per-prompt`
- `--adapt-epochs`
- `--adapt-lr`
- normalization retargeting for the first and last linear layers so changing
  feature/target normalization preserves the critic's raw predictions before
  adaptation
- paired `proposal_mode=frozen` and `proposal_mode=adapted` output rows

Focused test:

```text
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/codex_pycache pytest tests/test_model.py -q -k cross_checkpoint
```

Result:

```text
1 passed, 140 deselected
```

## Setup

Known May 13 full-grid upstream-open result-boundary source lineage:

```text
runs/2026-05-13_phase7_full_grid_upstream_open_result_boundary_retention/stage1_primary_full_grid/2026-05-13_153947_011891_model-c-op0-19-fullgrid-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-hard_best_result-rbtt0.25-rbtchunk64-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed2
```

Base critic:

- `300` train prompts / `100` heldout prompts
- single pairwise MLP critic
- `8` sparse forced scores per train prompt at the original train checkpoint
- top-8 proposal, then score proposed candidates only

Online calibration:

- warm-start from the base critic
- retarget normalization and fine-tune on fresh sparse scores at the eval
  checkpoint
- `100` adaptation epochs
- `adapt_lr=0.0003`

Run files:

```text
runs/2026-05-30_phase7_result_boundary_online_calibrated_critic/train100_eval400_800_adapt2_k8.json
runs/2026-05-30_phase7_result_boundary_online_calibrated_critic/train400_eval800_adapt2_k8.json
runs/2026-05-30_phase7_result_boundary_online_calibrated_critic/train400_eval800_adapt4_k8.json
runs/2026-05-30_phase7_result_boundary_online_calibrated_critic/train400_eval800_adapt8_k8.json
```

## Results

Top-8 proposal rescoring, scored-best equals full-enum best:

| Train -> eval | Mode | Fresh adapt scores | Top-8 recovery | Mean regret |
| --- | --- | ---: | ---: | ---: |
| step100 -> step400 | frozen | `0` | `0.11` | `5.59` |
| step100 -> step400 | adapted `k=2` | `600` | `0.36` | `2.55` |
| step100 -> step800 | frozen | `0` | `0.12` | `5.29` |
| step100 -> step800 | adapted `k=2` | `600` | `0.41` | `2.39` |
| step400 -> step800 | frozen | `0` | `0.23` | `3.70` |
| step400 -> step800 | adapted `k=2` | `600` | `0.59` | `1.34` |
| step400 -> step800 | adapted `k=4` | `1200` | `0.54` | `1.49` |
| step400 -> step800 | adapted `k=8` | `2400` | `0.62` | `1.25` |

For comparison, the previous same-state step800 critic with `8` sparse scores
per train prompt reached top-8 recovery `0.79` and mean regret `0.67`.

## Interpretation

Simple online calibration is helpful but not enough.

- A tiny fresh calibration budget can repair much of the frozen critic's worst
  forward-transfer collapse.
- The recovery is not monotonic with more samples in this warm-start procedure,
  and even `8` fresh scores per train prompt does not match the same-state
  critic.
- This means "refresh a frozen static critic with a little calibration" is not
  yet a scalable source-training proposal mechanism.

## Decision

```text
result_boundary_online_calibrated_critic_partial_negative
```

Do not wire this warm-start calibrated critic into source training as a solved
assignment replacement. A viable result-boundary proposal method now needs a
stronger online learner, active proposal/training co-design, or a different
credit-assignment mechanism.
