# 2026-05-30 Sampled Pairwise Preference Target Gate

## Question

Can a sparse answer-derived pairwise preference target train the result-space
calculator policy without predicting the full-enum argmin?

Instead of selecting one best result, this target samples result candidates,
scores their forced answer losses, and trains the policy logits to prefer
lower-loss sampled candidates over higher-loss sampled candidates.

## Tooling

Added a new Stage 1 local-target branch:

```text
sampled_pairwise_preference_uN[_gG]
```

Examples:

- `sampled_pairwise_preference_u8`
- `sampled_pairwise_preference_u16_g0p25`

The branch uses only sparse forced-result scores in the training loss. Full-grid
calculator accuracy and sampled controls remain evaluation-only.

Focused parser test:

```text
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/codex_pycache pytest tests/test_model.py -q -k 'phase7_memory_local_target_branch_parser'
```

Result: `1 passed, 118 deselected`.

## Run

```text
runs/2026-05-30_phase7_sampled_pairwise_preference_gate/fixed_grid_200
```

Configuration:

- fixed full-grid training, `operand_max=19`
- `200` Stage 1 target steps
- compared:
  - `sampled_pairwise_preference_u8`
  - `sampled_pairwise_preference_u16`
  - `sampled_pairwise_preference_u32`
  - `sampled_policy_reweighted_t1_k0_u32`

## Results

| Branch | Scored results | True candidate coverage | Sampled best true | Final exact calc | Final sampled normal |
| --- | ---: | ---: | ---: | ---: | ---: |
| pairwise `u8` | `8` | `0.1850` | `0.1850` | `0.0050` | `0.0078` |
| pairwise `u16` | `16` | `0.4050` | `0.4050` | `0.0050` | `0.0078` |
| pairwise `u32` | `32` | `0.8450` | `0.8450` | `0.0425` | `0.0234` |
| policy-reweighted `u32` | `32` | `0.8450` | n/a | `0.3350` | `0.3438` |

The pairwise branch did not have hidden best snapshots. For `u8` and `u16`,
the best exact-grid calc remained the step-0 value (`0.0100`). For `u32`, the
best was the final step (`0.0425`), still far below the sparse
policy-reweighted comparator.

## Decision

```text
sampled_pairwise_preference_target_negative
```

Interpretation:

- Pairwise preference is a different target construction, but it does not
  produce useful Stage 1 lift in this setting.
- The failure is not merely candidate coverage: `u32` sampled the true result
  in `84.5%` of prompts, yet reached only `0.0425` exact calc while the
  same-budget policy-reweighted branch reached `0.3350`.
- Do not continue with simple sampled pairwise-preference candidate-count or
  loss-gap sweeps as novelty.
- If pursuing pairwise ideas later, they need a materially different mechanism,
  such as policy-aware weighting, an uncertainty-aware active sampler, or a
  target that accumulates preferences without collapsing into noisy local
  pair constraints.
