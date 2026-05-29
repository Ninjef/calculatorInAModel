# 2026-05-29 Local-Target Stage 1 Lift Gate

## Aim

Follow the Stage 0 local-target propagation partial positive with the
predeclared short Stage 1 lift gate. The goal was to test whether softer
aligned local targets can actually move the learned calculator-result policy,
not just align at initialization.

## Code

Added:

```text
scripts/run_phase7_local_target_stage1_lift_gate.py
```

The runner trains each branch from the same initialized tiny natural `0..19`
result-space model with the frozen Phase 6 product semantic decoder. It uses
the exact `20 x 20` grid for training and records exact-grid calculator-policy
metrics every 25 steps. Sampled generation/intervention controls are recorded
at steps `0`, `100`, and `200`.

Branches:

- `hard_boundary`: known hard best-result boundary-target ceiling.
- `expected_loss`: ordinary full-enum expected answer loss baseline.
- `policy_reweighted_t1`: local target
  `softmax(log current_policy - forced_loss / 1.0)`.
- `logit_descent_p0p1`: local free-logit descent with proximity `0.1`.

## Run

Run root:

```text
runs/2026-05-29_phase7_local_target_stage1_lift_gate/full_grid_seed2_steps200_fast_controls
```

Command:

```text
python3 scripts/run_phase7_local_target_stage1_lift_gate.py \
  --control-eval-every 100 \
  --eval-samples 128 \
  --output-root runs/2026-05-29_phase7_local_target_stage1_lift_gate/full_grid_seed2_steps200_fast_controls
```

## Result

| Branch | Final exact-grid calc | Final sampled normal | Injection-zero | Forced-random | Oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hard_boundary` | `0.5500` | `0.4844` | `0.0234` | `0.0156` | `1.0000` |
| `expected_loss` | `0.0025` | `0.0000` | `0.0234` | `0.0156` | `1.0000` |
| `policy_reweighted_t1` | `0.5600` | `0.5391` | `0.0234` | `0.0156` | `1.0000` |
| `logit_descent_p0p1` | `0.2950` | `0.1953` | `0.0234` | `0.0156` | `1.0000` |

Exact-grid calculator-result accuracy curves:

| Step | Hard boundary | Expected loss | Policy reweighted `t=1` | Logit descent `p=0.1` |
| ---: | ---: | ---: | ---: | ---: |
| `0` | `0.0100` | `0.0100` | `0.0100` | `0.0100` |
| `50` | `0.1025` | `0.0025` | `0.0625` | `0.0500` |
| `100` | `0.1925` | `0.0025` | `0.1700` | `0.0850` |
| `150` | `0.4775` | `0.0025` | `0.4375` | `0.1650` |
| `200` | `0.5500` | `0.0025` | `0.5600` | `0.2950` |

## Decision

Label:

```text
local_target_policy_reweighted_stage1_lift_partial_positive
```

`policy_reweighted_t1` passes the short Stage 1 lift gate. It clearly beats
the failed expected-loss baseline and slightly beats the same-budget
hard-boundary run at step `200`. The intervention controls stay low relative
to normal, so the sampled answer path remains calculator-dependent.

`logit_descent_p0.1` is a weaker partial: it improves above chance but lags
the policy-reweighted and hard-boundary branches.

This is still not a scalable final method because both local-target branches
score every forced result class on every update.

## Anti-Rerun Note

Do not repeat this same seed-2 200-step comparison over `hard_boundary`,
`expected_loss`, `policy_reweighted_t1`, and `logit_descent_p0.1` as novelty.

Next useful tests:

- extend or replicate `policy_reweighted_t1` to test convergence and
  target-off retention;
- test a sampled/top-k/learned approximation to the policy-reweighted local
  target so the branch can start addressing scalability;
- avoid spending more compute on ordinary expected-loss training in this
  setup, since it again collapsed to near chance.

## Verification

Commands completed:

```text
python3 -m py_compile scripts/run_phase7_local_target_stage1_lift_gate.py
python3 scripts/run_phase7_local_target_stage1_lift_gate.py --operand-max 3 --steps 2 --eval-every 1 --control-eval-every 2 --eval-samples 32 --output-root runs/2026-05-29_phase7_local_target_stage1_lift_gate/smoke_op3_steps2_fast_controls
python3 scripts/run_phase7_local_target_stage1_lift_gate.py --control-eval-every 100 --eval-samples 128 --output-root runs/2026-05-29_phase7_local_target_stage1_lift_gate/full_grid_seed2_steps200_fast_controls
```

The full run wrote:

```text
runs/2026-05-29_phase7_local_target_stage1_lift_gate/full_grid_seed2_steps200_fast_controls/local_target_stage1_summary.json
runs/2026-05-29_phase7_local_target_stage1_lift_gate/full_grid_seed2_steps200_fast_controls/local_target_stage1_rows.csv
```
