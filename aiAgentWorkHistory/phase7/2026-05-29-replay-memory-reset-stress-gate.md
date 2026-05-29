# 2026-05-29 Replay-Memory Reset Stress Gate

## Question

Does replay-memory local-target training remain strong with finite per-prompt
caches, or does the positive rely on persistent fixed-grid transductive memory?

## Change

Added optional `_resetN` replay-memory branch syntax to
`scripts/run_phase7_local_target_stage1_lift_gate.py`.

Examples:

```text
memory_policy_reweighted_t1_u2_m30_reset50
memory_policy_reweighted_t1_u2_m30_r2_reset50
```

The reset suffix clears the cached loss/seen tables every `N` target-loss
calls. The runner logs:

- `target_memory_reset_interval`
- `target_memory_reset_count`
- `target_memory_did_reset`

## Commands

Smoke:

```bash
python3 scripts/run_phase7_local_target_stage1_lift_gate.py \
  --operand-max 3 \
  --steps 2 \
  --eval-every 1 \
  --control-eval-every 2 \
  --eval-samples 32 \
  --branches memory_policy_reweighted_t1_u2_m2_reset1,memory_policy_reweighted_t1_u2_m2 \
  --output-root runs/2026-05-29_phase7_local_target_gate/smoke_reset_op3
```

200-step reset stress:

```bash
python3 scripts/run_phase7_local_target_stage1_lift_gate.py \
  --steps 200 \
  --eval-every 25 \
  --control-eval-every 100 \
  --eval-samples 128 \
  --branches memory_policy_reweighted_t1_u2_m30,memory_policy_reweighted_t1_u2_m30_reset50,memory_policy_reweighted_t1_u2_m30_reset25,memory_policy_reweighted_t1_u2_m30_reset10,memory_policy_reweighted_t1_u8_m24 \
  --output-root runs/2026-05-29_phase7_local_target_gate/memory_reset_200
```

199-step boundary check:

```bash
python3 scripts/run_phase7_local_target_stage1_lift_gate.py \
  --steps 199 \
  --eval-every 25 \
  --control-eval-every 100 \
  --eval-samples 128 \
  --branches memory_policy_reweighted_t1_u2_m30,memory_policy_reweighted_t1_u2_m30_reset100,memory_policy_reweighted_t1_u2_m30_reset50 \
  --output-root runs/2026-05-29_phase7_local_target_gate/memory_reset_boundary_check_199
```

## Results

200-step reset stress:

| Branch | Exact calc | Sampled normal | Final observed results | Final true coverage | Final target argmax |
| --- | ---: | ---: | ---: | ---: | ---: |
| `memory_policy_reweighted_t1_u2_m30` | `0.6025` | `0.6016` | `38.7925` | `0.9925` | `0.9600` |
| `memory_policy_reweighted_t1_u2_m30_reset50` | `0.2500` | `0.2578` | `2.0000` | `0.0525` | `0.0525` |
| `memory_policy_reweighted_t1_u2_m30_reset25` | `0.1650` | `0.2188` | `2.0000` | `0.0525` | `0.0525` |
| `memory_policy_reweighted_t1_u2_m30_reset10` | `0.0950` | `0.1406` | `2.0000` | `0.0525` | `0.0525` |
| `memory_policy_reweighted_t1_u8_m24` | `0.5900` | `0.5391` | `39.0000` | `1.0000` | `0.9850` |

Intermediate curve check:

- `reset50` recovered to about `29.1` observed results and `0.73-0.78` true
  coverage midway through each window, but calculator accuracy still stayed far
  below the persistent-cache baseline.
- `reset10` only recovered to about `10.55` observed results and `0.26` true
  coverage between resets.

199-step boundary check:

| Branch | Exact calc | Sampled normal | Final observed results | Final true coverage | Final target argmax |
| --- | ---: | ---: | ---: | ---: | ---: |
| `memory_policy_reweighted_t1_u2_m30` | `0.5925` | `0.5938` | `38.7925` | `0.9925` | `0.9600` |
| `memory_policy_reweighted_t1_u2_m30_reset100` | `0.4575` | `0.4453` | `38.7800` | `0.9925` | `0.9650` |
| `memory_policy_reweighted_t1_u2_m30_reset50` | `0.2575` | `0.2812` | `36.3025` | `0.9525` | `0.9075` |

## Interpretation

Finite reset windows are mixed-negative for replay memory. The 200-step gate
shows sharp degradation when the cache is cleared, and the 199-step boundary
check shows this is not only because the final snapshot landed immediately
after a reset. `reset100` recovered nearly identical target coverage to
no-reset by the final step, but still lagged badly in learned calculator
accuracy.

The replay-memory approximation remains useful as a clue, but the current form
is too dependent on persistent prompt-identity caches to count as scalable.
The next local-target work should not tune reset intervals. It should test
streaming/non-exhaustive prompts or learned/generalized proposal memory.

## Verification

```bash
python3 -m py_compile scripts/run_phase7_local_target_stage1_lift_gate.py
PYTHONPATH=. pytest tests/test_model.py -q -k phase7_memory_local_target_branch_parser
```

Both passed before the main experiments.
