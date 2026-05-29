# 2026-05-29 Replay-Memory Streaming Prompt Gate

## Question

Does replay-memory local-target training keep its fixed-grid advantage when
target training uses sampled minibatches instead of the full exhaustive prompt
grid every step?

## Change

Added `--streaming-train-batch-size` to
`scripts/run_phase7_local_target_stage1_lift_gate.py`.

When this option is positive:

- target training uses freshly sampled fixed-width arithmetic minibatches;
- evaluation remains on the full exhaustive grid;
- replay-memory branches use prompt-keyed caches, not row-indexed caches, so a
  changed minibatch row cannot inherit another prompt's cached losses.

## Commands

Smoke:

```bash
python3 scripts/run_phase7_local_target_stage1_lift_gate.py \
  --operand-max 3 \
  --steps 2 \
  --streaming-train-batch-size 4 \
  --eval-every 1 \
  --control-eval-every 2 \
  --eval-samples 32 \
  --branches memory_policy_reweighted_t1_u2_m2,sampled_policy_reweighted_t1_k0_u4 \
  --output-root runs/2026-05-29_phase7_local_target_gate/smoke_streaming_op3
```

Batch 16, 200 steps:

```bash
python3 scripts/run_phase7_local_target_stage1_lift_gate.py \
  --steps 200 \
  --streaming-train-batch-size 16 \
  --eval-every 25 \
  --control-eval-every 100 \
  --eval-samples 128 \
  --branches policy_reweighted_t1,sampled_policy_reweighted_t1_k0_u32,memory_policy_reweighted_t1_u2_m30,memory_policy_reweighted_t1_u8_m24 \
  --output-root runs/2026-05-29_phase7_local_target_gate/streaming_b16_200
```

Batch 16, 800 steps:

```bash
python3 scripts/run_phase7_local_target_stage1_lift_gate.py \
  --steps 800 \
  --streaming-train-batch-size 16 \
  --eval-every 100 \
  --control-eval-every 400 \
  --eval-samples 128 \
  --branches policy_reweighted_t1,sampled_policy_reweighted_t1_k0_u32,memory_policy_reweighted_t1_u2_m30,memory_policy_reweighted_t1_u8_m24 \
  --output-root runs/2026-05-29_phase7_local_target_gate/streaming_b16_800
```

Batch 64, 200 steps:

```bash
python3 scripts/run_phase7_local_target_stage1_lift_gate.py \
  --steps 200 \
  --streaming-train-batch-size 64 \
  --eval-every 25 \
  --control-eval-every 100 \
  --eval-samples 128 \
  --branches policy_reweighted_t1,sampled_policy_reweighted_t1_k0_u32,memory_policy_reweighted_t1_u2_m30,memory_policy_reweighted_t1_u8_m24 \
  --output-root runs/2026-05-29_phase7_local_target_gate/streaming_b64_200
```

## Results

Batch 16, 200 steps:

| Branch | Exact calc | Sampled normal | Target coverage | Observed results | Prompt entries |
| --- | ---: | ---: | ---: | ---: | ---: |
| `policy_reweighted_t1` | `0.1100` | `0.1016` | `1.0000` | n/a | n/a |
| `sampled_policy_reweighted_t1_k0_u32` | `0.0700` | `0.0703` | `0.7500` | n/a | n/a |
| `memory_policy_reweighted_t1_u2_m30` | `0.0475` | `0.0156` | `0.5625` | `15.3750` | `400` |
| `memory_policy_reweighted_t1_u8_m24` | `0.0950` | `0.1016` | `0.8750` | `34.8125` | `400` |

Batch 16, 800 steps:

| Branch | Exact calc | Sampled normal | Target coverage | Observed results | Prompt entries |
| --- | ---: | ---: | ---: | ---: | ---: |
| `policy_reweighted_t1` | `0.2450` | `0.2578` | `1.0000` | n/a | n/a |
| `sampled_policy_reweighted_t1_k0_u32` | `0.2450` | `0.2578` | `0.7500` | n/a | n/a |
| `memory_policy_reweighted_t1_u2_m30` | `0.1850` | `0.2031` | `0.7500` | `32.5000` | `400` |
| `memory_policy_reweighted_t1_u8_m24` | `0.2650` | `0.2500` | `1.0000` | `38.9375` | `400` |

Batch 64, 200 steps:

| Branch | Exact calc | Sampled normal | Target coverage | Observed results | Prompt entries |
| --- | ---: | ---: | ---: | ---: | ---: |
| `policy_reweighted_t1` | `0.1650` | `0.2031` | `1.0000` | n/a | n/a |
| `sampled_policy_reweighted_t1_k0_u32` | `0.1475` | `0.1484` | `0.7969` | n/a | n/a |
| `memory_policy_reweighted_t1_u2_m30` | `0.0975` | `0.1172` | `0.7656` | `31.4531` | `400` |
| `memory_policy_reweighted_t1_u8_m24` | `0.1475` | `0.2031` | `1.0000` | `38.9688` | `400` |

## Interpretation

Streaming minibatches are mixed-negative for replay memory. The exact
`policy_reweighted_t1` ceiling itself is much slower than in the exhaustive
grid gate, so this is partly a minibatch-local-target optimization issue. But
the important replay-memory conclusion is that the strong fixed-grid advantage
does not survive: prompt-keyed `u8_m24` roughly matches the streaming
exact/raw baselines, while low-fresh `u2_m30` lags badly.

This rules out prompt-keyed replay memory as the scalable answer. Future
local-target work should require a learned/generalized proposal, estimator
correction, or different target construction. Otherwise, return compute to
source objectives aimed directly at additive handoff/readout geometry.

## Verification

```bash
python3 -m py_compile scripts/run_phase7_local_target_stage1_lift_gate.py
PYTHONPATH=. pytest tests/test_model.py -q -k 'phase7_memory_local_target_branch_parser or phase7_streaming_batch_and_prompt_memory_tables'
```

Both passed before the main experiments.
