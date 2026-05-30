# 2026-05-29 Pretrained Learned Proposal Gate

## Goal

Test whether proposal pretraining on random prompt/result forced-loss
observations gives the learned local-target proposal a real streaming
generalization mechanism.

## Mechanism

Extended learned proposal branch syntax with optional `_wN`, for example:

```text
learned_policy_reweighted_t1_u4_p28_h32_e1_w20
```

Before model training, the proposal MLP trains for `N` random prompt batches.
Each pretrain batch samples prompts and random result candidates, scores those
candidates with the real forced-result answer loss, and updates only the
proposal model. Main model training then proceeds normally, with online
proposal updates from observed candidates.

This is still not prompt-keyed replay memory and uses no true-result labels.

## Commands

Smoke:

```bash
python3 scripts/run_phase7_local_target_stage1_lift_gate.py \
  --operand-max 3 \
  --steps 2 \
  --eval-every 1 \
  --control-eval-every 2 \
  --eval-samples 32 \
  --learned-proposal-pretrain-batch-size 8 \
  --branches learned_policy_reweighted_t1_u2_p4_h16_e1_w2,sampled_policy_reweighted_t1_k0_u6 \
  --output-root runs/2026-05-29_phase7_pretrained_learned_proposal_gate/smoke_op3
```

Streaming 200-step screen:

```bash
python3 scripts/run_phase7_local_target_stage1_lift_gate.py \
  --steps 200 \
  --streaming-train-batch-size 16 \
  --eval-every 25 \
  --control-eval-every 100 \
  --eval-samples 128 \
  --learned-proposal-pretrain-batch-size 32 \
  --branches sampled_policy_reweighted_t1_k0_u32,learned_policy_reweighted_t1_u4_p28_h32_e1,learned_policy_reweighted_t1_u4_p28_h32_e1_w20,learned_policy_reweighted_t1_u4_p28_h32_e1_w50 \
  --output-root runs/2026-05-29_phase7_pretrained_learned_proposal_gate/streaming_b16_200_screen
```

Streaming 800-step stress:

```bash
python3 scripts/run_phase7_local_target_stage1_lift_gate.py \
  --steps 800 \
  --streaming-train-batch-size 16 \
  --eval-every 100 \
  --control-eval-every 200 \
  --eval-samples 128 \
  --learned-proposal-pretrain-batch-size 32 \
  --branches sampled_policy_reweighted_t1_k0_u32,learned_policy_reweighted_t1_u4_p28_h32_e1_w20 \
  --output-root runs/2026-05-29_phase7_pretrained_learned_proposal_gate/streaming_b16_800_w20
```

## Results

Smoke passed:

- `learned_policy_reweighted_t1_u2_p4_h16_e1_w2`: final normal `0.2188`,
  final calc `0.1250`.
- `sampled_policy_reweighted_t1_k0_u6`: final normal `0.1562`, final calc
  `0.1875`.

Streaming 200-step screen:

| Branch | Exact-grid calc | Sampled normal | Target argmax | Proposal argmin |
| --- | ---: | ---: | ---: | ---: |
| `sampled_policy_reweighted_t1_k0_u32` | `0.0700` | `0.0703` | `0.7500` | n/a |
| `learned_policy_reweighted_t1_u4_p28_h32_e1` | `0.0925` | `0.0938` | `1.0000` | `0.4375` |
| `learned_policy_reweighted_t1_u4_p28_h32_e1_w20` | `0.0975` | `0.0625` | `0.9375` | `0.2500` |
| `learned_policy_reweighted_t1_u4_p28_h32_e1_w50` | `0.0950` | `0.0547` | `1.0000` | `0.0625` |

Streaming 800-step stress:

| Branch | Exact-grid calc | True probability | Sampled normal | Injection-zero | Forced-random |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sampled_policy_reweighted_t1_k0_u32` | `0.2350` | `0.1401` | `0.2734` | `0.0234` | `0.0156` |
| `learned_policy_reweighted_t1_u4_p28_h32_e1_w20` | `0.2625` | `0.1529` | `0.1797` | `0.0234` | `0.0156` |

The final `_w20` current-batch target was strong (`1.0000` true-candidate
coverage, `0.9375` target argmax), but this did not translate into clean
sampled-normal behavior.

## Decision

```text
pretrained_learned_proposal_streaming_mixed_negative
```

Pretraining can slightly nudge calculator-result accuracy, but this simple
proposal model still does not produce robust functional streaming calculator
use. Do not keep tuning warmup count or pretrain batch size as novelty. A next
learned proposal needs a different generalization mechanism or validation
objective.
