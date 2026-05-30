# 2026-05-29 Learned Proposal Local-Target Gate

## Goal

Test whether a learned/generalized candidate proposal can make
`policy_reweighted_t1` local targets less dependent on broad uniform
forced-result scoring or fixed prompt-keyed replay memory.

## Mechanism

Added `learned_policy_reweighted_t<T>_u<U>_p<P>_h<H>_e<E>` to
`scripts/run_phase7_local_target_stage1_lift_gate.py`.

The branch:

- trains a small online MLP to predict forced-result answer loss from observed
  `(operand_a, operand_b, result_class)` candidate scores;
- proposes the `P` lowest predicted-loss result classes;
- adds `U` uniform exploration candidates;
- scores the combined candidate set with the real forced-result decoder;
- builds the same dense unique policy-reweighted target used by other sparse
  local-target branches.

The proposal uses no true-sum labels and no prompt-keyed replay cache.

## Commands

Smoke:

```bash
python3 scripts/run_phase7_local_target_stage1_lift_gate.py \
  --operand-max 3 \
  --steps 2 \
  --eval-every 1 \
  --control-eval-every 2 \
  --eval-samples 32 \
  --branches learned_policy_reweighted_t1_u2_p4_h16_e1,sampled_policy_reweighted_t1_k0_u6 \
  --output-root runs/2026-05-29_phase7_learned_proposal_local_target_gate/smoke_op3
```

Fixed-grid 200-step gate:

```bash
python3 scripts/run_phase7_local_target_stage1_lift_gate.py \
  --steps 200 \
  --eval-every 25 \
  --control-eval-every 100 \
  --eval-samples 128 \
  --branches policy_reweighted_t1,sampled_policy_reweighted_t1_k0_u32,learned_policy_reweighted_t1_u8_p24_h32_e1,learned_policy_reweighted_t1_u16_p16_h32_e1,learned_policy_reweighted_t1_u4_p28_h32_e1,learned_policy_reweighted_t1_u8_p24_h64_e3 \
  --output-root runs/2026-05-29_phase7_learned_proposal_local_target_gate/learned_proposal_200
```

Streaming stress:

```bash
python3 scripts/run_phase7_local_target_stage1_lift_gate.py \
  --steps 200 \
  --streaming-train-batch-size 16 \
  --eval-every 25 \
  --control-eval-every 100 \
  --eval-samples 128 \
  --branches policy_reweighted_t1,sampled_policy_reweighted_t1_k0_u32,learned_policy_reweighted_t1_u4_p28_h32_e1 \
  --output-root runs/2026-05-29_phase7_learned_proposal_local_target_gate/learned_proposal_streaming_b16_200
```

```bash
python3 scripts/run_phase7_local_target_stage1_lift_gate.py \
  --steps 800 \
  --streaming-train-batch-size 16 \
  --eval-every 100 \
  --control-eval-every 200 \
  --eval-samples 128 \
  --branches sampled_policy_reweighted_t1_k0_u32,learned_policy_reweighted_t1_u4_p28_h32_e1 \
  --output-root runs/2026-05-29_phase7_learned_proposal_local_target_gate/learned_proposal_streaming_b16_800
```

## Results

Smoke passed:

- `learned_policy_reweighted_t1_u2_p4_h16_e1`: final normal `0.2188`,
  final calc `0.1250`.
- `sampled_policy_reweighted_t1_k0_u6`: final normal `0.1562`, final calc
  `0.1875`.

Fixed-grid 200-step gate:

| Branch | Forced scores | Proposal coverage | Target argmax | Exact-grid calc | Sampled normal |
| --- | ---: | ---: | ---: | ---: | ---: |
| `policy_reweighted_t1` | full | n/a | `0.9925` | `0.5600` | `0.5391` |
| `sampled_policy_reweighted_t1_k0_u32` | `32` | n/a | `0.8350` | `0.3350` | `0.3438` |
| `learned_policy_reweighted_t1_u8_p24_h32_e1` | `32` | `1.0000` | `0.9225` | `0.4925` | `0.4141` |
| `learned_policy_reweighted_t1_u16_p16_h32_e1` | `32` | `1.0000` | `0.9325` | `0.5050` | `0.4141` |
| `learned_policy_reweighted_t1_u4_p28_h32_e1` | `32` | `1.0000` | `0.9175` | `0.5850` | `0.5703` |
| `learned_policy_reweighted_t1_u8_p24_h64_e3` | `32` | `1.0000` | `0.9875` | `0.4850` | `0.4766` |

All final fixed-grid controls stayed low:

- injection-zero `0.0234`;
- forced-random `0.0156`;
- oracle `1.0000`.

Streaming stress:

| Gate | Branch | Exact-grid calc | Sampled normal |
| --- | --- | ---: | ---: |
| batch `16`, 200 steps | `policy_reweighted_t1` | `0.1100` | `0.1016` |
| batch `16`, 200 steps | `sampled_policy_reweighted_t1_k0_u32` | `0.0700` | `0.0703` |
| batch `16`, 200 steps | `learned_policy_reweighted_t1_u4_p28_h32_e1` | `0.0925` | `0.0938` |
| batch `16`, 800 steps | `sampled_policy_reweighted_t1_k0_u32` | `0.2350` | `0.2734` |
| batch `16`, 800 steps | `learned_policy_reweighted_t1_u4_p28_h32_e1` | `0.2350` | `0.2656` |

At the final 800-step streaming snapshot, the learned branch had current-batch
proposal true-candidate coverage `1.0000` and target argmax `1.0000`, but that
did not translate into better full-grid calculator accuracy than raw `u32`.

## Decision

```text
learned_proposal_local_target_partial_fixed_grid_only
```

The simple learned proposal is a useful clue and a fixed-grid positive, but it
does not solve the scalability/generalization requirement. Do not continue by
tuning the same fixed-grid proposal knobs. A next learned-proposal attempt
needs an explicit streaming/generalization mechanism or validation objective.
