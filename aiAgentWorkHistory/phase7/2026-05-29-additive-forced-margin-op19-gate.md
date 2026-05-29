# 2026-05-29 Additive Forced-Margin Op19 Gate

## Question

Does the contrastive forced-margin source objective survive the full
`operand_max=19` setting and improve actual additive handoff against the
existing scheduled forced-true source objective?

The small gate was mixed-positive, so this test uses the matched 200-step
full-grid source point where scheduled forced-true previously reached `0.4150`
trusted 600-step handoff versus `0.2525` for baseline.

## Compute Adjustment

The direct 4-negative full-grid command was too costly locally. After roughly
ten minutes it had only written `config.json` and no training curve/checkpoint,
so I stopped it and treated that as a scalability warning.

I reran the same mechanism with one sampled negative per prompt. This keeps the
contrastive objective but changes the cost from true plus four wrong forced
passes to true plus one wrong forced pass.

## Commands

Source:

```bash
python3 scripts/overfit_one_batch.py \
  --variant model-c \
  --digits 2 \
  --steps 200 \
  --batch-size 64 \
  --eval-samples 400 \
  --lr 0.003 \
  --answer-loss-weight 0 \
  --operand-max 19 \
  --exhaustive-grid-batch \
  --calculator-operand-vocab-size 20 \
  --calculator-estimator direct_feedback_alignment \
  --calculator-action-head result_space \
  --calculator-read-position operand_spans \
  --calculator-read-span-width 2 \
  --calculator-injection-mode add \
  --calculator-bottleneck-mode answer_decoder \
  --calculator-output-format sum \
  --answer-decoder-interaction product \
  --semantic-decoder-checkpoint runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/stage0_candidates/tiny_operand_spans_dense/oracle_train/2026-05-12_113346_842566_model-c-oracle-op0-19-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00500_weights.pt \
  --semantic-decoder-checkpoint-load-scope semantic_decoder_only \
  --freeze-semantic-decoder \
  --result-policy-entropy-weight 0.05 \
  --result-policy-batch-diversity-weight 0.1 \
  --result-policy-improvement-assignment-weight 10 \
  --result-policy-stabilization-temperature 1 \
  --result-policy-stabilization-decay-steps 0 \
  --additive-forced-margin-loss-weight 0.5 \
  --additive-forced-margin-start-step 50 \
  --additive-forced-margin-negative-count 1 \
  --additive-forced-margin 0.05 \
  --snapshot-every 100 \
  --snapshot-samples 400 \
  --checkpoint-every 100 \
  --log-every 50 \
  --seed 13 \
  --n-layer 2 \
  --n-head 1 \
  --n-embd 16 \
  --mlp-expansion 1 \
  --calculator-hook-after-layer 1 \
  --run-root runs/2026-05-29_phase7_additive_forced_margin/op19_gate_steps200_neg1 \
  --device cpu
```

Geometry:

```bash
python3 scripts/run_additive_handoff_geometry_probe.py \
  --checkpoint runs/2026-05-29_phase7_additive_forced_margin/op19_gate_steps200_neg1/2026-05-29_173238_501749_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed15/checkpoint_snapshots/step_00200_weights.pt \
  --digits 2 \
  --operand-max 19 \
  --calculator-operand-vocab-size 20 \
  --slope-steps 0,25,50 \
  --seed 13 \
  --output-root runs/2026-05-29_phase7_additive_forced_margin/op19_gate_steps200_neg1/geometry_probe
```

Trusted handoff:

```bash
python3 scripts/overfit_one_batch.py \
  --variant model-c \
  --digits 2 \
  --steps 600 \
  --batch-size 64 \
  --eval-samples 400 \
  --lr 0.003 \
  --answer-loss-weight 1 \
  --operand-max 19 \
  --exhaustive-grid-batch \
  --calculator-operand-vocab-size 20 \
  --calculator-estimator ste \
  --calculator-action-head result_space \
  --calculator-read-position operand_spans \
  --calculator-read-span-width 2 \
  --calculator-injection-mode add \
  --calculator-bottleneck-mode none \
  --calculator-output-format sum \
  --answer-decoder-interaction product \
  --semantic-decoder-checkpoint runs/2026-05-29_phase7_additive_forced_margin/op19_gate_steps200_neg1/2026-05-29_173238_501749_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed15/checkpoint_snapshots/step_00200_weights.pt \
  --semantic-decoder-checkpoint-load-scope compatible_model \
  --freeze-semantic-decoder \
  --freeze-calculator-policy \
  --snapshot-every 100 \
  --snapshot-samples 400 \
  --checkpoint-every 100 \
  --log-every 100 \
  --seed 13 \
  --n-layer 2 \
  --n-head 1 \
  --n-embd 16 \
  --mlp-expansion 1 \
  --calculator-hook-after-layer 1 \
  --run-root runs/2026-05-29_phase7_additive_forced_margin/op19_gate_steps200_neg1/handoff600_step200 \
  --device cpu
```

## Results

Source:

| Step | Source calc | Snapshot normal | Margin loss | Margin active fraction |
| ---: | ---: | ---: | ---: | ---: |
| `50` | `0.1600` | n/a | `0.0566` | `0.9875` |
| `100` | `0.2225` | `0.2100` | `0.0081` | `0.1775` |
| `150` | `0.3125` | n/a | `0.0045` | `0.1200` |
| `200` | `0.3225` | `0.3450` | `0.0028` | `0.0550` |

Final source eval exact-match was `0.3600`.

Geometry:

| Branch | Source calc | Forced best=true | True-best gap | Slope final loss @50 |
| --- | ---: | ---: | ---: | ---: |
| forced-margin, 1 negative | `0.3225` | `0.6725` | `0.0102` | `1.4660` |

Trusted 600-step frozen-policy additive handoff:

| Step | Normal | Injection-zero | Oracle | Forced-random | Learned calc |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `400` | `0.6300` | `0.0000` | `0.5300` | `0.0700` | `0.3200` |
| `500` | `0.6875` | `0.0000` | `0.4925` | `0.0525` | `0.3300` |
| `600` | `0.7050` | `0.0000` | `0.5350` | `0.0250` | `0.3375` |

Final eval exact-match was `0.6600`.

Matched 200-step anchors:

| Branch | Source calc | Source final eval | Slope final loss @50 | Handoff final eval |
| --- | ---: | ---: | ---: | ---: |
| baseline prior | `0.2875` | `0.2825` | `1.8058` | `0.2525` |
| scheduled forced-true prior | `0.2800` | `0.2750` | `1.0360` | `0.4150` |
| forced-margin, 1 negative | `0.3225` | `0.3600` | `1.4660` | `0.6600` |

## Interpretation

Positive full-grid early handoff gate for the budgeted one-negative variant.

The one-negative forced-margin auxiliary improved source policy acquisition
over scheduled forced-true at the matched 200-step point and produced a much
stronger trusted 600-step handoff (`0.6600` final eval versus `0.4150`).
Controls stayed low at the handoff snapshot, so this is calculator-dependent
transfer rather than a pure-neuron bypass.

The 50-step slope was misleadingly pessimistic relative to actual handoff,
which reinforces the standing rule: geometry/slope probes are diagnostics, not
selectors. The many-negative full-grid version is not the scalable branch
unless its cost is reduced.

## Decision

```text
additive_forced_margin_op19_neg1_handoff_positive
```

Do not repeat the same seed-13, `operand_max=19`, 200-step one-negative
forced-margin source plus 600-step handoff as novelty. Do not rerun the
4-negative full-grid branch without a compute-reduction plan.

Next useful test: extend one-negative forced-margin to a longer source horizon
(`400/600`) and verify with trusted 600-step handoff, or replicate on a fresh
seed if the explicit question is stability.
