# 2026-05-29 Additive Forced-Margin Source Auxiliary Gate

## Question

Can a contrastive additive source-geometry auxiliary shape handoff geometry
without competing with source policy acquisition?

The prior scheduled forced-true auxiliary improved additive geometry and
handoff, but it trains only the true forced result. This gate tests a different
mechanism: make the true forced result lower-loss than sampled wrong forced
results under the additive downstream path.

## Code Change

Added optional CLI flags to `scripts/overfit_one_batch.py`:

- `--additive-forced-margin-loss-weight`
- `--additive-forced-margin-start-step`
- `--additive-forced-margin-ramp-steps`
- `--additive-forced-margin-negative-count`
- `--additive-forced-margin`

When active, the auxiliary temporarily switches the source model to additive
mode, forces the true result class and sampled wrong result classes, and adds a
hinge loss:

```text
max(0, margin + true_forced_answer_loss - hardest_wrong_forced_answer_loss)
```

## Commands

Focused verification:

```bash
python3 -m py_compile scripts/overfit_one_batch.py
PYTHONPATH=. pytest tests/test_model.py -q -k 'additive_forced_margin or late_source_recovery'
```

Small source gate:

```bash
python3 scripts/overfit_one_batch.py \
  --variant model-c \
  --digits 2 \
  --steps 100 \
  --operand-max 9 \
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
  --answer-loss-weight 0 \
  --result-policy-improvement-assignment-weight 10 \
  --result-policy-entropy-weight 0.05 \
  --result-policy-batch-diversity-weight 0.1 \
  --result-policy-stabilization-decay-steps 0 \
  --additive-forced-margin-loss-weight 0.5 \
  --additive-forced-margin-start-step 50 \
  --additive-forced-margin-negative-count 4 \
  --additive-forced-margin 0.05 \
  --eval-samples 100 \
  --snapshot-every 50 \
  --snapshot-samples 100 \
  --checkpoint-every 100 \
  --log-every 50 \
  --seed 13 \
  --n-layer 2 \
  --n-head 1 \
  --n-embd 16 \
  --mlp-expansion 1 \
  --calculator-hook-after-layer 1 \
  --run-root runs/2026-05-29_phase7_additive_forced_margin/small_gate_margin_sched50
```

Geometry probe:

```bash
python3 scripts/run_additive_handoff_geometry_probe.py \
  --checkpoint runs/2026-05-29_phase7_additive_forced_margin/small_gate_margin_sched50/2026-05-29_170913_379729_model-c-op0-9-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed15/checkpoint_snapshots/step_00100_weights.pt \
  --digits 2 \
  --operand-max 9 \
  --calculator-operand-vocab-size 20 \
  --slope-steps 0,25,50 \
  --seed 13 \
  --output-root runs/2026-05-29_phase7_additive_forced_margin/small_gate_margin_sched50/geometry_probe
```

## Results

Source:

| Step | Result-policy accuracy | Top-3 result accuracy | Margin loss | Margin active fraction |
| ---: | ---: | ---: | ---: | ---: |
| `50` | `0.1600` | `0.5200` | `0.0718` | `1.0000` |
| `100` | `0.4100` | `0.8100` | `0.0175` | `0.3400` |

Final source eval exact-match was `0.3800`.

Geometry:

| Branch | Source calc | Forced best=true | Forced top3=true | True loss | Best loss | True-best gap | Slope final loss @50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scheduled forced-margin | `0.4100` | `0.6200` | `0.7500` | `2.5461` | `2.5379` | `0.0082` | `1.0238` |

Prior comparable small-gate anchors:

| Branch | Source calc | Final eval | Forced best=true | Forced top3=true | Slope final loss @50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | `0.3500` | `0.3800` | `0.0000` | `0.0000` | `1.5305` |
| always-on forced-true | `0.2800` | `0.2800` | `0.5900` | `0.6900` | `0.7367` |
| scheduled forced-true | `0.3900` | `0.4000` | `0.5100` | `0.5600` | `0.7979` |
| scheduled forced-margin | `0.4100` | `0.3800` | `0.6200` | `0.7500` | `1.0238` |

## Interpretation

Mixed-positive.

The scheduled forced-margin auxiliary did not hurt source policy acquisition in
the small gate: source result accuracy reached `0.4100`, slightly above the
scheduled forced-true prior. It also improved forced-result ranking more than
scheduled forced-true (`0.6200` best / `0.7500` top-3).

The caveat is the downstream slope: its 50-step slope final loss `1.0238` was
better than baseline but worse than scheduled forced-true. So the margin
objective is a real new mechanism, but not clearly superior.

## Decision

```text
additive_forced_margin_source_aux_mixed_positive_small_gate
```

Do not repeat the same small gate as novelty.

Next useful test, if pursuing this branch: run a full-grid `operand_max=19`
scheduled forced-margin source gate with targeted geometry and trusted
handoff/readout validation against the existing scheduled forced-true source
objective.
