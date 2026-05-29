# 2026-05-29 Additive Forced-True Source Auxiliary Gate

## Purpose

Test a source-acquisition objective that changes training pressure toward
additive non-bottleneck readout geometry instead of merely ranking completed
source checkpoints.

The auxiliary runs during bottleneck source acquisition. It temporarily
evaluates the same model with `calculator_bottleneck_mode="none"`, forces the
true calculator result class, and adds answer loss through the ordinary
additive downstream path:

```text
--additive-forced-true-loss-weight
```

This is still prescriptive, like the hard improvement-assignment source branch,
but it asks a different question: can source training directly shape additive
readout geometry?

## Implementation

Updated `scripts/overfit_one_batch.py`:

- Added `TrainConfig.additive_forced_true_loss_weight`.
- Added `temporary_calculator_bottleneck_mode`.
- Added `additive_forced_true_result_loss`.
- Added CLI/config/metrics/curve logging/run-name support for
  `--additive-forced-true-loss-weight`.
- Rejected negative weights.

Smoke:

```text
python3 -m py_compile scripts/overfit_one_batch.py

python3 scripts/overfit_one_batch.py \
  --variant model-c --digits 2 --steps 2 --operand-max 3 \
  --exhaustive-grid-batch --calculator-operand-vocab-size 20 \
  --calculator-estimator direct_feedback_alignment \
  --calculator-action-head result_space \
  --calculator-read-position operand_spans --calculator-read-span-width 2 \
  --calculator-injection-mode add \
  --calculator-bottleneck-mode answer_decoder \
  --calculator-output-format sum --answer-decoder-interaction product \
  --semantic-decoder-checkpoint runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/stage0_candidates/tiny_operand_spans_dense/oracle_train/2026-05-12_113346_842566_model-c-oracle-op0-19-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00500_weights.pt \
  --semantic-decoder-checkpoint-load-scope semantic_decoder_only \
  --freeze-semantic-decoder --answer-loss-weight 0 \
  --result-policy-improvement-assignment-weight 1 \
  --result-policy-entropy-weight 0.01 \
  --result-policy-batch-diversity-weight 0.01 \
  --result-policy-stabilization-decay-steps 0 \
  --additive-forced-true-loss-weight 0.5 \
  --eval-samples 16 --snapshot-every 0 --checkpoint-every 0 \
  --log-every 1 --seed 101 \
  --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 \
  --calculator-hook-after-layer 1 \
  --run-root runs/2026-05-29_phase7_additive_forced_true_aux/smoke
```

The first smoke attempt failed because the semantic decoder checkpoint was a
tiny `n_embd=16` model while the script default is `n_embd=128`. Re-running with
the checkpoint architecture passed and logged the auxiliary loss/objective.

## Gate

The intended `operand_max=19`, `400`-step gate with in-training additive
handoff probes was too slow on the local MPS runtime, so it was stopped and
reduced to a small mechanism gate:

```text
runs/2026-05-29_phase7_additive_forced_true_aux/small_gate
```

Shared setup:

- `operand_max=9`
- `calculator_operand_vocab_size=20`
- `steps=100`
- same seed `13`
- no-decay source stabilization:
  - result-policy improvement assignment weight `10`
  - entropy `0.05`
  - batch diversity `0.1`
- frozen semantic decoder from the 2026-05-12 product answer-decoder checkpoint
- checkpoints and snapshots at step `100`

Branches:

| Branch | Aux weight | Source calc @100 | Source normal snapshot | Final eval exact |
| --- | ---: | ---: | ---: | ---: |
| baseline | `0.0` | `0.3500` | `0.3400` | `0.3800` |
| additive forced true | `0.5` | `0.2800` | `0.3200` | `0.2800` |

The auxiliary strongly reduced its own forced-true additive loss:

| Step | Aux forced-true loss |
| ---: | ---: |
| `0` | `2.6350` |
| `50` | `1.4473` |
| `100` | `0.8043` |

## Geometry Probe

Then probed both step-100 checkpoints with:

```text
python3 scripts/run_additive_handoff_geometry_probe.py \
  --checkpoint <baseline_step100> <aux_step100> \
  --digits 2 --operand-max 9 --calculator-operand-vocab-size 20 \
  --slope-steps 0,25,50 --seed 13 \
  --output-root runs/2026-05-29_phase7_additive_forced_true_aux/small_gate/geometry_probe
```

Key results:

| Branch | Calc acc | Forced best=true | Forced top3=true | True loss | Best loss | True-best gap | Slope final loss @50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | `0.3500` | `0.0000` | `0.0000` | `2.6463` | `2.6265` | `0.0197` | `1.5305` |
| aux `0.5` | `0.2800` | `0.5900` | `0.6900` | `0.8043` | `0.7881` | `0.0161` | `0.7367` |

## Interpretation

Mixed-positive.

The auxiliary clearly changes additive readout geometry: the true result became
the best forced additive result on `59%` of the small grid, versus `0%` for the
baseline. It also lowered the additive model's initial and 50-step downstream
loss.

However, the same auxiliary weakened source calculator-policy acquisition at
this budget (`0.28` vs `0.35` calc accuracy, `0.28` vs `0.38` final exact). The
naive version appears to split optimization pressure between two useful but
not automatically compatible goals: choose the correct result in the bottleneck
policy, and make the additive readout understand the true result.

## Decision

```text
additive_forced_true_source_aux_mixed_positive_small_gate
```

Do not repeat this exact small `operand_max=9`, `steps=100`, aux-weight `0.5`
gate as novelty.

Allowed next tests:

- Run a full `operand_max=19` gate only with a lighter evaluation plan:
  source-only checkpoints first, then targeted standalone handoff/geometry
  probes.
- Add a schedule or gate so the auxiliary turns on after source policy
  acquisition reaches a threshold instead of competing from step `0`.
- Combine this geometry objective with a policy-retention/anchor term and
  evaluate against the established 600-step additive handoff gate.

