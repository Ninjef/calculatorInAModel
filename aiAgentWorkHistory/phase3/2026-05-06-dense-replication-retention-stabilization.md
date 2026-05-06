# Dense Replication and Retention Stabilization

Date: 2026-05-06

## Task

Added the smallest scheduler needed to test whether post-handoff full-enum interface training washes out the Track B joint identity curriculum after the auxiliary identity pressure reaches exactly `0.0`.

## Code Changes

- Added `--adaptive-interface-loss-decay-steps`.
- Added `--adaptive-interface-loss-floor`.
- Added `adaptive_interface_weight`, mirroring `auxiliary_operand_weight` semantics.
- Applied the scheduled weight to both `adaptive_interface` and action-loss/full-enum interface objectives.
- Logged `adaptive_interface_loss_weight` in `training_curve.csv`.
- Saved `adaptive_interface_loss_decay_steps`, `adaptive_interface_loss_floor`, and `final_adaptive_interface_loss_weight` in `metrics.json`.
- Preserved old behavior when the new decay args are omitted.

## Tests

```text
python3 -m pytest tests/test_data.py tests/test_model.py -q
59 passed in 2.55s
```

Added tests for:

- interface-weight schedule helper behavior, including decay to exactly `0.0`;
- CLI/config/metrics/CSV smoke coverage showing the scheduled interface weight reaches exactly `0.0`.

## Smoke Verification

Tiny joint full-enum decay smoke:

```text
python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --steps 1 --batch-size 4 --eval-samples 4 --operand-max 19 --calculator-operand-vocab-size 20 --calculator-estimator action_loss_full_enum_joint_interface --calculator-action-head joint_pair --semantic-decoder-checkpoint runs/2026-05-03_114747_070474_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed4/checkpoint_snapshots/step_00550_weights.pt --adaptive-interface-loss-weight 1.0 --adaptive-interface-loss-decay-steps 1 --adaptive-interface-loss-floor 0.0 --input-proj-lr 0.001 --upstream-lr 0.0003 --calculator-read-position operands --calculator-bottleneck-mode answer_decoder --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --action-loss-full-enum-temperature 1.0 --action-loss-full-enum-chunk-size 64 --aux-operand-loss-weight 5.0 --aux-operand-loss-decay-steps 1 --aux-operand-loss-grad-upstream --snapshot-every 1 --checkpoint-every 1 --snapshot-samples 4 --seed 9001 --log-every 1 --run-root /private/tmp/calculatorInAModel_phase3_decay_smoke
```

Confirmed `final_adaptive_interface_loss_weight=0.0` and `final_aux_operand_loss_weight=0.0`.

## Calibration Runs

The full Stage 1/Stage 2 three-seed ladder was not completed in this turn. A single matched seed pair was run because each 225-step full-enum run took roughly five and a half minutes.

Stage 1 constant full-enum interface objective:

```text
runs/2026-05-06_093654_713217_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed213
```

Stage 2 interface objective decayed to zero at the aux handoff:

```text
runs/2026-05-06_094254_634251_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-ifacedecay150-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed213
```

## Snapshot Trajectories

| Variant | Step | Interface weight | Aux weight | Normal | Injection-zero | Forced-random | Oracle | Pair exact | Calc result acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Constant interface | 125 | `1.0000` | `0.8333` | `0.1953` | `0.0078` | `0.0234` | `0.9375` | `0.1406` | `0.2188` |
| Constant interface | 150 | `1.0000` | `0.0000` | `0.1484` | `0.0000` | `0.0234` | `0.8906` | `0.1172` | `0.1719` |
| Constant interface | 175 | `1.0000` | `0.0000` | `0.1641` | `0.0156` | `0.0078` | `0.9297` | `0.1406` | `0.1719` |
| Constant interface | 200 | `1.0000` | `0.0000` | `0.1172` | `0.0078` | `0.0391` | `0.9609` | `0.1094` | `0.1250` |
| Constant interface | 225 | `1.0000` | `0.0000` | `0.0547` | `0.0156` | `0.0078` | `0.9531` | `0.0156` | `0.0547` |
| Decayed interface | 125 | `0.1667` | `0.8333` | `0.2031` | `0.0078` | `0.0234` | `0.9375` | `0.1406` | `0.2109` |
| Decayed interface | 150 | `0.0000` | `0.0000` | `0.1250` | `0.0000` | `0.0234` | `0.8906` | `0.1172` | `0.1484` |
| Decayed interface | 175 | `0.0000` | `0.0000` | `0.1484` | `0.0156` | `0.0078` | `0.9297` | `0.1094` | `0.1484` |
| Decayed interface | 200 | `0.0000` | `0.0000` | `0.2344` | `0.0078` | `0.0391` | `0.9609` | `0.1797` | `0.2422` |
| Decayed interface | 225 | `0.0000` | `0.0000` | `0.1484` | `0.0156` | `0.0078` | `0.9531` | `0.1094` | `0.1641` |

Both runs had `final_aux_operand_loss_weight=0.0`, `final_input_proj_anchor_weight=0.0`, `freeze_semantic_decoder=true`, `freeze_upstream_encoder=false`, and trainable groups `calculator_hook.pair_proj` plus `upstream`.

## Finding

The scheduler is ready for the intended replication ladder. On this single matched seed, interface decay did not improve the exact aux-zero checkpoint at step 150, but it did produce a better retained step-200 snapshot and avoided the sharp step-225 collapse of the constant-interface run. This is a calibration result only; it does not satisfy the strong-positive thresholds and should not be treated as a replicated finding.

## Next Step

Complete the remaining Stage 1/Stage 2 seeds and select checkpoints over steps 125, 150, 175, 200, and 225. Only then decide whether to run the slower aux-decay stabilizer.
