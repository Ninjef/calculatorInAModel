# Matched Retention Ladder for Joint Identity Curriculum

Date: 2026-05-06

## Task

Completed the matched three-seed constant-vs-decayed interface ladder for the phase-3 joint identity curriculum. The goal was to test whether turning off the underidentified full-enum interface objective at the aux-zero handoff reproducibly preserves retained joint identity structure.

Note on seed naming: the command-line seed arguments were `211`, `221`, and `231`; `scripts/overfit_one_batch.py` stores `seed = args.seed + num_digits`, so run directories are `seed213`, `seed223`, and `seed233`.

## Code Changes

- Added `scripts/summarize_matched_retention_ladder.py` to read `diagnostic_snapshots.csv`, `training_curve.csv`, and `metrics.json`, apply the task selection rule, and emit JSON/Markdown summary artifacts.
- Added `scripts/run_matched_retention_ladder_diagnostics.py` to run the canonical diagnostic stack for selected checkpoints, with `--skip-existing` support for interrupted runs.
- Added focused tests for aux-zero/interface-zero selection behavior in `tests/test_model.py`.

## Verification

```text
python3 -m pytest tests/test_data.py tests/test_model.py -q
61 passed in 2.94s
```

## Runs

| Seed arg | Stored seed | Condition | Run |
| ---: | ---: | --- | --- |
| 211 | 213 | constant | `runs/2026-05-06_093654_713217_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed213` |
| 211 | 213 | decayed | `runs/2026-05-06_094254_634251_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-ifacedecay150-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed213` |
| 221 | 223 | constant | `runs/2026-05-06_102008_954535_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed223` |
| 221 | 223 | decayed | `runs/2026-05-06_102830_781108_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-ifacedecay150-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed223` |
| 231 | 233 | constant | `runs/2026-05-06_103642_800215_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed233` |
| 231 | 233 | decayed | `runs/2026-05-06_104421_920658_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-ifacedecay150-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed233` |

Summary artifacts:

```text
runs/2026-05-06_phase3_matched_retention_ladder_summary/summary.json
runs/2026-05-06_phase3_matched_retention_ladder_summary/summary.md
```

## Selected Checkpoints

Selection window was steps `125, 150, 175, 200, 225`. Constant runs required `aux_operand_loss_weight == 0.0`; decayed runs also required `adaptive_interface_loss_weight == 0.0`. Tie-breakers were lower injection-zero, lower forced-random, higher oracle, then higher calculator-result accuracy.

| Seed arg | Condition | Step | Interface | Aux | Normal | Inj0 | Rand | Oracle | Pair | Calc | Checkpoint |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 211 | constant | 175 | `1.0000` | `0.0000` | `0.1641` | `0.0156` | `0.0078` | `0.9297` | `0.1406` | `0.1719` | `runs/2026-05-06_093654_713217_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed213/checkpoint_snapshots/step_00175_weights.pt` |
| 211 | decayed | 200 | `0.0000` | `0.0000` | `0.2344` | `0.0078` | `0.0391` | `0.9609` | `0.1797` | `0.2422` | `runs/2026-05-06_094254_634251_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-ifacedecay150-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed213/checkpoint_snapshots/step_00200_weights.pt` |
| 221 | constant | 150 | `1.0000` | `0.0000` | `0.1562` | `0.0078` | `0.0156` | `0.9141` | `0.0938` | `0.1562` | `runs/2026-05-06_102008_954535_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed223/checkpoint_snapshots/step_00150_weights.pt` |
| 221 | decayed | 150 | `0.0000` | `0.0000` | `0.2031` | `0.0078` | `0.0156` | `0.9141` | `0.1484` | `0.2109` | `runs/2026-05-06_102830_781108_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-ifacedecay150-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed223/checkpoint_snapshots/step_00150_weights.pt` |
| 231 | constant | 150 | `1.0000` | `0.0000` | `0.1562` | `0.0000` | `0.0234` | `0.8906` | `0.1484` | `0.2109` | `runs/2026-05-06_103642_800215_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed233/checkpoint_snapshots/step_00150_weights.pt` |
| 231 | decayed | 200 | `0.0000` | `0.0000` | `0.1562` | `0.0000` | `0.0156` | `0.9375` | `0.1406` | `0.1719` | `runs/2026-05-06_104421_920658_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-ifacedecay150-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed233/checkpoint_snapshots/step_00200_weights.pt` |

Aggregate selected snapshots:

| Condition | Runs | Mean pair | Mean calc | Mean normal | Mean inj0 | Mean rand | Mean oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| constant | 3 | `0.1276` | `0.1797` | `0.1589` | `0.0078` | `0.0156` | `0.9115` |
| decayed | 3 | `0.1562` | `0.2083` | `0.1979` | `0.0052` | `0.0234` | `0.9375` |

## Snapshot Trajectories

Seed arg `211`:

| Cond | Step | Iface | Aux | Normal | Inj0 | Rand | Oracle | Pair | Calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| constant | 125 | `1.0000` | `0.8333` | `0.1953` | `0.0078` | `0.0234` | `0.9375` | `0.1406` | `0.2188` |
| constant | 150 | `1.0000` | `0.0000` | `0.1484` | `0.0000` | `0.0234` | `0.8906` | `0.1172` | `0.1719` |
| constant | 175 | `1.0000` | `0.0000` | `0.1641` | `0.0156` | `0.0078` | `0.9297` | `0.1406` | `0.1719` |
| constant | 200 | `1.0000` | `0.0000` | `0.1172` | `0.0078` | `0.0391` | `0.9609` | `0.1094` | `0.1250` |
| constant | 225 | `1.0000` | `0.0000` | `0.0547` | `0.0156` | `0.0078` | `0.9531` | `0.0156` | `0.0547` |
| decayed | 125 | `0.1667` | `0.8333` | `0.2031` | `0.0078` | `0.0234` | `0.9375` | `0.1406` | `0.2109` |
| decayed | 150 | `0.0000` | `0.0000` | `0.1250` | `0.0000` | `0.0234` | `0.8906` | `0.1172` | `0.1484` |
| decayed | 175 | `0.0000` | `0.0000` | `0.1484` | `0.0156` | `0.0078` | `0.9297` | `0.1094` | `0.1484` |
| decayed | 200 | `0.0000` | `0.0000` | `0.2344` | `0.0078` | `0.0391` | `0.9609` | `0.1797` | `0.2422` |
| decayed | 225 | `0.0000` | `0.0000` | `0.1484` | `0.0156` | `0.0078` | `0.9531` | `0.1094` | `0.1641` |

Seed arg `221`:

| Cond | Step | Iface | Aux | Normal | Inj0 | Rand | Oracle | Pair | Calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| constant | 125 | `1.0000` | `0.8333` | `0.1719` | `0.0000` | `0.0391` | `0.9297` | `0.1406` | `0.1875` |
| constant | 150 | `1.0000` | `0.0000` | `0.1562` | `0.0078` | `0.0156` | `0.9141` | `0.0938` | `0.1562` |
| constant | 175 | `1.0000` | `0.0000` | `0.0781` | `0.0078` | `0.0312` | `0.9141` | `0.0312` | `0.0781` |
| constant | 200 | `1.0000` | `0.0000` | `0.0625` | `0.0078` | `0.0234` | `0.9531` | `0.0156` | `0.0625` |
| constant | 225 | `1.0000` | `0.0000` | `0.0703` | `0.0000` | `0.0312` | `0.9375` | `0.0078` | `0.0625` |
| decayed | 125 | `0.1667` | `0.8333` | `0.1953` | `0.0000` | `0.0391` | `0.9297` | `0.1562` | `0.2031` |
| decayed | 150 | `0.0000` | `0.0000` | `0.2031` | `0.0078` | `0.0156` | `0.9141` | `0.1484` | `0.2109` |
| decayed | 175 | `0.0000` | `0.0000` | `0.1484` | `0.0078` | `0.0312` | `0.9141` | `0.1094` | `0.1562` |
| decayed | 200 | `0.0000` | `0.0000` | `0.1094` | `0.0078` | `0.0234` | `0.9531` | `0.0391` | `0.1094` |
| decayed | 225 | `0.0000` | `0.0000` | `0.1250` | `0.0000` | `0.0312` | `0.9375` | `0.0938` | `0.1328` |

Seed arg `231`:

| Cond | Step | Iface | Aux | Normal | Inj0 | Rand | Oracle | Pair | Calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| constant | 125 | `1.0000` | `0.8333` | `0.2578` | `0.0000` | `0.0312` | `0.9453` | `0.2109` | `0.2734` |
| constant | 150 | `1.0000` | `0.0000` | `0.1562` | `0.0000` | `0.0234` | `0.8906` | `0.1484` | `0.2109` |
| constant | 175 | `1.0000` | `0.0000` | `0.0938` | `0.0156` | `0.0312` | `0.9453` | `0.0703` | `0.1016` |
| constant | 200 | `1.0000` | `0.0000` | `0.0547` | `0.0000` | `0.0156` | `0.9375` | `0.0391` | `0.0625` |
| constant | 225 | `1.0000` | `0.0000` | `0.0625` | `0.0078` | `0.0234` | `0.9531` | `0.0312` | `0.0625` |
| decayed | 125 | `0.1667` | `0.8333` | `0.2422` | `0.0000` | `0.0312` | `0.9453` | `0.1953` | `0.2578` |
| decayed | 150 | `0.0000` | `0.0000` | `0.1875` | `0.0000` | `0.0234` | `0.8906` | `0.1406` | `0.2109` |
| decayed | 175 | `0.0000` | `0.0000` | `0.1562` | `0.0156` | `0.0312` | `0.9453` | `0.1250` | `0.1719` |
| decayed | 200 | `0.0000` | `0.0000` | `0.1562` | `0.0000` | `0.0156` | `0.9375` | `0.1406` | `0.1719` |
| decayed | 225 | `0.0000` | `0.0000` | `0.1484` | `0.0078` | `0.0234` | `0.9531` | `0.1406` | `0.1875` |

## Full Diagnostic Summary

| Seed arg | Cond | Step | Built-in | Canon normal | Canon inj0 | Canon rand | Canon oracle | Canon pair | Canon calc | Private ans | Private pair | Private calc | L-T gap | L-best | Tie<=1e-3 | Eff pairs |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 211 | constant | 175 | `0.0625` | `0.1133` | `0.0078` | `0.0352` | `0.9336` | `0.0625` | `0.1328` | `0.1350` | `0.0975` | `0.1500` | `8.0278` | `0.0234` | `0.1094` | `396.7` |
| 211 | decayed | 200 | `0.1016` | `0.1367` | `0.0078` | `0.0234` | `0.9219` | `0.1133` | `0.1445` | `0.1425` | `0.1000` | `0.1525` | `7.4498` | `0.0156` | `0.2188` | `396.4` |
| 221 | constant | 150 | `0.0703` | `0.1914` | `0.0078` | `0.0195` | `0.9609` | `0.1055` | `0.1992` | `0.1525` | `0.0900` | `0.1575` | `7.1521` | `0.0000` | `0.1484` | `397.9` |
| 221 | decayed | 150 | `0.1250` | `0.1484` | `0.0000` | `0.0430` | `0.8789` | `0.1133` | `0.1641` | `0.1700` | `0.1175` | `0.1775` | `7.8160` | `0.0000` | `0.1484` | `397.7` |
| 231 | constant | 150 | `0.0664` | `0.1992` | `0.0117` | `0.0156` | `0.9258` | `0.1875` | `0.2266` | `0.1625` | `0.1425` | `0.1825` | `7.7091` | `0.0234` | `0.1797` | `397.8` |
| 231 | decayed | 200 | `0.1602` | `0.1289` | `0.0039` | `0.0195` | `0.9570` | `0.1172` | `0.1406` | `0.1575` | `0.1375` | `0.1700` | `7.7719` | `0.0156` | `0.1484` | `397.2` |

All selected checkpoints had `final_aux_operand_loss_weight=0.0`, `final_input_proj_anchor_weight=0.0`, `freeze_semantic_decoder=true`, `freeze_upstream_encoder=false`, and trainable groups `calculator_hook.pair_proj` and `upstream`. Decayed selections had both selected-step and final interface weights exactly `0.0`.

Private group behavior did not reveal a clean hidden operand protocol: all-pair private pair exact ranged from `0.0900` to `0.1425` in constants and from `0.1000` to `0.1375` in decayed runs. Small/no-carry/symmetric groups moved around by seed, but no group crossed the strong-positive criterion.

## Decision

Weak positive, not strong positive.

The decayed-interface condition beat its matched constant run on selected snapshot pair exact and calculator-result accuracy for seed args `211` and `221`, and decayed aggregates were higher than constant aggregates. This satisfies the weak-positive ladder signal at the snapshot-selection level. It does not satisfy the strong-positive thresholds: no decayed run reached canonical pair exact `0.35`, private pair exact `0.35`, or private calculator-result accuracy `0.40`; pair logits remained nearly uniform with effective pair counts around `397`; and the `221` decayed checkpoint's 256-sample oracle-at-eval fell below `0.90`.

## Recommendation

Do not tune broadly. If continuing this curriculum, run exactly one slower-decay stabilizer on the two best seed args, `211` and `221`:

```text
--aux-operand-loss-decay-steps 300
--adaptive-interface-loss-decay-steps 300
--steps 375
```

If that narrow stabilizer does not materially improve private pair exact and private calculator-result accuracy, stop tuning this auxiliary identity curriculum and move to a sharper identifiability environment where the answer signal itself identifies operands.
