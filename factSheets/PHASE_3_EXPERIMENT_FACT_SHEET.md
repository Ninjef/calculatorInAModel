# Phase 3 Experiment Fact Sheet

## Track A: joint pair-action interface smoke

Date: 2026-05-06

### Setup

- Implemented `calculator_action_head=joint_pair` and `calculator_estimator=action_loss_full_enum_joint_interface`.
- Strict bottleneck was preserved: `digits=2`, `operand_max=19`, operand vocab `20`, `2L/1H/16d/mlp1`, hook after layer `1`, read position `operands`, `calculator_bottleneck_mode=answer_decoder`.
- Training targets used full-enum answer NLL over all `20 x 20` action pairs, converted to a soft pair distribution with temperature `1.0`.
- The joint objective trained pair logits directly and did not marginalize into independent A/B targets.
- Primary claim constraints held: `final_aux_operand_loss_weight=0.0`, `final_input_proj_anchor_weight=0.0`, `freeze_upstream_encoder=true`, and trainable parameters were limited to `calculator_hook.pair_proj`.

### Smoke Run

Run:

```text
runs/2026-05-06_082124_888126_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.0003-uplr0.0003-fullt1-fullchunk64-answer_decoder/model-c-2digit-seed103
```

Start checkpoint:

```text
runs/2026-05-03_114747_070474_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed4/checkpoint_snapshots/step_00550_weights.pt
```

Result summary:

| Metric | Value |
| --- | ---: |
| Final built-in eval exact | 0.04883 |
| Snapshot step-200 normal exact | 0.06250 |
| Snapshot step-200 injection-zero exact | 0.00000 |
| Snapshot step-200 oracle-at-eval exact | 0.92188 |
| Snapshot step-200 pair exact | 0.00000 |
| Snapshot step-200 calc result acc | 0.05469 |
| Snapshot step-200 mean pair entropy | 5.99086 |

### Post-Smoke Diagnostics

Canonical causal:

| Metric | Value |
| --- | ---: |
| Normal exact | 0.04688 |
| Injection-zero exact | 0.00000 |
| Forced-random exact | 0.06250 |
| Oracle-at-eval exact | 0.90625 |
| Pair exact | 0.00000 |
| Result-equivalent pair accuracy | 0.04688 |
| Classification | `calculator_ignored_or_bypassed` |
| Bottleneck label | `strict_bottleneck_unvalidated` |

Full-enum joint diagnostic:

| Metric | Value |
| --- | ---: |
| Best NLL | 0.09967 |
| Learned NLL | 5.72066 |
| True NLL | 0.09967 |
| Learned-minus-true gap | 5.62099 |
| Learned-minus-best gap | 5.62100 |
| Learned-best fraction | 0.00000 |
| Tie-aware learned-best <= 1e-3 | 0.10938 |
| Best result matches true sum | 0.90625 |
| Learned result matches true sum | 0.03125 |
| Teacher effective pairs | 29.21159 |
| Pair-logit effective pairs | 399.75911 |

Private protocol:

| Metric | Value |
| --- | ---: |
| All-pair answer exact | 0.04500 |
| Operand exact | 0.00000 |
| Pair exact | 0.00000 |
| Calculator result accuracy | 0.02750 |
| Learned result distribution | `{"12": 316, "21": 84}` |

### Finding

Useful negative smoke result. The full-enum teacher remains informative: true actions have very low NLL and the true sum is best/result-best often. But the direct joint pair head did not learn the teacher distribution in 200 steps from the selected checkpoint. Its output distribution stayed nearly uniform by entropy while argmax actions collapsed to a small set of result classes. Because pair exact, result-equivalent pair accuracy, private-protocol accuracy, and learned-best fraction did not improve, the selected-checkpoint ladder and Stage-B ladder were not run.

### Recommendation

No-go on continuing this exact Track A implementation to the primary ladder. Move next to Track B: an identifiability curriculum where task structure rewards operand identity directly, so addition-only result equivalence cannot hide incorrect pairs.

## Track B: decayed joint identity curriculum

Date: 2026-05-06

### Setup

- Chose the narrow auxiliary-identity-query design, implemented as decayed true-pair CE on the joint pair-action head rather than adding new prompt tokens. This preserves the existing strict semantic-decoder checkpoint and avoids a vocab migration confound.
- Code change: `scripts/overfit_one_batch.py` now routes `auxiliary_operand_loss` through `calculator_hook.pair_proj` when `calculator_action_head=joint_pair`, with the default detached-upstream path still updating only `pair_proj` and `--aux-operand-loss-grad-upstream` allowing curriculum gradients into the encoder.
- Test change: `tests/test_model.py` adds `test_joint_auxiliary_operand_loss_updates_pair_projection_only`.
- Fixed architecture/bottleneck controls were preserved: `digits=2`, `operand_max=19`, operand vocab `20`, `2L/1H/16d/mlp1`, hook after layer `1`, read position `operands`, `calculator_bottleneck_mode=answer_decoder`, `freeze_semantic_decoder=true`, `answer_loss_weight=1.0`, and `input_proj_anchor_weight=0.0`.
- Verification: `python3 -m pytest tests/test_data.py tests/test_model.py -q` -> `58 passed`.

### Commands

Frozen-upstream pair-head-only smoke:

```text
python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --steps 300 --batch-size 64 --eval-samples 256 --operand-max 19 --calculator-operand-vocab-size 20 --calculator-estimator action_loss_full_enum_joint_interface --calculator-action-head joint_pair --semantic-decoder-checkpoint runs/2026-05-03_114747_070474_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed4/checkpoint_snapshots/step_00550_weights.pt --adaptive-interface-loss-weight 1.0 --input-proj-lr 0.0003 --upstream-lr 0.0003 --freeze-upstream-encoder --calculator-read-position operands --calculator-bottleneck-mode answer_decoder --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --action-loss-full-enum-temperature 1.0 --action-loss-full-enum-chunk-size 64 --aux-operand-loss-weight 1.0 --aux-operand-loss-decay-steps 150 --snapshot-every 50 --checkpoint-every 50 --snapshot-samples 64 --seed 201 --log-every 50
```

Upstream-open identity-curriculum smoke:

```text
python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --steps 300 --batch-size 64 --eval-samples 256 --operand-max 19 --calculator-operand-vocab-size 20 --calculator-estimator action_loss_full_enum_joint_interface --calculator-action-head joint_pair --semantic-decoder-checkpoint runs/2026-05-03_114747_070474_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed4/checkpoint_snapshots/step_00550_weights.pt --adaptive-interface-loss-weight 1.0 --input-proj-lr 0.001 --upstream-lr 0.0003 --calculator-read-position operands --calculator-bottleneck-mode answer_decoder --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --action-loss-full-enum-temperature 1.0 --action-loss-full-enum-chunk-size 64 --aux-operand-loss-weight 5.0 --aux-operand-loss-decay-steps 150 --aux-operand-loss-grad-upstream --snapshot-every 50 --checkpoint-every 50 --snapshot-samples 64 --seed 202 --log-every 50
```

Step-150 retained-checkpoint diagnostics:

```text
python3 -m scripts.run_causal_calculator_protocol_diagnostics --checkpoint runs/2026-05-06_091027_975713_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed204/checkpoint_snapshots/step_00150_weights.pt --samples 256 --digits 2 --operand-max 19 --seed 5204 --forced-result-sweep --forced-result-batch-size 64 --output-dir runs/2026-05-06_091027_975713_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed204/step150_canonical_causal_diagnostics

python3 scripts/run_full_enum_action_loss_diagnostic.py --checkpoint runs/2026-05-06_091027_975713_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed204/checkpoint_snapshots/step_00150_weights.pt --samples 128 --batch-size 64 --digits 2 --operand-max 19 --temperature 1.0 --chunk-size 64 --seed 6204 --output-root runs/2026-05-06_091027_975713_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed204/step150_full_enum_action_loss

python3 scripts/diagnose_private_protocol.py --checkpoint runs/2026-05-06_091027_975713_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed204/checkpoint_snapshots/step_00150_weights.pt --digits 2 --operand-max 19 --seed 7204 --output-dir runs/2026-05-06_091027_975713_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed204/step150_private_protocol_diagnostics
```

### Runs

| Variant | Run | Selected checkpoint | Aux at selected | Final aux | Final anchor | Trainable groups |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Frozen upstream | `runs/2026-05-06_090608_992797_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.0003-uplr0.0003-fullt1-fullchunk64-answer_decoder-aux1-auxdecay150/model-c-2digit-seed203` | final step 300 | `0.0` | `0.0` | `0.0` | `calculator_hook.pair_proj` |
| Upstream-open | `runs/2026-05-06_091027_975713_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed204` | `checkpoint_snapshots/step_00150_weights.pt` | `0.0` | `0.0` | `0.0` | `calculator_hook.pair_proj`, `upstream` |

The frozen-upstream run was a useful negative: aux CE stayed near `log(400)`, pair entropy stayed near-uniform, final pair exact was `0.0156`, and final eval exact was `0.0234`. Operand identity was not linearly recoverable from the frozen Track-A readout by the pair head alone.

### Snapshot Selection

Upstream-open snapshots:

| Step | Aux weight | Normal | Injection-zero | Oracle | Pair exact | Calc result acc |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `5.0000` | `0.0781` | `0.0000` | `0.9219` | `0.0000` | `0.0781` |
| 50 | `3.3333` | `0.1250` | `0.0000` | `0.9063` | `0.0781` | `0.1250` |
| 100 | `1.6667` | `0.2969` | `0.0000` | `0.9531` | `0.2500` | `0.3125` |
| 150 | `0.0000` | `0.3906` | `0.0000` | `0.9063` | `0.2188` | `0.3906` |
| 200 | `0.0000` | `0.1563` | `0.0156` | `0.9063` | `0.1563` | `0.1563` |
| 250 | `0.0000` | `0.0313` | `0.0156` | `0.9531` | `0.0313` | `0.0313` |
| 300 | `0.0000` | `0.0625` | `0.0156` | `0.9219` | `0.0313` | `0.0625` |

The retained signal peaked exactly at the aux-zero handoff and then degraded, so the primary diagnostic checkpoint is step 150, not the final step 300.

### Step-150 Canonical Causal

| Metric | Value |
| --- | ---: |
| Normal exact | `0.2500` |
| Injection-zero exact | `0.0078` |
| Forced-zero exact | `0.0078` |
| Forced-random exact | `0.0156` |
| Oracle-at-eval exact | `0.9414` |
| Pair exact | `0.2305` |
| Result-equivalent pair accuracy | `0.2813` |
| Calculator-result accuracy | `0.2813` |
| Mean pair entropy | `5.9888` |
| Mean pair confidence | `0.0031` |
| Classification | `causally_useful_opaque_private_code` |
| Bottleneck label | `strict_bottleneck_unvalidated` |

Read-vector corruptions reduced answer exact to `0.0352` for A and `0.0195` for B, while swap interventions were no-ops in this setup.

### Step-150 Full-Enum Action Loss

| Metric | Value |
| --- | ---: |
| Best NLL | `0.0534` |
| Learned NLL | `5.2831` |
| True NLL | `0.0534` |
| Learned-minus-true gap | `5.2297` |
| Learned-minus-best gap | `5.2297` |
| Learned-best fraction | `0.0234` |
| Tie-aware learned-best <= 1e-3 | `0.2891` |
| Best result matches true sum | `0.9297` |
| Learned result matches true sum | `0.2891` |
| Teacher effective pairs | `30.5096` |
| Pair-logit effective pairs | `398.9657` |

### Step-150 Private Protocol

| Metric | Value |
| --- | ---: |
| All-pair answer exact | `0.2400` |
| Operand exact | `0.1925` |
| Pair exact | `0.1925` |
| Calculator-result accuracy | `0.2575` |
| Majority-mapped operand exact | `0.1975` |
| Majority-mapped calculator-result accuracy | `0.2425` |
| Result-code majority accuracy | `0.3025` |

Group behavior:

| Group | Count | Answer exact | Operand exact | Calc result acc |
| --- | ---: | ---: | ---: | ---: |
| all | 400 | `0.2400` | `0.1925` | `0.2575` |
| carry | 345 | `0.2406` | `0.1855` | `0.2609` |
| no-carry | 55 | `0.2364` | `0.2364` | `0.2364` |
| large operand | 300 | `0.2500` | `0.1800` | `0.2667` |
| small operands | 100 | `0.2100` | `0.2300` | `0.2300` |
| symmetric | 20 | `0.3500` | `0.3500` | `0.3500` |

### Comparison

Against the Track A joint smoke, step-150 improves on all required Track-B positive gates:

| Metric | Joint smoke | Step-150 identity curriculum |
| --- | ---: | ---: |
| Canonical pair exact | `0.0000` | `0.2305` |
| Canonical result-equivalent pair acc | `0.0469` | `0.2813` |
| Full-enum learned-minus-true gap | `5.6210` | `5.2297` |
| Full-enum learned-best fraction | `0.0000` | `0.0234` |
| Private all-pair answer exact | `0.0450` | `0.2400` |
| Private pair exact | `0.0000` | `0.1925` |
| Private calculator-result accuracy | `0.0275` | `0.2575` |

Against the best Phase 2 checkpoints, this is not yet competitive. Phase 2 selected/self-training checkpoints reached private all-pair answer around `0.5300..0.5375`, operand exact around `0.5500..0.5675`, calculator-result accuracy around `0.5750..0.5775`, and best learned-minus-true gaps around `1.9814..2.5646`. The Track-B retained checkpoint is therefore a positive result versus Track A, but not a new project-best protocol.

### Finding

Mixed positive. The curriculum proves that the joint pair interface can retain nonzero true-pair structure after identity pressure reaches exactly `0.0`, and it remains calculator-dependent under injection-zero/forced-random/oracle controls. However, the protocol is diffuse: pair-logit effective action count is still almost all `400` pairs, confidence is near uniform, and the retained signal degrades quickly after the aux-zero handoff.

### Recommendation

Go for one follow-up Track-B/C bridge, not a broad ladder yet: repeat the upstream-open curriculum across seeds with dense checkpoint selection around the aux-zero handoff, and add a retention stabilizer or slower aux decay only if the retained checkpoint remains calculator-dependent with aux exactly `0.0`. No-go on frozen-upstream pair-head-only identity curriculum.

## Track B: interface-loss decay scheduler calibration

Date: 2026-05-06

### Implementation

- Added `--adaptive-interface-loss-decay-steps` and `--adaptive-interface-loss-floor` to `scripts/overfit_one_batch.py`.
- The scheduled weight reuses `--adaptive-interface-loss-weight` as the initial value and mirrors aux-weight semantics: initial `<= 0` stays `0`, decay steps `<= 0` preserves old constant behavior, otherwise weight decays linearly to the floor.
- The scheduled weight now multiplies both `adaptive_interface` and action-loss/full-enum interface objectives.
- `training_curve.csv` logs `adaptive_interface_loss_weight`; `metrics.json` records `adaptive_interface_loss_decay_steps`, `adaptive_interface_loss_floor`, and `final_adaptive_interface_loss_weight`.
- Verification: `python3 -m pytest tests/test_data.py tests/test_model.py -q` -> `59 passed`.

### One-Seed Matched Calibration

The full six-run compact ladder is not completed here; one matched seed pair was run to validate the mechanism and calibrate runtime. Each 225-step full-enum run took about five and a half minutes on MPS.

Constant full-enum interface objective:

```text
runs/2026-05-06_093654_713217_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed213
```

Interface objective decayed to zero with aux at step 150:

```text
runs/2026-05-06_094254_634251_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-ifacedecay150-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed213
```

### Snapshot Comparison

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

### Finding

The scheduler works and preserves old behavior when omitted. In the one matched seed, decaying the underidentified full-enum interface objective to zero did not improve the exact aux-zero handoff at step 150, but it did avoid the severe step-225 collapse seen in the constant-interface run and produced a better step-200 retained snapshot. This is calibration only, not a strong positive: pair entropy remained near uniform, and canonical pair exact stayed below the task threshold.

### Recommendation

Complete the intended three-seed Stage 1/Stage 2 ladder before deciding on the slower-aux-decay stabilizer. Use dense selection over steps 125, 150, 175, 200, and 225, and run the full canonical/private/action diagnostics only on the selected retained checkpoints.

## Track B/C: matched retention ladder for interface decay

Date: 2026-05-06

### Setup

- Completed the matched three-seed ladder for the joint identity curriculum under the fixed phase-3 controls.
- Seed arguments were `211`, `221`, and `231`; `overfit_one_batch.py` stores run seeds as `seed + digits`, so run directories are `seed213`, `seed223`, and `seed233`.
- Each seed has a constant full-enum interface run and a matched interface-decay run with `--adaptive-interface-loss-decay-steps 150 --adaptive-interface-loss-floor 0.0`.
- Selection used steps `125, 150, 175, 200, 225`, maximizing snapshot `pair_exact_match` among aux-zero rows; decayed runs additionally required snapshot interface weight exactly `0.0`.
- Summary artifacts: `runs/2026-05-06_phase3_matched_retention_ladder_summary/summary.json` and `summary.md`.
- Added helper scripts:
  - `scripts/summarize_matched_retention_ladder.py`
  - `scripts/run_matched_retention_ladder_diagnostics.py`
- Verification: `python3 -m pytest tests/test_data.py tests/test_model.py -q` -> `61 passed`.

### Selected Snapshot Checkpoints

| Seed arg | Stored seed | Condition | Step | Interface | Aux | Normal | Inj0 | Rand | Oracle | Pair | Calc |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 211 | 213 | constant | 175 | `1.0000` | `0.0000` | `0.1641` | `0.0156` | `0.0078` | `0.9297` | `0.1406` | `0.1719` |
| 211 | 213 | decayed | 200 | `0.0000` | `0.0000` | `0.2344` | `0.0078` | `0.0391` | `0.9609` | `0.1797` | `0.2422` |
| 221 | 223 | constant | 150 | `1.0000` | `0.0000` | `0.1562` | `0.0078` | `0.0156` | `0.9141` | `0.0938` | `0.1562` |
| 221 | 223 | decayed | 150 | `0.0000` | `0.0000` | `0.2031` | `0.0078` | `0.0156` | `0.9141` | `0.1484` | `0.2109` |
| 231 | 233 | constant | 150 | `1.0000` | `0.0000` | `0.1562` | `0.0000` | `0.0234` | `0.8906` | `0.1484` | `0.2109` |
| 231 | 233 | decayed | 200 | `0.0000` | `0.0000` | `0.1562` | `0.0000` | `0.0156` | `0.9375` | `0.1406` | `0.1719` |

Aggregate selected snapshots:

| Condition | Runs | Mean pair exact | Mean calc result acc | Mean normal | Mean inj0 | Mean rand | Mean oracle |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| constant | 3 | `0.1276` | `0.1797` | `0.1589` | `0.0078` | `0.0156` | `0.9115` |
| decayed | 3 | `0.1562` | `0.2083` | `0.1979` | `0.0052` | `0.0234` | `0.9375` |

### Full Diagnostics

All selected checkpoints received the canonical causal diagnostics, full-enum action-loss diagnostics, and private all-pair protocol diagnostics.

| Seed arg | Condition | Step | Built-in | Canon normal | Canon inj0 | Canon rand | Canon oracle | Canon pair | Private pair | Private calc | Learned-true gap | Tie <=1e-3 | Logit eff pairs |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 211 | constant | 175 | `0.0625` | `0.1133` | `0.0078` | `0.0352` | `0.9336` | `0.0625` | `0.0975` | `0.1500` | `8.0278` | `0.1094` | `396.7` |
| 211 | decayed | 200 | `0.1016` | `0.1367` | `0.0078` | `0.0234` | `0.9219` | `0.1133` | `0.1000` | `0.1525` | `7.4498` | `0.2188` | `396.4` |
| 221 | constant | 150 | `0.0703` | `0.1914` | `0.0078` | `0.0195` | `0.9609` | `0.1055` | `0.0900` | `0.1575` | `7.1521` | `0.1484` | `397.9` |
| 221 | decayed | 150 | `0.1250` | `0.1484` | `0.0000` | `0.0430` | `0.8789` | `0.1133` | `0.1175` | `0.1775` | `7.8160` | `0.1484` | `397.7` |
| 231 | constant | 150 | `0.0664` | `0.1992` | `0.0117` | `0.0156` | `0.9258` | `0.1875` | `0.1425` | `0.1825` | `7.7091` | `0.1797` | `397.8` |
| 231 | decayed | 200 | `0.1602` | `0.1289` | `0.0039` | `0.0195` | `0.9570` | `0.1172` | `0.1375` | `0.1700` | `7.7719` | `0.1484` | `397.2` |

All selected runs had `final_aux_operand_loss_weight=0.0`, `final_input_proj_anchor_weight=0.0`, `freeze_semantic_decoder=true`, `freeze_upstream_encoder=false`, and trainable groups `calculator_hook.pair_proj` plus `upstream`. For decayed selections, the selected snapshot interface weight and `final_adaptive_interface_loss_weight` were both exactly `0.0`.

### Finding

Weak positive, not strong positive. Decay improved selected snapshot retention on seed args `211` and `221`, and the decayed aggregate beat constant on selected pair exact and calculator-result accuracy. The causal controls remain mostly healthy, with injection-zero/forced-random near chance and oracle recovery high in snapshots. However, full diagnostics are less flattering: no decayed seed approaches the strong-positive thresholds, pair-logit effective pairs remain near-uniform at roughly all `400` actions, and the `221` decayed checkpoint's 256-sample oracle recovery fell below `0.90`.

### Recommendation

Do not broaden this curriculum. If continuing it at all, run exactly the specified slower-decay stabilizer on the two best seed args only, `211` and `221`:

```text
--aux-operand-loss-decay-steps 300
--adaptive-interface-loss-decay-steps 300
--steps 375
```

If that one stabilizer does not materially improve private pair exact and private calculator-result accuracy, move to a sharper identifiability environment where the answer signal itself identifies operands.
