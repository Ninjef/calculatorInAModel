# Identifiability Curriculum After Joint-Smoke Collapse

Date: 2026-05-06

## Task

Started Track B after the Track A joint pair-action smoke collapsed to near-uniform pair logits and zero pair exact. The selected curriculum was a narrow auxiliary identity query implemented directly on the joint pair-action interface: during curriculum, train the pair head toward the true `(a,b)` action; after the weight decays to exactly `0.0`, evaluate whether any pair protocol is retained under the strict addition bottleneck.

This deliberately avoided adding prompt tokens or changing the vocabulary, because that would have invalidated the existing strict semantic-decoder checkpoint and introduced a decoder migration confound.

## Code Changes

- Updated `scripts/overfit_one_batch.py` so `auxiliary_operand_loss` supports `calculator_action_head=joint_pair`.
- In the default detached-upstream path, joint aux loss computes read vectors without gradient into the encoder and trains only `calculator_hook.pair_proj`.
- With `--aux-operand-loss-grad-upstream`, joint aux loss uses `calculator_read_pair_logits`, allowing curriculum gradients into both `pair_proj` and upstream encoder parameters.
- Added `tests/test_model.py::test_joint_auxiliary_operand_loss_updates_pair_projection_only`.

Verification:

```text
python3 -m pytest tests/test_data.py tests/test_model.py -q
58 passed in 3.36s
```

## Runs

Frozen-upstream pair-head-only smoke:

```text
python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --steps 300 --batch-size 64 --eval-samples 256 --operand-max 19 --calculator-operand-vocab-size 20 --calculator-estimator action_loss_full_enum_joint_interface --calculator-action-head joint_pair --semantic-decoder-checkpoint runs/2026-05-03_114747_070474_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed4/checkpoint_snapshots/step_00550_weights.pt --adaptive-interface-loss-weight 1.0 --input-proj-lr 0.0003 --upstream-lr 0.0003 --freeze-upstream-encoder --calculator-read-position operands --calculator-bottleneck-mode answer_decoder --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --action-loss-full-enum-temperature 1.0 --action-loss-full-enum-chunk-size 64 --aux-operand-loss-weight 1.0 --aux-operand-loss-decay-steps 150 --snapshot-every 50 --checkpoint-every 50 --snapshot-samples 64 --seed 201 --log-every 50
```

Run path:

```text
runs/2026-05-06_090608_992797_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.0003-uplr0.0003-fullt1-fullchunk64-answer_decoder-aux1-auxdecay150/model-c-2digit-seed203
```

Result: useful negative. Final eval exact `0.0234`, final pair exact `0.0156`, aux CE stayed near `log(400)`, and pair entropy stayed nearly uniform. The frozen Track-A readout was not enough for the pair head alone to learn operand identity.

Upstream-open identity-curriculum smoke:

```text
python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --steps 300 --batch-size 64 --eval-samples 256 --operand-max 19 --calculator-operand-vocab-size 20 --calculator-estimator action_loss_full_enum_joint_interface --calculator-action-head joint_pair --semantic-decoder-checkpoint runs/2026-05-03_114747_070474_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed4/checkpoint_snapshots/step_00550_weights.pt --adaptive-interface-loss-weight 1.0 --input-proj-lr 0.001 --upstream-lr 0.0003 --calculator-read-position operands --calculator-bottleneck-mode answer_decoder --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --action-loss-full-enum-temperature 1.0 --action-loss-full-enum-chunk-size 64 --aux-operand-loss-weight 5.0 --aux-operand-loss-decay-steps 150 --aux-operand-loss-grad-upstream --snapshot-every 50 --checkpoint-every 50 --snapshot-samples 64 --seed 202 --log-every 50
```

Run path:

```text
runs/2026-05-06_091027_975713_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed204
```

Selected retained checkpoint:

```text
runs/2026-05-06_091027_975713_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed204/checkpoint_snapshots/step_00150_weights.pt
```

At step 150, the aux weight had decayed to exactly `0.0`. The overall run also has `final_aux_operand_loss_weight=0.0`, `final_input_proj_anchor_weight=0.0`, `freeze_semantic_decoder=true`, `freeze_upstream_encoder=false`, and trainable groups `calculator_hook.pair_proj` plus `upstream`.

## Diagnostics

Canonical causal command:

```text
python3 -m scripts.run_causal_calculator_protocol_diagnostics --checkpoint runs/2026-05-06_091027_975713_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed204/checkpoint_snapshots/step_00150_weights.pt --samples 256 --digits 2 --operand-max 19 --seed 5204 --forced-result-sweep --forced-result-batch-size 64 --output-dir runs/2026-05-06_091027_975713_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed204/step150_canonical_causal_diagnostics
```

Canonical causal summary:

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
| Classification | `causally_useful_opaque_private_code` |
| Bottleneck | `strict_bottleneck_unvalidated` |

Full-enum action diagnostic command:

```text
python3 scripts/run_full_enum_action_loss_diagnostic.py --checkpoint runs/2026-05-06_091027_975713_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed204/checkpoint_snapshots/step_00150_weights.pt --samples 128 --batch-size 64 --digits 2 --operand-max 19 --temperature 1.0 --chunk-size 64 --seed 6204 --output-root runs/2026-05-06_091027_975713_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed204/step150_full_enum_action_loss
```

Full-enum summary:

| Metric | Value |
| --- | ---: |
| Best NLL | `0.0534` |
| Learned NLL | `5.2831` |
| True NLL | `0.0534` |
| Learned-minus-true gap | `5.2297` |
| Learned-best fraction | `0.0234` |
| Tie-aware learned-best <= 1e-3 | `0.2891` |
| Learned result matches true sum | `0.2891` |
| Teacher effective pairs | `30.5096` |
| Pair-logit effective pairs | `398.9657` |

Private protocol command:

```text
python3 scripts/diagnose_private_protocol.py --checkpoint runs/2026-05-06_091027_975713_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed204/checkpoint_snapshots/step_00150_weights.pt --digits 2 --operand-max 19 --seed 7204 --output-dir runs/2026-05-06_091027_975713_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.001-uplr0.0003-fullt1-fullchunk64-answer_decoder-aux5-auxdecay150/model-c-2digit-seed204/step150_private_protocol_diagnostics
```

Private summary:

| Metric | Value |
| --- | ---: |
| All-pair answer exact | `0.2400` |
| Operand exact | `0.1925` |
| Pair exact | `0.1925` |
| Calculator-result accuracy | `0.2575` |
| Majority-mapped operand exact | `0.1975` |
| Result-code majority accuracy | `0.3025` |

## Comparison

The retained checkpoint beats the Track A joint smoke on canonical pair exact (`0.2305` vs `0.0000`), result-equivalent pair accuracy (`0.2813` vs `0.0469`), full-enum learned-minus-true gap (`5.2297` vs `5.6210`), learned-best fraction (`0.0234` vs `0.0000`), private all-pair answer exact (`0.2400` vs `0.0450`), private pair exact (`0.1925` vs `0.0000`), and private calculator-result accuracy (`0.2575` vs `0.0275`).

It does not beat the best Phase 2 checkpoints. Phase 2 selected/self-training checkpoints reached private all-pair answer around `0.5300..0.5375`, operand exact around `0.5500..0.5675`, calculator-result accuracy around `0.5750..0.5775`, and learned-minus-true gaps around `1.9814..2.5646`.

## Recommendation

This is a mixed positive. The identity curriculum can create retained joint pair structure after aux reaches exactly zero, and the retained behavior is calculator-dependent. But the learned pair distribution is still diffuse and the signal degrades after the zero-supervision handoff.

Recommended next move: one small replication ladder around this upstream-open curriculum with dense checkpoint selection near the aux-zero handoff. Do not continue frozen-upstream pair-head-only identity curriculum. Do not claim a project-best protocol until a retained checkpoint approaches Phase 2 private-protocol strength while preserving the stricter joint-interface interpretation.
